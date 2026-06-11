from src.utils import ttl_classcache
import asyncio
from docker.errors import NotFound, APIError
import uuid
from pathlib import Path
import os
import subprocess as sp
import random
from docker.models.containers import Container
from typing import Union, Dict, Any, List, Tuple
import docker
import logging
from .instances import instance_types, DockerInstance
from .spawn_logic import spawner_scripts
from .config_patterns import GenericDockerConfig, AutoConfig

logger = logging.getLogger(__name__)

DEFAULT_PORT_RANGE = "10240-10340"


USED_MEMORY_THRESHOLD = 100


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv"
    info = sp.check_output(command.split()).decode('ascii').strip().split('\n')
    field_names = info[0].split(", ")
    field_names = [fn.replace("[MiB]", "").replace(".", "_").strip() for fn in field_names]
    rows = {}
    for row in info[1:]:
        row = row.split(", ")
        row_dict = dict(zip(field_names, row))
        for k, v in row_dict.items():
            if 'MiB' in v:
                v = v.removesuffix('MiB')
            row_dict[k] = int(v)
        rows[row_dict['index']] = row_dict

    return rows


def find_gpu_ids(gpu_needed, discard_memory_thr, exclude=None):
    gpu_usage = get_gpu_memory()
    excluded = exclude or set()
    excluded_str = {str(x) for x in excluded}
    vacant_gpu = [str(k) for k, v in gpu_usage.items()
                  if v['memory_used'] < discard_memory_thr and str(k) not in excluded_str]
    if len(vacant_gpu) < gpu_needed:
        return None
    return random.sample(vacant_gpu, gpu_needed)


SERVER_ERROR_PATTERN = "{} Please consult your server administrator."


def make_server_error(error_text):
    return SERVER_ERROR_PATTERN.format(error_text)


class InstanceManager:
    def __init__(
        self, config_directory: str, port_range=DEFAULT_PORT_RANGE,
        max_memory_thr=USED_MEMORY_THRESHOLD, default_idle_time=120,
        api_db=None
    ):
        self._store: Dict[str, DockerInstance] = {}
        self._known_configs: Dict[str,  GenericDockerConfig] = {}
        self._config_stamps: Dict[str, float] = {}
        self._config_dir = Path(config_directory)
        assert self._config_dir.exists(), "config_directory can't be found, please check the path"
        self._api_db = api_db

        port_ranges = tuple(map(int, port_range.split('-')))
        self._known_ports = set(range(*port_ranges))

        self._default_idle_time = default_idle_time
        self._load_or_update_library()
        self.instance_id = str(uuid.uuid4())[:8]
        self.discard_memory_thr = max_memory_thr
        self.prefix = "dockermanaged_vllm"
        self._docker_client = docker.from_env()
        self.remove_possible_orphans()
        self.reconnect_existing_containers()

        self._spawner_lock = asyncio.Lock()
        self._inflight_gpus: set = set()
        self._inflight_ports: set = set()
        self._inflight_aliases: dict = {}  # alias -> asyncio.Event

    def remove_possible_orphans(self):
        containers = self._docker_client.containers.list()
        for c in containers:
            if c.name.startswith(self.prefix):
                labels = c.labels or {}
                # Only remove containers from THIS process instance (same instance_id)
                # Containers from previous runs will be reconnected instead
                if labels.get("manager_instance_id") == self.instance_id:
                    logger.info("Removing orphan container: %s", c.name)
                    try:
                        c.remove(force=True)
                    except Exception:
                        c.kill()

    def reconnect_existing_containers(self):
        """Reconnect to containers from a previous run."""
        if self._api_db is None:
            return
        containers = self._docker_client.containers.list()
        reconnected = 0
        for c in containers:
            if not c.name.startswith(self.prefix):
                continue
            parts = c.name.split('__', 1)
            if len(parts) < 2:
                continue
            instance_name = parts[1]
            config = None
            for cfg in self._known_configs.values():
                if cfg.model_alias.replace('/', '_') == instance_name:
                    config = cfg
                    break
            if not config:
                logger.warning("No config for container %s, skipping reconnect", c.name)
                continue
            port_bindings = c.attrs.get('NetworkSettings', {}).get('Ports', {})
            port_info = port_bindings.get('8000/tcp', [])
            if not port_info:
                logger.warning("No port mapping for %s, skipping reconnect", c.name)
                continue
            port = int(port_info[0]['HostPort'])
            url = f"http://localhost:{port}"
            api_key = self._api_db.get_container_key(config.alias)
            if not api_key:
                logger.warning("No API key for %s in DB, skipping reconnect", c.name)
                continue
            gpu_ids = self._extract_container_gpu_ids(c)
            idle_limit = config.max_idle_time if config.max_idle_time is not None else self._default_idle_time
            instance_cls = instance_types[config.config_type]
            instance = instance_cls(config.alias, url, api_key, c, idle_limit, gpu_ids=gpu_ids)
            if instance.check_health():
                self.track_new_instance(instance)
                reconnected += 1
                logger.info("Reconnected: %s at %s (GPUs: %s)", config.alias, url, gpu_ids)
            else:
                logger.warning("Container %s unhealthy during reconnect, removing", c.name)
                try:
                    c.remove(force=True)
                except Exception:
                    c.kill()
        if reconnected:
            logger.info("Reconnected %d container(s)", reconnected)

    def get_allocated_ports(self):
        return {v.port for v in self._store.values()}

    def get_vacant_ports(self, n=1):
        available = list(
            self._known_ports - self.get_allocated_ports() - self._inflight_ports
        )
        if len(available) < n:
            return None
        return random.sample(available, k=n)

    def get_gpu_ids(self, n=1, exclude=None):
        gpu_ids = find_gpu_ids(n, self.discard_memory_thr, exclude=exclude)
        if gpu_ids is None:
            return []
        return gpu_ids

    @staticmethod
    def get_gpu_info(gpu_ids):
        """Return GPU info dicts with 'index' and 'uuid' for the given local IDs."""
        if not gpu_ids:
            return []
        try:
            cmd = "nvidia-smi --query-gpu=index,uuid --format=csv,noheader"
            out = sp.check_output(cmd.split()).decode("ascii").strip()
        except Exception:
            return [{"index": gid, "uuid": "unknown"} for gid in gpu_ids]
        id_set = {str(g) for g in gpu_ids}
        result = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] in id_set:
                result.append({"index": parts[0], "uuid": parts[1]})
        for gid in gpu_ids:
            if not any(r["index"] == str(gid) for r in result):
                result.append({"index": str(gid), "uuid": "unknown"})
        return result

    def track_new_instance(self, instance: DockerInstance):
        self._store[instance.name_id] = instance

    @ttl_classcache(ttl=10)
    def _load_or_update_library(self):

        for p in self._config_dir.glob("*json"):
            config_path_str = p.as_posix()
            ch_time = p.lstat().st_mtime
            recorded_ch_time = self._config_stamps.get(config_path_str, None)
            if recorded_ch_time != ch_time:
                try:
                    config: GenericDockerConfig = AutoConfig.from_path(p)
                except Exception as e:
                    logger.error("Failed to load config %s: %s", p, e)
                    continue
                self._known_configs[config.alias] = config
                self._config_stamps[config_path_str] = ch_time
                if config.remote_url is not None:
                    if config.alias in self._store:
                        self.purge_instance(config.alias)

        return

    async def fetch_known_models(self):
        self._load_or_update_library()
        await self.remove_idle_or_crashed_instances_async(remove_idle=False)
        model_aliases = sorted(self._known_configs.keys())
        spawned_model_aliases = set(self._store.keys())
        model_lens = [self._known_configs[mn].max_model_len for mn in model_aliases]
        status = [
            "spawned" if mn in spawned_model_aliases or self._known_configs[mn].remote_url is not None
            else "offloaded"
            for mn in model_aliases
        ]
        return sorted(zip(model_aliases, model_lens, status), key=lambda x: (x[2] == "offloaded", -x[1]))

    def fetch_spawned_models(self):
        self.remove_idle_or_crashed_instances()
        model_aliases = sorted(self._store.keys())
        model_lens = [self._known_configs[mn].max_model_len for mn in model_aliases]
        return list(zip(model_aliases, model_lens))

    async def try_spawn_by_alias(self, instance_alias):
        # No need for spawning
        if instance_alias in self._store:
            return True
        # Check if model_alias is registered in the system
        self._load_or_update_library()
        if instance_alias not in self._known_configs:
            return (False, f"{instance_alias} is not registered in Config libriary.")

        config = self._known_configs[instance_alias]

        # Remote models: guard against double-spawn with asyncio.Event
        if config.remote_url is not None:
            return await self._try_spawn_remote(instance_alias, config)

        # Local models: evict idle containers one at a time until GPUs are free.
        # nvidia-smi is run outside lock via asyncio.to_thread.
        # Port reservation happens atomically with GPU reservation under lock.
        while True:
            # Phase 1: snapshot inflight GPUs under lock, check if already spawned
            async with self._spawner_lock:
                if instance_alias in self._store:
                    return True
                inflight_snapshot = frozenset(self._inflight_gpus)

            # Phase 2: nvidia-smi outside lock (may block on subprocess)
            gpu_ids = await asyncio.to_thread(
                self.get_gpu_ids, config.gpu_needed, inflight_snapshot
            )

            # Phase 3: double-check, reserve GPU + ports atomically under lock
            async with self._spawner_lock:
                if instance_alias in self._store:
                    return True
                # Check that found GPUs were not claimed while we waited
                if gpu_ids and set(gpu_ids) & self._inflight_gpus:
                    continue  # collision — retry from phase 1
                if gpu_ids:
                    ports = self.get_vacant_ports(config.ports_needed)
                    if ports is None:
                        return (False, "No ports available.")
                    self._inflight_gpus.update(gpu_ids)
                    self._inflight_ports.update(ports)
                    break
                # No GPU — try evicting the single most-idle expired instance
                evict_result = self._find_and_pop_expired_instance()
                if evict_result is None:
                    return (False, "Not enough vacant GPU to spawn Container at this time.")
                evicted_key, evicted_inst = evict_result

            # Outside lock — Docker API may block for seconds
            try:
                await asyncio.to_thread(evicted_inst.stop_container)
            finally:
                if self._api_db is not None:
                    self._api_db.delete_container_key(evicted_key)
            # Wait for GPU driver to release memory, then retry
            await asyncio.sleep(3)

        # Spawn outside the lock — GPU + ports protected by _inflight_gpus/_inflight_ports
        logger.info("Spawning %s (GPUs: %s, ports: %s)...", config.model_alias, gpu_ids, ports)
        try:
            spawned = await self.spawn_docker(config, gpu_ids=gpu_ids)
            return spawned
        finally:
            async with self._spawner_lock:
                self._inflight_gpus.difference_update(gpu_ids)
                self._inflight_ports.difference_update(ports)

    async def _try_spawn_remote(self, instance_alias, config):
        """Spawn a remote model, guarding against double-spawn via asyncio.Event."""
        while True:
            async with self._spawner_lock:
                if instance_alias in self._store:
                    return True
                if instance_alias not in self._inflight_aliases:
                    # We are the first — claim the slot and exit lock
                    event = asyncio.Event()
                    self._inflight_aliases[instance_alias] = event
                    break
                # Someone else is spawning — grab the event under lock
                event = self._inflight_aliases[instance_alias]

            # Outside lock — wait for the parallel spawn to complete
            await event.wait()
            # After waking: model may be in _store (success) or not (failure)
            if instance_alias in self._store:
                return True
            # Spawn failed — retry (the inflight entry is already cleared)
            continue

        # We won the slot — spawn
        try:
            spawned = await self.spawn_docker(config)
            return spawned
        finally:
            async with self._spawner_lock:
                if self._inflight_aliases.get(instance_alias) is event:
                    self._inflight_aliases.pop(instance_alias, None)
            event.set()  # wake all waiters

    async def fetch_instance_url(self, alias):
        if alias not in self._known_configs:
            return (
                f"{alias} is not registered in the system. Please check available models.",
                None,
                None
            )
        spawned = await self.try_spawn_by_alias(alias)
        logger.debug("Spawn result for %s: %s", alias, spawned)
        error = None
        if alias not in self._store:
            error = f"{alias} can't be deployed at this time."
        if isinstance(spawned, tuple):
            _, error = spawned
        if error is not None:
            return (
                make_server_error(error),
                None,
                None
            )
        i = self._store.get(alias)
        if i is None:
            return (
                make_server_error(f"{alias} was unloaded during request."),
                None,
                None
            )
        i.reset_access_timer()
        logger.debug("Instance URL for %s: %s", alias, i.url)
        return ("OK", i.url, i.key)

    def remove_idle_or_crashed_instances(self, remove_idle=True):
        for k in list(self._store.keys()):
            v = self._store.get(k)
            if v is None:
                continue
            healthy = v.check_health()
            logger.debug("Instance %s: health=%s, idle=%s, max_idle=%s, remove_idle=%s, expired=%s",
                         k, healthy, v.get_time_idle(), v.max_idle_time, remove_idle, v.expired())
            if not healthy or (remove_idle and v.expired()):
                self._remove_instance(k)

    async def remove_idle_or_crashed_instances_async(self, remove_idle=True):
        """Non-blocking version: snapshot store, check health outside lock, then remove dead/evict idle."""
        async with self._spawner_lock:
            snapshot = list(self._store.items())

        if not snapshot:
            return

        def _check_health(items):
            results = []
            for k, v in items:
                healthy = v.check_health()
                should_remove = not healthy or (remove_idle and v.expired())
                results.append((k, v, healthy, should_remove))
            return results

        checked = await asyncio.to_thread(_check_health, snapshot)

        for k, v, healthy, should_remove in checked:
            logger.debug("Instance %s: health=%s, idle=%s, max_idle=%s, remove_idle=%s, expired=%s",
                         k, healthy, v.get_time_idle(), v.max_idle_time, remove_idle, v.expired())
            if should_remove:
                self._remove_instance(k)

    def _remove_instance(self, k):
        """Remove an instance from store, stop its container, delete API key."""
        v = self._store.pop(k, None)
        if v is None:
            return
        v.stop_container()
        if self._api_db is not None:
            self._api_db.delete_container_key(k)

    def _find_and_pop_expired_instance(self):
        """Find the most-idle expired instance and pop it from _store (under lock).
        Returns (key, instance) or None. Caller must handle Docker API cleanup outside lock."""
        expired = [
            (k, v) for k, v in self._store.items()
            if not v.is_virtual and v.expired()
        ]
        if not expired:
            return None
        k, v = max(expired, key=lambda kv: kv[1].get_time_idle())
        logger.info(
            "Evicting idle instance %s (idle %.0fm, limit %sm) to free GPU",
            k, v.get_time_idle(), v.max_idle_time,
        )
        self._store.pop(k)
        return k, v

    async def _evict_one_idle_instance(self):
        """Kill the most-idle expired instance. Returns True if evicted."""
        async with self._spawner_lock:
            result = self._find_and_pop_expired_instance()
            if result is None:
                return False
            evicted_key, evicted_inst = result

        try:
            await asyncio.to_thread(evicted_inst.stop_container)
        finally:
            if self._api_db is not None:
                self._api_db.delete_container_key(evicted_key)
        return True

    def purge_instance(self, k):
        self._remove_instance(k)

    def purge_all_instances(self):
        for k in list(self._store.keys()):
            self.purge_instance(k)
        if self._api_db is not None:
            self._api_db.clear_container_keys()

    @staticmethod
    def _collect_container_diag(container):
        lines = []
        try:
            container.reload()
            lines.append(f"status={container.status}")
        except Exception as e:
            lines.append(f"status=unknown ({e})")
        try:
            logs = container.logs(tail=50).decode("utf-8", errors="replace")
            if logs.strip():
                lines.append(f"logs (last 50 lines):\n{logs}")
            else:
                lines.append("logs: <empty>")
        except Exception as e:
            lines.append(f"logs: unavailable ({e})")
        return "\n".join(lines)

    @staticmethod
    def _extract_container_gpu_ids(container):
        """Extract GPU local IDs from container's device requests."""
        try:
            reqs = container.attrs.get("HostConfig", {}).get("DeviceRequests", []) or []
            for req in reqs:
                ids = req.get("DeviceIDs") or []
                if ids:
                    return list(ids)
        except Exception:
            pass
        return []

    async def spawn_docker(self, config: GenericDockerConfig, startup_time: int = 30, retry_count: int = 35, gpu_ids=None):
        args_builder = spawner_scripts[config.spawn_script]

        # Generate per-model API key
        api_key = str(uuid.uuid4())
        if self._api_db is not None:
            self._api_db.save_container_key(config.alias, api_key)

        def _rollback_api_key():
            if self._api_db is not None:
                self._api_db.delete_container_key(config.alias)

        # Remote configs don't need local GPUs or ports
        if config.remote_url is not None:
            args = args_builder(config, [], api_key)
            instance_cls = instance_types[config.config_type]
            instance = instance_cls(
                config.alias, args.api_url, args.api_key,
                None, -1
            )
            if not instance.check_health():
                _rollback_api_key()
                return (False, "Remote model is not reachable.")
            self.track_new_instance(instance)
            return True

        if gpu_ids is None:
            gpu_ids = self.get_gpu_ids(config.gpu_needed)
        if not gpu_ids:
            _rollback_api_key()
            return (False, "Not enough vacant GPU to spawn Container at this time.")
        gpu = docker.types.DeviceRequest(device_ids=gpu_ids, capabilities=[['gpu']])

        ports = self.get_vacant_ports(config.ports_needed)
        if ports is None:
            _rollback_api_key()
            return (False, "No ports available.")
        args = args_builder(config, ports, api_key)

        client = self._docker_client
        name_str = f"{self.prefix}__{args.instance_name}"
        try:
            container = client.containers.run(
                args.docker_name,
                command=args.command,
                name=name_str,
                detach=True,
                auto_remove=False,
                tty=True,
                mounts=args.mounts,
                ports=args.port_map,
                device_requests=[gpu],
                shm_size="12G",
                environment=args.env_args,
                labels={"manager_instance_id": self.instance_id}
            )
        except Exception as e:
            _rollback_api_key()
            return (False, str(e))

        idle_limit = config.max_idle_time if config.max_idle_time is not None else self._default_idle_time

        instance_cls = instance_types[config.config_type]
        instance = instance_cls(
            config.alias, args.api_url, args.api_key,
            container, idle_limit, gpu_ids=gpu_ids
        )

        # Make sure container started with exponential backoff
        delay = 1
        max_delay = startup_time
        while instance.container_exists() and not instance.check_api_health() and retry_count:
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
            retry_count -= 1

        if not instance.check_health():
            diag = self._collect_container_diag(container)
            logger.error("Container %s failed to start.\n%s", config.model_alias, diag)
            instance.stop_container()
            del instance
            _rollback_api_key()
            return (False, f"Failed to start. {diag}")

        self.track_new_instance(instance)
        return True
