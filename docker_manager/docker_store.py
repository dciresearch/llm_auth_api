from src.utils import ttl_classcache
import asyncio
from docker.errors import NotFound, APIError
import uuid
from pathlib import Path
import subprocess as sp
import random
from typing import Dict
import docker
import logging
from .instances import instance_types, DockerInstance
from .spawn_logic import spawner_scripts
from .config_patterns import GenericDockerConfig, AutoConfig

logger = logging.getLogger(__name__)

DEFAULT_PORT_RANGE = "10240-10340"

USED_MEMORY_THRESHOLD = 100


def get_gpu_memory():
    """Query nvidia-smi for per-GPU memory usage.
    Returns dict: {local_index: {index, memory_used, memory_total, uuid}}
    Uses UUID as the stable host-side identifier.
    """
    command = "nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv"
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
            try:
                row_dict[k] = int(v)
            except ValueError:
                pass  # keep string values (e.g. uuid)
        rows[row_dict['index']] = row_dict
    return rows


def find_free_gpu_uuids(gpu_needed, discard_memory_thr, exclude_uuids=None):
    """Find UUIDs of GPUs with memory usage below threshold.

    Returns list of UUID strings (host-stable identifiers), or None if not enough.
    exclude_uuids: set of UUID strings already reserved (inflight).
    """
    gpu_usage = get_gpu_memory()
    excluded = exclude_uuids or set()
    vacant = [
        v['uuid'] for v in gpu_usage.values()
        if v['memory_used'] < discard_memory_thr and v['uuid'] not in excluded
    ]
    logger.debug(
        "GPU scan: threshold=%dMiB, total=%d, vacant=%d, excluded=%s, vacant_uuids=%s",
        discard_memory_thr, len(gpu_usage), len(vacant), excluded, vacant
    )
    if len(vacant) < gpu_needed:
        return None
    return random.sample(vacant, gpu_needed)


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
        self._known_configs: Dict[str, GenericDockerConfig] = {}
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
        # Serialises nvidia-smi + GPU reservation to prevent two concurrent
        # spawn requests both seeing the same GPU as free before either has
        # updated _inflight_gpus. This is the root cause of "both models land
        # on GPU 0" when spawning two models simultaneously.
        self._gpu_alloc_lock = asyncio.Lock()
        # UUIDs of GPUs currently being provisioned (container starting up).
        # Uses UUIDs (not local indices) so they are stable across container
        # boundaries — the manager may run inside a Docker container with
        # remapped local indices (e.g. host GPU 2 appears as index 1 inside
        # the manager container), but UUIDs are always the same on the host.
        self._inflight_gpu_uuids: set = set()
        self._inflight_ports: set = set()
        self._inflight_aliases: dict = {}  # alias -> asyncio.Event

        logger.info(
            "InstanceManager started (id=%s, memory_threshold=%dMiB, idle_timeout=%dm)",
            self.instance_id, self.discard_memory_thr, self._default_idle_time
        )

    def remove_possible_orphans(self):
        containers = self._docker_client.containers.list(all=True)
        removed = 0
        for c in containers:
            if not c.name.startswith(self.prefix):
                continue
            if c.status in ('exited', 'dead', 'created'):
                logger.info("Startup: removing stale container %s (status=%s)", c.name, c.status)
                try:
                    c.remove(force=True)
                    removed += 1
                except Exception as e:
                    logger.warning("Failed to remove stale container %s: %s", c.name, e)
                continue
            labels = c.labels or {}
            if labels.get("manager_instance_id") == self.instance_id:
                logger.info("Startup: removing orphan container %s", c.name)
                try:
                    c.remove(force=True)
                    removed += 1
                except Exception as e:
                    logger.warning("Failed to remove orphan container %s: %s", c.name, e)
        if removed:
            logger.info("Startup: removed %d stale/orphan container(s)", removed)

    def reconnect_existing_containers(self):
        """Reconnect to healthy containers from a previous manager run."""
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
                logger.warning("Reconnect: no config for container %s, skipping", c.name)
                continue
            port_bindings = c.attrs.get('NetworkSettings', {}).get('Ports', {})
            port_info = port_bindings.get('8000/tcp', [])
            if not port_info:
                logger.warning("Reconnect: no port mapping for %s, skipping", c.name)
                continue
            port = int(port_info[0]['HostPort'])
            url = f"http://localhost:{port}"
            api_key = self._api_db.get_container_key(config.alias)
            if not api_key:
                logger.warning("Reconnect: no API key for %s in DB, skipping", c.name)
                continue
            gpu_uuids = self._extract_container_gpu_uuids(c)
            idle_limit = config.max_idle_time if config.max_idle_time is not None else self._default_idle_time
            instance_cls = instance_types[config.config_type]
            instance = instance_cls(config.alias, url, api_key, c, idle_limit, gpu_ids=gpu_uuids)
            if instance.check_health():
                self.track_new_instance(instance)
                reconnected += 1
                logger.info("Reconnected: %s at %s (GPU UUIDs: %s)", config.alias, url, gpu_uuids)
            else:
                logger.warning("Reconnect: container %s unhealthy, removing", c.name)
                _force_remove_container(c)
        if reconnected:
            logger.info("Reconnected %d container(s) from previous run", reconnected)

    def get_allocated_ports(self):
        return {v.port for v in self._store.values()}

    def get_vacant_ports(self, n=1):
        available = list(
            self._known_ports - self.get_allocated_ports() - self._inflight_ports
        )
        if len(available) < n:
            return None
        return random.sample(available, k=n)

    def get_free_gpu_uuids(self, n=1, exclude_uuids=None):
        """Return n free GPU UUIDs or empty list."""
        result = find_free_gpu_uuids(n, self.discard_memory_thr, exclude_uuids=exclude_uuids)
        if result is None:
            return []
        return result

    @staticmethod
    def get_gpu_info(gpu_ids):
        """Return GPU info dicts with 'index' and 'uuid' for the given IDs (index or UUID)."""
        if not gpu_ids:
            return []
        try:
            cmd = "nvidia-smi --query-gpu=index,uuid --format=csv,noheader"
            out = sp.check_output(cmd.split()).decode("ascii").strip()
        except Exception:
            return [{"index": "unknown", "uuid": gid} for gid in gpu_ids]
        id_set = {str(g) for g in gpu_ids}
        result = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and (parts[0] in id_set or parts[1] in id_set):
                result.append({"index": parts[0], "uuid": parts[1]})
        for gid in gpu_ids:
            if not any(r["uuid"] == str(gid) or r["index"] == str(gid) for r in result):
                result.append({"index": "unknown", "uuid": str(gid)})
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
        # Fast path: already in store — but verify it's actually alive.
        if instance_alias in self._store:
            if self._store[instance_alias].check_health():
                return True
            logger.warning(
                "[%s] instance in store but unhealthy — removing before respawn", instance_alias
            )
            self._remove_instance(instance_alias)

        self._load_or_update_library()
        if instance_alias not in self._known_configs:
            return (False, f"{instance_alias} is not registered in Config library.")

        config = self._known_configs[instance_alias]

        if config.remote_url is not None:
            return await self._try_spawn_remote(instance_alias, config)

        # Local model: reserve GPUs atomically.
        # _gpu_alloc_lock ensures only one coroutine at a time runs nvidia-smi
        # and updates _inflight_gpu_uuids, eliminating the race where two
        # concurrent spawns both see the same GPU as free.
        while True:
            async with self._gpu_alloc_lock:
                async with self._spawner_lock:
                    if instance_alias in self._store:
                        return True

                exclude = frozenset(self._inflight_gpu_uuids)
                logger.debug(
                    "[%s] scanning GPUs (need=%d, inflight_uuids=%s)",
                    instance_alias, config.gpu_needed, exclude
                )
                gpu_uuids = await asyncio.to_thread(
                    self.get_free_gpu_uuids, config.gpu_needed, exclude
                )

                async with self._spawner_lock:
                    if instance_alias in self._store:
                        return True
                    if gpu_uuids:
                        ports = self.get_vacant_ports(config.ports_needed)
                        if ports is None:
                            return (False, "No ports available.")
                        self._inflight_gpu_uuids.update(gpu_uuids)
                        self._inflight_ports.update(ports)
                        logger.info(
                            "[%s] reserved GPU UUIDs=%s ports=%s",
                            instance_alias, gpu_uuids, ports
                        )
                        break
                    # No free GPU — try evicting the most-idle expired instance.
                    evict_result = self._find_and_pop_expired_instance()
                    if evict_result is None:
                        logger.warning(
                            "[%s] no free GPUs and no evictable instances", instance_alias
                        )
                        return (False, "Not enough vacant GPU to spawn container at this time.")
                    evicted_key, evicted_inst = evict_result

            # Outside both locks — Docker stop may block for seconds.
            logger.info("[%s] waiting for evicted instance %s to free GPU", instance_alias, evicted_key)
            try:
                await asyncio.to_thread(evicted_inst.stop_container)
            finally:
                if self._api_db is not None:
                    self._api_db.delete_container_key(evicted_key)
            logger.info("[%s] eviction done, waiting 3s for GPU memory to clear", instance_alias)
            await asyncio.sleep(3)

        # GPU UUIDs and ports are reserved; spawn outside locks.
        try:
            spawned = await self.spawn_docker(config, gpu_uuids=gpu_uuids)
            return spawned
        finally:
            async with self._spawner_lock:
                self._inflight_gpu_uuids.difference_update(gpu_uuids)
                self._inflight_ports.difference_update(ports)
            logger.debug(
                "[%s] released inflight reservation (uuids=%s ports=%s)",
                instance_alias, gpu_uuids, ports
            )

    async def _try_spawn_remote(self, instance_alias, config):
        """Spawn a remote model, guarding against double-spawn via asyncio.Event."""
        while True:
            async with self._spawner_lock:
                if instance_alias in self._store:
                    return True
                if instance_alias not in self._inflight_aliases:
                    event = asyncio.Event()
                    self._inflight_aliases[instance_alias] = event
                    break
                event = self._inflight_aliases[instance_alias]
            await event.wait()
            if instance_alias in self._store:
                return True
            continue

        try:
            spawned = await self.spawn_docker(config)
            return spawned
        finally:
            async with self._spawner_lock:
                if self._inflight_aliases.get(instance_alias) is event:
                    self._inflight_aliases.pop(instance_alias, None)
            event.set()

    async def fetch_instance_url(self, alias):
        if alias not in self._known_configs:
            return (
                f"{alias} is not registered in the system. Please check available models.",
                None,
                None
            )
        spawned = await self.try_spawn_by_alias(alias)
        error = None
        if alias not in self._store:
            error = f"{alias} can't be deployed at this time."
        if isinstance(spawned, tuple):
            _, error = spawned
        if error is not None:
            logger.warning("[%s] fetch_instance_url failed: %s", alias, error)
            return (make_server_error(error), None, None)
        i = self._store.get(alias)
        if i is None:
            return (make_server_error(f"{alias} was unloaded during request."), None, None)
        i.reset_access_timer()
        logger.debug("[%s] serving at %s", alias, i.url)
        return ("OK", i.url, i.key)

    def remove_idle_or_crashed_instances(self, remove_idle=True):
        for k in list(self._store.keys()):
            v = self._store.get(k)
            if v is None:
                continue
            healthy = v.check_health()
            idle_min = v.get_time_idle()
            if not healthy:
                logger.info("[%s] unhealthy, removing", k)
                self._remove_instance(k)
            elif remove_idle and v.expired():
                logger.info("[%s] idle for %.0fm (limit %sm), removing", k, idle_min, v.max_idle_time)
                self._remove_instance(k)
            else:
                logger.debug("[%s] healthy, idle=%.0fm, max_idle=%sm", k, idle_min, v.max_idle_time)

    async def remove_idle_or_crashed_instances_async(self, remove_idle=True):
        """Non-blocking version: snapshot store, check health in thread, then remove."""
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
            if should_remove:
                reason = "unhealthy" if not healthy else f"idle {v.get_time_idle():.0f}m"
                logger.info("[%s] removing (%s)", k, reason)
                self._remove_instance(k)
            else:
                logger.debug("[%s] healthy, idle=%.0fm", k, v.get_time_idle())

    def _remove_instance(self, k):
        """Remove instance from store, stop container, delete API key."""
        v = self._store.pop(k, None)
        if v is None:
            return
        logger.info("[%s] stopping container (gpu_uuids=%s)", k, v.gpu_ids)
        v.stop_container()
        if self._api_db is not None:
            self._api_db.delete_container_key(k)
        logger.info("[%s] removed", k)

    def _find_and_pop_expired_instance(self):
        """Pop the most-idle expired instance from _store (must be called under _spawner_lock)."""
        expired = [
            (k, v) for k, v in self._store.items()
            if not v.is_virtual and v.expired()
        ]
        if not expired:
            return None
        k, v = max(expired, key=lambda kv: kv[1].get_time_idle())
        logger.info(
            "[%s] selected for eviction (idle=%.0fm, limit=%sm, gpu_uuids=%s)",
            k, v.get_time_idle(), v.max_idle_time, v.gpu_ids
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
        logger.info("Purging all instances (%d)", len(self._store))
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
    def _extract_container_gpu_uuids(container):
        """Extract GPU UUIDs from container's DeviceRequests (host-stable identifiers)."""
        try:
            reqs = container.attrs.get("HostConfig", {}).get("DeviceRequests", []) or []
            for req in reqs:
                ids = req.get("DeviceIDs") or []
                if ids:
                    # UUIDs start with "GPU-"; plain integers are local indices (legacy).
                    uuids = [i for i in ids if str(i).startswith("GPU-")]
                    if uuids:
                        return uuids
                    return list(ids)
        except Exception:
            pass
        return []

    async def spawn_docker(
        self, config: GenericDockerConfig,
        startup_time: int = 30, retry_count: int = 35,
        gpu_uuids=None
    ):
        args_builder = spawner_scripts[config.spawn_script]

        api_key = str(uuid.uuid4())
        if self._api_db is not None:
            self._api_db.save_container_key(config.alias, api_key)

        def _rollback_api_key():
            if self._api_db is not None:
                self._api_db.delete_container_key(config.alias)

        # Remote model — no GPU or port needed.
        if config.remote_url is not None:
            args = args_builder(config, [], api_key)
            instance_cls = instance_types[config.config_type]
            instance = instance_cls(config.alias, args.api_url, args.api_key, None, -1)
            if not instance.check_health():
                _rollback_api_key()
                return (False, "Remote model is not reachable.")
            self.track_new_instance(instance)
            return True

        # GPU allocation (fallback path — normally gpu_uuids are pre-reserved by caller).
        if gpu_uuids is None:
            gpu_uuids = self.get_free_gpu_uuids(config.gpu_needed)
        if not gpu_uuids:
            _rollback_api_key()
            return (False, "Not enough vacant GPU to spawn container at this time.")

        ports = self.get_vacant_ports(config.ports_needed)
        if ports is None:
            _rollback_api_key()
            return (False, "No ports available.")

        args = args_builder(config, ports, api_key)

        # Pass UUIDs to Docker — stable across container boundaries.
        # When the manager itself runs inside a container with remapped GPU
        # indices (e.g. host GPU 2 = local index 1), using UUIDs ensures the
        # child container always gets the correct host GPU.
        gpu = docker.types.DeviceRequest(device_ids=gpu_uuids, capabilities=[['gpu']])

        client = self._docker_client
        name_str = f"{self.prefix}__{args.instance_name}"

        logger.info(
            "[%s] docker run: image=%s gpu_uuids=%s port=%s",
            config.model_alias, args.docker_name, gpu_uuids, ports
        )

        container = None
        for _attempt in range(2):
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
                logger.info("[%s] container started: id=%s", config.model_alias, container.short_id)
                break
            except docker.errors.APIError as e:
                if e.status_code == 409 and _attempt == 0:
                    logger.warning(
                        "[%s] container name conflict (%s), removing stale and retrying",
                        config.model_alias, name_str
                    )
                    try:
                        stale = client.containers.get(name_str)
                        stale.remove(force=True)
                        logger.info("[%s] stale container removed", config.model_alias)
                    except Exception as rm_err:
                        logger.warning("[%s] failed to remove stale container: %s", config.model_alias, rm_err)
                    continue
                logger.error("[%s] docker API error: %s", config.model_alias, e)
                _rollback_api_key()
                return (False, str(e))
            except Exception as e:
                logger.error("[%s] unexpected error during docker run: %s", config.model_alias, e)
                _rollback_api_key()
                return (False, str(e))

        if container is None:
            _rollback_api_key()
            return (False, "Failed to create container after retry.")

        idle_limit = config.max_idle_time if config.max_idle_time is not None else self._default_idle_time
        instance_cls = instance_types[config.config_type]
        instance = instance_cls(
            config.alias, args.api_url, args.api_key,
            container, idle_limit, gpu_ids=gpu_uuids
        )

        # Poll for readiness with exponential backoff.
        logger.info(
            "[%s] waiting for container to become ready (max ~%ds, %d retries)...",
            config.model_alias, startup_time * retry_count, retry_count
        )
        delay = 1
        max_delay = startup_time
        attempts_left = retry_count
        while instance.container_exists() and not instance.check_api_health() and attempts_left:
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
            attempts_left -= 1
            logger.debug(
                "[%s] health check pending (retries_left=%d, next_delay=%ds)",
                config.model_alias, attempts_left, delay
            )

        if not instance.check_health():
            diag = self._collect_container_diag(container)
            logger.error("[%s] failed to start.\n%s", config.model_alias, diag)
            _force_remove_container(container, label=config.model_alias)
            _rollback_api_key()
            return (False, f"Failed to start. {diag}")

        self.track_new_instance(instance)
        logger.info(
            "[%s] ready at %s (gpu_uuids=%s)",
            config.model_alias, args.api_url, gpu_uuids
        )
        return True


def _force_remove_container(container, label=""):
    """Kill + remove a Docker container, tolerating already-gone containers."""
    try:
        container.kill()
    except (NotFound, APIError):
        pass
    except Exception as e:
        logger.warning("[%s] kill failed during cleanup: %s", label, e)
    try:
        container.remove(force=True)
        logger.info("[%s] container removed", label)
    except (NotFound, APIError):
        pass
    except Exception as e:
        logger.error("[%s] remove failed during cleanup: %s", label, e)
