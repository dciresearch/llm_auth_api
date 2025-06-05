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
from .instances import instance_types, DockerInstance
from .spawn_logic import spawner_scripts
from .config_patterns import GenericDockerConfig, AutoConfig

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


def find_gpu_ids(gpu_needed, discard_memory_thr):
    gpu_usage = get_gpu_memory()
    vacant_gpu = [k for k, v in gpu_usage.items() if v['memory_used'] < discard_memory_thr]
    if len(vacant_gpu) < gpu_needed:
        return None
    return random.sample(vacant_gpu, gpu_needed)


SERVER_ERORR_PATTERN = "{} Please consult your server administrator."


def make_server_error(error_text):
    return SERVER_ERORR_PATTERN.format(error_text)


spawner_lock = asyncio.Lock()


class InstanceManager:
    def __init__(
        self, config_directory: str, port_range=DEFAULT_PORT_RANGE,
        max_memory_thr=USED_MEMORY_THRESHOLD, default_idle_time=120
    ):
        self._store: Dict[str, DockerInstance] = {}
        self._known_configs: Dict[str,  GenericDockerConfig] = {}
        self._config_stamps: Dict[str, float] = {}
        self._config_dir = Path(config_directory)
        assert self._config_dir.exists(), "config_directory can't be found, please check the path"

        port_ranges = tuple(map(int, port_range.split('-')))
        self._known_ports = set(range(*port_ranges))

        self._default_idle_time = default_idle_time
        self._load_or_update_library()
        self.API_KEY = str(uuid.uuid4())
        self.discard_memory_thr = max_memory_thr
        self.prefix = "dockermanaged_vllm"
        self.remove_possible_orphans()

    def remove_possible_orphans(self):
        client = docker.from_env()
        containers = client.containers.list()
        print(containers)
        for c in containers:
            if c.name.startswith(self.prefix):
                c.kill()

    def get_allocated_ports(self):
        return {v.port for v in self._store.values()}

    def get_vacant_ports(self, n=1):
        return random.sample(
            list(self._known_ports - self.get_allocated_ports()),
            k=n
        )

    def get_gpu_ids(self, n=1):
        gpu_ids = find_gpu_ids(n, self.discard_memory_thr)
        if gpu_ids is None:
            return []
        gpu_ids = list(map(str, gpu_ids))
        return gpu_ids

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
                    print(e)
                    continue
                self._known_configs[config.alias] = config
                self._config_stamps[config_path_str] = ch_time
                if config.remote_url is not None:
                    if config.alias in self._store:
                        self.purge_instance(config.alias)

        return

    async def fetch_known_models(self):
        self._load_or_update_library()
        # Remove lost dockers
        self.remove_idle_or_crashed_instances(remove_idle=False)
        model_aliases = sorted(self._known_configs.keys())
        spawned_model_aliases = set(self._store.keys())
        model_lens = [self._known_configs[mn].max_model_len for mn in model_aliases]
        status = ["spawned" if mn in spawned_model_aliases else "offloaded" for mn in model_aliases]
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

        # TODO make gpuid-based lock
        async with spawner_lock:
            # in case the party has started
            if instance_alias in self._store:
                return True
            config = self._known_configs[instance_alias]
            # Remove lost dockers
            self.remove_idle_or_crashed_instances(remove_idle=False)

            # Check if we have gpus to spawn new docker
            gpu_ids = self.get_gpu_ids(config.gpu_needed)
            # Try removing idle containers to free up space
            if not gpu_ids:
                self.remove_idle_or_crashed_instances()
            print(f"Found gpus {gpu_ids} for {config.model_alias}. Spawning...")
            # TODO add timeout spawn
            spawned = await self.spawn_docker(config)
            return spawned

    async def fetch_instance_url(self, alias):
        if alias not in self._known_configs:
            return (
                f"{alias} is not registered in the system. Please check available models.",
                None,
                None
            )
        spawned = await self.try_spawn_by_alias(alias)
        print(spawned)
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
        i = self._store[alias]
        i.reset_access_timer()
        print(i.url)
        return ("OK", i.url, i.key)

    def remove_idle_or_crashed_instances(self, remove_idle=True):
        for k in list(self._store.keys()):
            v = self._store[k]
            print(v, v.check_health(), v.get_time_idle(), v.max_idle_time, remove_idle and v.expired())
            if (
                (
                    not v.check_health()
                ) or (
                    remove_idle and v.expired()
                )
            ):
                v.stop_container()
                self._store.pop(k)

    def purge_instance(self, k):
        v = self._store.pop(k)
        del v

    def purge_all_instances(self):
        for k in list(self._store.keys()):
            self.purge_instance(k)

    async def spawn_docker(self, config: GenericDockerConfig, startup_time: int = 20, retry_count: int = 20):
        args_builder = spawner_scripts[config.spawn_script]
        gpu_ids = self.get_gpu_ids(config.gpu_needed)
        # No gpus to spawn new docker
        if not gpu_ids:
            return (False, "Not enough vacant GPU to spawn Container at this time.")
        gpu = docker.types.DeviceRequest(device_ids=gpu_ids, capabilities=[['gpu']])

        ports = self.get_vacant_ports(config.ports_needed)

        args = args_builder(
            config, ports, self.API_KEY
        )

        if args.docker_name is None:
            container = None
            idle_limit = -1
        else:
            client = docker.from_env()
            name_str = f"{self.prefix}__{args.instance_name}"
            try:
                container = client.containers.run(
                    args.docker_name,
                    command=args.command,
                    name=name_str,
                    detach=True,
                    auto_remove=True,
                    tty=True,
                    mounts=args.mounts,
                    ports=args.port_map,
                    device_requests=[gpu],
                    shm_size="12G",
                    environment=args.env_args
                )
            except Exception as e:
                return (False, str(e))

            idle_limit = config.max_idle_time if config.max_idle_time is not None else self._default_idle_time

        instance_cls = instance_types[config.config_type]
        instance = instance_cls(
            config.alias, args.api_url, args.api_key,
            container, idle_limit
        )

        # Make sure container started
        # TODO make dynamic startup check
        while instance.container_exists() and not instance.check_api_health() and retry_count:
            await asyncio.sleep(startup_time)
            retry_count -= 1

        if not instance.check_health():
            del instance
            return (False, "Failed to start due to Container internal error.")

        self.track_new_instance(instance)
        return True
