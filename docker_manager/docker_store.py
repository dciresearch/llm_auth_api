from src.utils import ttl_classcache
import asyncio
import dataclasses
import requests
from docker.errors import NotFound, APIError
import uuid
from pathlib import Path
import os
import subprocess as sp
import random
import time
from docker.models.containers import Container
from typing import Union, Dict, Any, List
import json
import docker


def read_json_config(path):
    with open(path) as f:
        res = json.load(f)
        return res


@dataclasses.dataclass(init=False)
class VllmConfig:
    model_parent_dir: str
    model_name: str
    model_alias: str
    gpu_needed: int = 1
    min_util: float = 0.95
    max_model_len: int = -1
    extra_args: dict = lambda: {}
    max_idle_time = None
    tags: List[str] = lambda: []

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        self.max_model_len = self.extra_args.get("max_model_len", self.max_model_len)

    @classmethod
    def from_path(cls, path):
        config = read_vllm_config(path)
        return cls(**config)

    def __getitem__(self, key):
        return getattr(self, key)


def read_vllm_config(path):
    config = read_json_config(path)
    if 'extra_args' not in config:
        config['extra_args'] = {}
    return config


DEFAULT_VLLM_DOCKER_NAME = 'vllm_server'
DEFAULT_DOCKER_MODEL_DIR = "/workdir/models"
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


def get_gpu_breakdown(gpu_needed):
    tp = 1
    pp = 1
    while gpu_needed > 1:
        if gpu_needed % 2 == 0:
            tp *= 2
            gpu_needed //= 2
        else:
            pp = gpu_needed
            gpu_needed = 1
    return tp, pp


def is_vllm_up(port, url='localhost'):
    try:
        r = requests.get(url=f'http://{url}:{port}/health')
        return r.status_code == 200
    except:
        return False


class VllmInstance:
    def __init__(self, model_alias: str, vllm_port: int, api_key: str, container: Container, max_idle_time: int):
        self.model_alias = model_alias
        self.vllm_port = vllm_port
        self.api_key = api_key
        self.container = container
        self._max_idle_time = max_idle_time

        self._time_created = time.time()
        self._last_accessed = time.time()

    def reset_access_timer(self):
        self._last_accessed = time.time()

    def check_api_health(self):
        return is_vllm_up(self.port)

    def container_exists(self):
        try:
            self.container.reload()
            return True
        except NotFound:
            return False

    def check_container_health(self):
        if self.container_exists():
            return self.container.status == 'running'

    def check_health(self):
        # print(self.container.status)
        if self.check_container_health():
            return self.check_api_health()
        return False

    @property
    def port(self) -> int:
        return self.vllm_port

    @property
    def key(self) -> int:
        return self.api_key

    @property
    def max_idle_time(self):
        return self._max_idle_time

    def get_time_idle(self):
        return (time.time()-self._last_accessed) // 60

    def expired(self):
        return self.get_time_idle() > self.max_idle_time

    def stop_container(self):
        while True:
            try:
                self.container.kill()
            except (NotFound, APIError):
                break

    def __del__(self):
        self.stop_container()


SERVER_ERORR_PATTERN = "{} Please consult your server administrator."


def make_server_error(error_text):
    return SERVER_ERORR_PATTERN.format(error_text)


spawner_lock = asyncio.Lock()


class InstanceManager:
    def __init__(
        self, config_directory: str, port_range=DEFAULT_PORT_RANGE,
        max_memory_thr=USED_MEMORY_THRESHOLD, default_idle_time=120
    ):
        self._store: Dict[str, VllmInstance] = {}
        self._known_configs: Dict[str,  VllmConfig] = {}
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
                c.stop()

    def get_allocated_ports(self):
        return {v.port for v in self._store.values()}

    def get_vacant_port(self):
        return random.choice(list(self._known_ports - self.get_allocated_ports()))

    def track_new_instance(self, instance: VllmInstance):
        self._store[instance.model_alias] = instance

    @ttl_classcache(ttl=10)
    def _load_or_update_library(self):

        for p in self._config_dir.glob("*json"):
            config_path_str = p.as_posix()
            ch_time = p.lstat().st_mtime
            recorded_ch_time = self._config_stamps.get(config_path_str, None)
            if recorded_ch_time != ch_time:
                config: VllmConfig = VllmConfig.from_path(p)
                self._known_configs[config.model_alias] = config
                self._config_stamps[config_path_str] = ch_time
        return

    async def fetch_known_models(self):
        self._load_or_update_library()
        # Remove lost dockers
        self.remove_idle_or_crashed_instances(remove_idle=False)
        model_names = sorted(self._known_configs.keys())
        spawned_model_names = set(self._store.keys())
        model_lens = [self._known_configs[mn].max_model_len for mn in model_names]
        status = ["spawned" if mn in spawned_model_names else "offloaded" for mn in model_names]
        return sorted(zip(model_names, model_lens, status), key=lambda x: (x[2] == "offloaded", -x[1]))

    def fetch_spawned_models(self):
        self.remove_idle_or_crashed_instances()
        model_names = sorted(self._store.keys())
        model_lens = [self._known_configs[mn].max_model_len for mn in model_names]
        return list(zip(model_names, model_lens))

    async def try_spawn_by_alias(self, model_alias):
        # No need for spawning
        if model_alias in self._store:
            return True
        # Check if model_alias is registered in the system
        self._load_or_update_library()
        if model_alias not in self._known_configs:
            return (False, f"{model_alias} is not registered in Config libriary.")

        # TODO make gpuid-based lock
        async with spawner_lock:
            # in case the party has started
            if model_alias in self._store:
                return True
            config = self._known_configs[model_alias]

            # Remove lost dockers
            self.remove_idle_or_crashed_instances(remove_idle=False)

            # Check if we have gpus to spawn new docker
            gpu_ids = self.get_gpu_ids(config)
            # Try removing idle containers to free up space
            if not gpu_ids:
                self.remove_idle_or_crashed_instances()
            print(f"Found gpus {gpu_ids} for {config.model_alias}. Spawning...")
            spawned = await self.spawn_docker(config)
            return spawned

    async def fetch_instance_port(self, model_alias):
        if model_alias not in self._known_configs:
            return (
                f"{model_alias} is not registered in the system. Please check available models.",
                None,
                None
            )
        spawned = await self.try_spawn_by_alias(model_alias)
        error = None
        if model_alias not in self._store:
            error = f"{model_alias} can't be deployed at this time."
        if isinstance(spawned, tuple):
            _, error = spawned
        if error is not None:
            return (
                make_server_error(error),
                None,
                None
            )
        i = self._store[model_alias]
        i.reset_access_timer()
        return ("OK", i.port, i.key)

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

    def purge_all_instances(self):
        for k in list(self._store.keys()):
            v = self._store.pop(k)
            del v

    def get_gpu_ids(self, config: VllmConfig):
        gpu_ids = find_gpu_ids(config.gpu_needed, self.discard_memory_thr)
        if gpu_ids is None:
            return []
        gpu_ids = list(map(str, gpu_ids))
        return gpu_ids

    async def spawn_docker(self, config: VllmConfig, startup_time: int = 20, retry_count: int = 6):
        gpu_ids = self.get_gpu_ids(config)
        # No gpus to spawn new docker
        if not gpu_ids:
            return (False, "Not enough vacant GPU to spawn Container at this time.")
        gpu = docker.types.DeviceRequest(device_ids=gpu_ids, capabilities=[['gpu']])
        tp, pp = get_gpu_breakdown(config.gpu_needed)

        port = self.get_vacant_port()

        mount = docker.types.Mount(
            target=DEFAULT_DOCKER_MODEL_DIR, source=config.model_parent_dir, type='bind'
        )

        extra_args = " ".join(f"--{k} {v}" for k, v in config.extra_args.items())

        api_key = self.API_KEY
        command = f"--model {DEFAULT_DOCKER_MODEL_DIR}/{config.model_name} --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --gpu-memory-utilization {config.min_util} {extra_args} --api-key {api_key}"

        client = docker.from_env()

        model_alias = config.model_alias.replace('/', '_')
        name_str = f"{self.prefix}__{model_alias}"

        try:
            container = client.containers.run(
                DEFAULT_VLLM_DOCKER_NAME,
                command=command,
                name=name_str,
                detach=True,
                auto_remove=True,
                tty=True,
                mounts=[mount],
                ports={8000: port},
                device_requests=[gpu],
                shm_size="12G",
            )
        except Exception as e:
            return (False, str(e))

        idle_limit = config.max_idle_time if config.max_idle_time is not None else self._default_idle_time

        instance = VllmInstance(config.model_alias, port, api_key, container,
                                idle_limit)

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
