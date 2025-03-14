from docker.errors import NotFound
import uuid
from pathlib import Path
import os
import subprocess as sp
import random
import time
from docker.models.containers import Container
from typing import Union, Dict, Any
import json
import docker


def read_json_config(path):
    with open(path) as f:
        res = json.load(f)
        return res


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


class LlmInstance:
    def __init__(self, model_alias: str, vllm_port: int, api_key: str, container: Container):
        self.model_alias = model_alias
        self.vllm_port = vllm_port
        self.api_key = api_key
        self.container = container

        self.time_created = time.time()
        self.last_accessed = time.time()

    def reset_access_timer(self):
        self.last_accessed = time.time()

    def check_health(self):
        try:
            self.container.reload()
        except NotFound:
            return False
        return self.container.status == 'running'

    def get_port(self) -> int:
        return int(self.vllm_port)

    def get_key(self) -> int:
        return self.api_key[:]

    def get_time_idle(self):
        return (time.time()-self.last_accessed) // 60

    def __del__(self):
        try:
            self.container.stop()
        except NotFound:
            pass


class InstanceManager:
    def __init__(
        self, config_directory: str, port_range=DEFAULT_PORT_RANGE,
        max_memory_thr=USED_MEMORY_THRESHOLD, max_idle_time=120
    ):
        self._store: Dict[str, LlmInstance] = {}
        self._known_configs: Dict[str, Dict[str, Any]] = {}
        self._config_stamps: Dict[str, float] = {}
        self._config_dir = Path(config_directory)
        assert self._config_dir.exists(), "config_directory can't be found, please check the path"
        port_ranges = tuple(map(int, port_range.split('-')))
        self._known_ports = set(range(*port_ranges))
        self._max_idle_time = max_idle_time
        self._load_or_update_library()
        self.API_KEY = str(uuid.uuid4())
        self.discard_memory_thr = max_memory_thr
        self.prefix = "dockermanaged_vllm"
        self.remove_possible_orphans()

    def remove_possible_orphans(self):
        client = docker.from_env()
        containers = client.containers.list()
        for c in containers:
            if c.name.startswith(self.prefix):
                c.stop()

    def get_allocated_ports(self):
        return {v.vllm_port for v in self._store.values()}

    def get_vacant_port(self):
        return random.choice(list(self._known_ports - self.get_allocated_ports()))

    def track_new_instance(self, instance: LlmInstance):
        self._store[instance.model_alias] = instance

    def _load_or_update_library(self):
        for p in self._config_dir.glob("*json"):
            config_path_str = p.as_posix()
            ch_time = p.lstat().st_mtime
            recorded_ch_time = self._config_stamps.get(config_path_str, None)
            if recorded_ch_time != ch_time:
                config = read_vllm_config(p)
                self._known_configs[config['model_alias']] = config
                self._config_stamps[config_path_str] = ch_time
        return

    def fetch_known_models(self):
        self._load_or_update_library()
        model_names = sorted(self._known_configs.keys())
        spawned_model_names = set(self._store.keys())
        model_lens = [self._known_configs[mn]['extra_args'].get('max_model_len', None) for mn in model_names]
        status = ["spawned" if mn in spawned_model_names else "offloaded" for mn in model_names]
        return sorted(zip(model_names, model_lens, status), key=lambda x: (x[2] == "offloaded", x[1]))

    def fetch_spawned_models(self):
        self.remove_idle_or_crashed_instances()
        model_names = sorted(self._store.keys())
        model_lens = [self._known_configs[mn]['extra_args'].get('max_model_len', None) for mn in model_names]
        return list(zip(model_names, model_lens))

    def try_spawn_by_alias(self, model_alias):
        # Check if model_alias is registered in the system
        self._load_or_update_library()
        if model_alias not in self._known_configs:
            return False
        config = self._known_configs[model_alias]

        # Remove lost dockers
        self.remove_idle_or_crashed_instances(remove_idle=False)
        # No need for spawning
        if model_alias in self._store:
            return True

        # Check if we have gpus to spawn new docker
        gpu_ids = self.get_gpu_ids(config)
        # Try removing idle containers to free up space
        if not gpu_ids:
            self.remove_idle_or_crashed_instances()
        spawned = self.spawn_docker(config)
        return spawned

    def fetch_instance_port(self, model_alias):
        if model_alias not in self._known_configs:
            return (
                f"{model_alias} is not registered in the system. Please check available models.",
                None,
                None
            )
        self.try_spawn_by_alias(model_alias)
        if model_alias not in self._store:
            return (
                f"{model_alias} can't be deployed at this time. Please consult your server administrator.",
                None,
                None
            )
        i = self._store[model_alias]
        i.reset_access_timer()
        return ("OK", i.get_port(), i.get_key())

    def remove_idle_or_crashed_instances(self, remove_idle=True):
        for k in list(self._store.keys()):
            v = self._store[k]
            print(v, v.check_health(), v.get_time_idle(), self._max_idle_time)
            if (
                (
                    not v.check_health()
                ) or (
                    remove_idle and v.get_time_idle() > self._max_idle_time
                )
            ):
                del v
                self._store.pop(k)

    def purge_all_instances(self):
        for k in list(self._store.keys()):
            v = self._store.pop(k)
            del v

    def get_gpu_ids(self, config):
        gpu_ids = find_gpu_ids(config["gpu_needed"], self.discard_memory_thr)
        if gpu_ids is None:
            return []
        gpu_ids = list(map(str, gpu_ids))
        return gpu_ids

    def spawn_docker(self, config: Dict[str, Any], startup_time: int = 60):
        gpu_ids = self.get_gpu_ids(config)
        # No gpus to spawn new docker
        if not gpu_ids:
            return False
        gpu = docker.types.DeviceRequest(device_ids=gpu_ids, capabilities=[['gpu']])
        tp, pp = get_gpu_breakdown(config["gpu_needed"])

        port = self.get_vacant_port()

        mount = docker.types.Mount(
            target=DEFAULT_DOCKER_MODEL_DIR, source=config['model_parent_dir'], type='bind'
        )

        extra_args = " ".join(f"--{k} {v}" for k, v in config["extra_args"].items())

        api_key = self.API_KEY
        command = f"--model {DEFAULT_DOCKER_MODEL_DIR}/{config['model_name']} --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --gpu-memory-utilization {config['min_util']} {extra_args} --api-key {api_key}"

        client = docker.from_env()

        name_str = f"{self.prefix}__{config['model_alias']}"

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

        instance = LlmInstance(config['model_alias'], port, api_key, container)

        # Make sure container started
        # TODO make dynamic startup check
        time.sleep(startup_time)
        if not instance.check_health():
            del instance
            return False

        self.track_new_instance(instance)
        return True
