import json
import dataclasses
from typing import Union, Dict, Any, List, Tuple


def read_json_config(path):
    with open(path) as f:
        res = json.load(f)
        return res


def read_config(path):
    config = read_json_config(path)
    if 'extra_args' not in config:
        config['extra_args'] = {}
    return config


@dataclasses.dataclass(init=False)
class GenericDockerConfig:
    alias: str
    gpu_needed: int = 1
    ports_needed: int = 1
    config_type: str = None
    max_idle_time = None
    tags: List[str] = lambda: []
    remote_url: str = None
    remote_key: str = None
    spawn_script: str = None

    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_path(cls, path):
        try:
            config = read_config(path)
        except Exception as e:
            print(e)
            return
        return cls(**config)

    def __getitem__(self, key):
        return getattr(self, key)


@dataclasses.dataclass(init=False)
class VllmConfig(GenericDockerConfig):
    model_alias: str
    model_name: str = None
    model_parent_dir: str = None
    min_util: float = 0.95
    max_model_len: int = -1
    extra_args: dict = lambda: {}
    use_v1: bool = False
    spawn_script: str = "vllm_default"
    alias: str = None

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
        self.max_model_len = self.extra_args.get("max_model_len", self.max_model_len)
        self.alias = self.model_alias


config_type_map = {
    "vllm": VllmConfig
}


class AutoConfig:
    def __init__(self, **config):
        raise EnvironmentError(f"{self.__class__.__name__} is designed to be instantiated ")

    @classmethod
    def from_config(self, config):
        cls = config_type_map[config['config_type']]
        return cls(**config)

    @classmethod
    def from_path(self, path):
        try:
            config = read_config(path)
        except Exception as e:
            print(e)
            return
        return self.from_config(config)
