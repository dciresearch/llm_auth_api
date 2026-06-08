import json
import dataclasses
import logging
from typing import Union, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


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
    tags: List[str] = dataclasses.field(default_factory=list)
    remote_url: str = None
    remote_key: str = None
    spawn_script: str = None

    def __init__(self, **kwargs):
        names = {f.name for f in dataclasses.fields(self)}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

    @classmethod
    def from_path(cls, path):
        try:
            config = read_config(path)
        except Exception as e:
            logger.error("Failed to read config from %s: %s", path, e)
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
    extra_args: dict = dataclasses.field(default_factory=dict)
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
        raise EnvironmentError(f"{self.__class__.__name__} is not designed to be instantiated directly")

    @classmethod
    def from_config(cls, config):
        cfg_cls = config_type_map[config['config_type']]
        return cfg_cls(**config)

    @classmethod
    def from_path(cls, path):
        try:
            config = read_config(path)
        except Exception as e:
            logger.error("Failed to read config from %s: %s", path, e)
            return
        return cls.from_config(config)
