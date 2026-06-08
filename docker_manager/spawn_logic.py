from .config_patterns import GenericDockerConfig, VllmConfig
import docker
import dataclasses


@dataclasses.dataclass
class DockerLaunchArgs:
    alias: str
    docker_name: str = None
    command: str = None
    mounts: list = None
    env_args: str = None
    port_map: dict = None
    instance_name: str = None
    api_url: str = None
    api_key: str = None


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


def get_vllm_docker_spawn_args(config: VllmConfig, ports, API_KEY, docker_name = 'vllm_server'):
    args = DockerLaunchArgs(config.model_alias)
    if config.remote_url is not None:
        args.api_url = config.remote_url
        args.api_key = config.remote_key
        config.gpu_needed = 0
        return args

    args.docker_name = docker_name

    mount = docker.types.Mount(
        target="/workdir/models", source=config.model_parent_dir, type='bind'
    )
    args.mounts = [mount]

    args.port_map = {8000: ports[0]}
    args.instance_name = config.model_alias.replace('/', '_')
    args.api_url = f"http://localhost:{ports[0]}"
    args.api_key = API_KEY

    tp, pp = get_gpu_breakdown(config.gpu_needed)
    extra_args = " ".join(f"--{k} {v}" for k, v in config.extra_args.items())
    args.command = f"--model /workdir/models/{config.model_name} --tensor-parallel-size {tp} --pipeline-parallel-size {pp} --gpu-memory-utilization {config.min_util} {extra_args} --api-key {args.api_key} --max_num_seqs 256"
    if not config.use_v1:
        args.env_args = ["VLLM_USE_V1=0"]

    return args

def get_vllm_docker_spawn_args_experimental(*args, **kwargs):
    return get_vllm_docker_spawn_args(*args, **kwargs, docker_name = "vllm/vllm-openai:nightly")

spawner_scripts = {
    "vllm_default": get_vllm_docker_spawn_args,
    "vllm_experimental": get_vllm_docker_spawn_args_experimental
}
