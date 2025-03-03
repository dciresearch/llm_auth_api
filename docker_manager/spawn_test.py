import sys
import signal
import atexit
import docker
import json
import time
import docker.types

containers = []


def exit_handler(*args):
    for c in containers:
        c.kill()


def int_handler(*args):
    sys.exit(0)


def read_config(path):
    with open(path) as f:
        res = json.load(f)
        return res


atexit.register(exit_handler)
signal.signal(signal.SIGTERM, int_handler)
signal.signal(signal.SIGINT, int_handler)


path = "./llm_docker_configs/qwen3b.json"

conf = read_config(path)

mount = docker.types.Mount(target="/workdir/models", source=conf['model_parent_dir'], type='bind')

command = "--model /workdir/models/{model_name} --tensor-parallel-size {tp} --gpu-memory-utilization {util}"

command = command.format(model_name=conf['model_name'], tp=conf['tp'], util=conf['min_util'])

port = 1234

gpu = docker.types.DeviceRequest(device_ids=["0"], capabilities=[['gpu']])
client = docker.from_env()

container = client.containers.run(
    'vllm_server',
    command=command,
    detach=True,
    auto_remove=True,
    tty=True,
    mounts=[mount],
    ports={8000: port},
    device_requests=[gpu]
)
containers.append(container)

while True:
    time.sleep(10)
