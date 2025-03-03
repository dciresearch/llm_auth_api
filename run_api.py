import time
import subprocess
import shlex
from src.utils import load_global_config
import signal
import atexit
import sys

CFG = load_global_config()['celery_config']

DETACHED_PROCESS = 0x00000008

manager_command = f"uvicorn manager_server:app --port {CFG['manager_port']} --host 0.0.0.0 --reload"
worker_command = "celery -A celery_tasks worker -l info --pool=threads --concurrency=100"
app_command = f"uvicorn main_api:app --port {CFG['app_port']} --host 0.0.0.0 --reload"
dbapi_command = f"sqlite_web -p {CFG['dbapi_port']} --host 0.0.0.0 ./database/generic.db"


children = []


def clean_manager():
    for p in children:
        p.kill()


def int_handler(*args):
    sys.exit(0)


atexit.register(clean_manager)
signal.signal(signal.SIGTERM, int_handler)
signal.signal(signal.SIGINT, int_handler)


for c in [dbapi_command, manager_command, worker_command, app_command]:
    process = subprocess.Popen(shlex.split(c))
    children.append(process)

while True:
    time.sleep(10)
