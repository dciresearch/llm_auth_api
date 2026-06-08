import time
import subprocess
import shlex
from src.utils import load_global_config
import signal
import atexit
import sys
import redis


CFG = load_global_config()['celery_config']
ADMIN_CFG = load_global_config().get('admin_config', {})
DEBUG = load_global_config().get('debug', False)

DETACHED_PROCESS = 0x00000008

reload_flag = " --reload" if DEBUG else ""
manager_command = f"uvicorn manager_server:app --port {CFG['manager_port']} --host 0.0.0.0{reload_flag}"
worker_command = "celery -A celery_tasks worker -l info --without-gossip --pool=threads --concurrency=10000"
app_command = f"uvicorn main_api:app --port {CFG['app_port']} --host 0.0.0.0{reload_flag}"
dbapi_command = f"sqlite_web -p {CFG['dbapi_port']} --host 127.0.0.1 ./database/generic.db"
admin_port = ADMIN_CFG.get('admin_port', 6334)
admin_command = f"uvicorn admin_panel:app --port {admin_port} --host 0.0.0.0{reload_flag}"


children = []
def clean_manager():
    for p in children:
        p.kill()


def int_handler(*args):
    sys.exit(0)


atexit.register(clean_manager)
signal.signal(signal.SIGTERM, int_handler)
signal.signal(signal.SIGINT, int_handler)


for c in [dbapi_command, manager_command, worker_command, app_command, admin_command]:
    process = subprocess.Popen(shlex.split(c))
    children.append(process)

while True:
    time.sleep(10)
