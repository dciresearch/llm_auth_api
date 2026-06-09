import time
import subprocess
import shlex
from src.utils import load_global_config
import signal
import atexit
import sys
import redis
import httpx


CFG = load_global_config()['celery_config']
ADMIN_CFG = load_global_config().get('admin_config', {})
MANAGER_CFG = load_global_config().get('manager_config', {})
DEBUG = load_global_config().get('debug', False)

DETACHED_PROCESS = 0x00000008

reload_flag = " --reload" if DEBUG else ""
manager_command = f"uvicorn manager_server:app --port {CFG['manager_port']} --host 0.0.0.0{reload_flag}"
worker_command = "celery -A celery_tasks worker -l info --without-gossip --pool=threads --concurrency=10000"
app_command = f"uvicorn main_api:app --port {CFG['app_port']} --host 0.0.0.0{reload_flag}"
dbapi_command = f"sqlite_web -p {CFG['dbapi_port']} --host 127.0.0.1 ./database/generic.db"
admin_port = ADMIN_CFG.get('admin_port', 6334)
admin_command = f"uvicorn admin_panel:app --port {admin_port} --host 0.0.0.0{reload_flag}"
manager_port = CFG['manager_port']
manager_secret = MANAGER_CFG.get('manager_secret', '')


children = []


def terminate_children():
    """Send SIGTERM to all children and wait briefly."""
    for p in children:
        try:
            p.terminate()
        except Exception:
            pass
    deadline = time.time() + 5
    for p in children:
        remaining = max(0.1, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass


def kill_children():
    """Force kill any remaining children."""
    for p in children:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def graceful_shutdown():
    """SIGINT (Ctrl+C): tell manager to kill containers, then stop everything."""
    try:
        headers = {}
        if manager_secret:
            headers["Authorization"] = f"Bearer {manager_secret}"
        httpx.post(
            f"http://localhost:{manager_port}/shutdown",
            headers=headers,
            timeout=10
        )
    except Exception:
        pass
    terminate_children()
    kill_children()
    sys.exit(0)


def restart_only():
    """SIGTERM (restart): stop processes but leave containers alive."""
    terminate_children()
    kill_children()
    sys.exit(0)


signal.signal(signal.SIGINT, lambda *a: graceful_shutdown())
signal.signal(signal.SIGTERM, lambda *a: restart_only())


for c in [dbapi_command, manager_command, worker_command, app_command, admin_command]:
    process = subprocess.Popen(shlex.split(c))
    children.append(process)

while True:
    time.sleep(10)
