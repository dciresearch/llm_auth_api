import sys
sys.path.insert(0, "/workdir")
from src.utils import load_global_config
import httpx

cfg = load_global_config()
port = cfg['celery_config']['manager_port']
secret = cfg['manager_config'].get('manager_secret', '')

headers = {}
if secret:
    headers["Authorization"] = f"Bearer {secret}"

r = httpx.get(f"http://localhost:{port}/instances", headers=headers, timeout=10)
instances = r.json().get("instances", [])

if not instances:
    print("No running instances")
else:
    print(f"{'Name':<35} {'Health':<8} {'Idle':<8} {'Max Idle':<10} {'Virtual':<8} {'Expired'}")
    print("-" * 90)
    for i in instances:
        print(f"{i['name']:<35} {str(i['health']):<8} {i['idle_minutes']}m{'':<6} {i['max_idle_minutes']}m{'':<8} {str(i['is_virtual']):<8} {i['expired']}")
