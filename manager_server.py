import atexit
import sys
import signal
import os
import asyncio
import logging
from functools import partial
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from docker_manager.docker_store import InstanceManager
from src.utils import load_global_config
from src.api_database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)


router = APIRouter()

CFG = load_global_config()['manager_config']
MANAGER_SECRET = CFG.get('manager_secret', '')

db_path = "./database/generic.db"
api_db = Database(db_path)

manager = InstanceManager(
    "./llm_docker_configs",
    max_memory_thr=CFG['max_used_memory_per_gpu'],
    default_idle_time=CFG['default_idle_time'],
    api_db=api_db
)


def clean_manager():
    manager.purge_all_instances()


def int_handler(*args):
    sys.exit(0)


signal.signal(signal.SIGTERM, int_handler)
signal.signal(signal.SIGINT, int_handler)


@router.get("/library")
async def fetch_library_model_list():
    return {"models": await manager.fetch_known_models()}


@router.get("/spawned")
async def fetch_spawned_model_list():
    await manager.remove_idle_or_crashed_instances_async()
    model_aliases = sorted(manager._store.keys())
    model_lens = [manager._known_configs[mn].max_model_len
                  if mn in manager._known_configs else -1
                  for mn in model_aliases]
    return {"models": list(zip(model_aliases, model_lens))}


@router.get("/models")
async def fetch_model_url(model_alias: str):
    msg, url, api_key = await manager.fetch_instance_url(model_alias)
    return {"message": msg, "url": url, "key": api_key}


@router.get("/instances")
async def fetch_instances():
    await manager.remove_idle_or_crashed_instances_async(remove_idle=False)
    snapshot = list(manager._store.items())

    def build_report(items):
        result = []
        for name, inst in items:
            result.append({
                "name": name,
                "url": inst.api_url,
                "is_virtual": inst.is_virtual,
                "health": inst.check_health(),
                "idle_minutes": int(inst.get_time_idle()),
                "max_idle_minutes": inst.max_idle_time,
                "expired": inst.expired(),
                "gpu_ids": inst.gpu_ids,
                "gpu_info": manager.get_gpu_info(inst.gpu_ids),
            })
        return result

    result = await asyncio.to_thread(build_report, snapshot)
    return {"instances": result}


@router.post("/shutdown")
async def shutdown():
    """Kill all containers and exit. Called by run_api.py on SIGINT."""
    clean_manager()
    os._exit(0)

app = FastAPI()
app.include_router(router)


@app.middleware("http")
async def manager_auth(request: Request, call_next):
    if not MANAGER_SECRET:
        return await call_next(request)
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    # Allow /shutdown without auth for internal use
    if request.url.path == "/shutdown":
        return await call_next(request)
    if token != MANAGER_SECRET:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
