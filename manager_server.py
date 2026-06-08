import atexit
import sys
import signal
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from docker_manager.docker_store import InstanceManager
from src.utils import load_global_config


router = APIRouter()

CFG = load_global_config()['manager_config']
MANAGER_SECRET = CFG.get('manager_secret', '')
manager = InstanceManager(
    "./llm_docker_configs",
    max_memory_thr=CFG['max_used_memory_per_gpu'],
    default_idle_time=CFG['default_idle_time']
)


def clean_manager():
    manager.purge_all_instances()


def int_handler(*args):
    sys.exit(0)


atexit.register(clean_manager)
signal.signal(signal.SIGTERM, int_handler)
signal.signal(signal.SIGINT, int_handler)


@router.get("/library")
async def fetch_library_model_list():
    return {"models": await manager.fetch_known_models()}


@router.get("/spawned")
async def fetch_spawned_model_list():
    return {"models": manager.fetch_spawned_models()}


@router.get("/models")
async def fetch_model_url(model_alias: str):
    msg, url, api_key = await manager.fetch_instance_url(model_alias)
    return {"message": msg, "url": url, "key": api_key}

app = FastAPI()
app.include_router(router)


@app.middleware("http")
async def manager_auth(request: Request, call_next):
    if not MANAGER_SECRET:
        return await call_next(request)
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if token != MANAGER_SECRET:
        return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
