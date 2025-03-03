import time
from fastapi.responses import JSONResponse, Response, StreamingResponse
from src.utils import load_global_config, make_error, to_fastapi_response, extract_request_details, with_cancellation
from celery_tasks import send_vllm_request
import json
import asyncio
import importlib
import inspect
import tempfile
from argparse import Namespace
from http import HTTPStatus
from typing import AsyncIterator, Set
from src.api_database import Database

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware


TIMEOUT_KEEP_ALIVE = 5  # seconds


prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
# logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()

db_path = "./database/generic.db"
api_db = Database(db_path)
router = APIRouter()

TIME_TO_EXPIRE = 600


async def wait_for_task(task):
    timeout = time.time() + TIME_TO_EXPIRE
    while time.time() < timeout:
        await asyncio.sleep(1)
        # print(task.id, task.state)
        if task.state == 'SUCCESS':
            return task.result
    return make_error(f"Task failed with state {task.state}")


async def post_to_queue(raw_request, command):
    token = extract_auth_token(raw_request)
    _, _, priority = api_db.check_user_key(token)

    try:
        request_json = await raw_request.json()
    except json.decoder.JSONDecodeError:
        request_json = None
    request_json = {"command": command, "args": request_json}

    task = send_vllm_request.apply_async(
        args=[request_json], priority=priority,
        expires=TIME_TO_EXPIRE,  # datetime.datetime.now(tz=pytz.timezone('Etc/GMT+2')) + timedelta(seconds=5),
    )
    res = await wait_for_task(task)
    return json.loads(res)


@router.get("/health")
async def health(raw_request: Request):
    """Health check."""

    return await post_to_queue(raw_request, 'check_health')


@router.post("/tokenize")
async def tokenize(request, raw_request: Request):
    pass


@router.post("/detokenize")
async def detokenize(request, raw_request: Request):
    pass


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    return await post_to_queue(raw_request, 'model_list')


@router.get("/version")
async def show_version():
    pass


@router.post("/v1/chat/completions")
@with_cancellation
async def create_chat_completion(raw_request: Request):
    return await post_to_queue(raw_request, 'chat_completion')


@router.post("/v1/completions")
@with_cancellation
async def create_completion(request, raw_request: Request):
    return await post_to_queue(raw_request, 'completion')


@router.post("/v1/embeddings")
async def create_embedding(request, raw_request: Request):
    pass


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None)
    else:
        app = FastAPI()
    app.include_router(router)
    app.root_path = args.root_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        chat = app.state.openai_serving_chat
        err = chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    # if token := envs.VLLM_API_KEY or args.api_key:

    #     @app.middleware("http")
    #     async def authentication(request: Request, call_next):
    #         root_path = "" if args.root_path is None else args.root_path
    #         if request.method == "OPTIONS":
    #             return await call_next(request)
    #         if not request.url.path.startswith(f"{root_path}/v1"):
    #             return await call_next(request)
    #         if request.headers.get("Authorization") != "Bearer " + token:
    #             return JSONResponse(content={"error": "Unauthorized"},
    #                                 status_code=401)
    #         return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


app = FastAPI()
app.include_router(router)


def extract_auth_token(request):
    if request.headers.get("Authorization") is None:
        return None
    return request.headers.get("Authorization").removeprefix("Bearer ")


@app.middleware("http")
async def authentication(request: Request, call_next):
    request_dict = await extract_request_details(request)
    if request.method == "OPTIONS":
        return await call_next(request)

    token = extract_auth_token(request)
    exists, user_id, priority = api_db.check_user_key(token)
    if not exists:
        return JSONResponse(content={"error": "Unauthorized"},
                            status_code=401)

    response = await call_next(request)
    response = await to_fastapi_response(response)
    response_dict = json.loads(response.body)

    if 'status_code' not in response_dict and request_dict['body'] is not None:
        model_name = request_dict['body'].get('model', None)
        api_db.save_response(request_dict, response_dict, user_id, model_name)
    return response
