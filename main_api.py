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
import copy
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import _StreamingResponse

from celery_tasks import celery_app

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
    stream = False
    try:
        request_json = await raw_request.json()
        stream = request_json.get('stream', False)
    except json.decoder.JSONDecodeError:
        request_json = None
    request_json = {"command": command, "args": request_json}
    task = send_vllm_request.apply_async(
            args=[request_json], priority=priority,
            expires=TIME_TO_EXPIRE,  # datetime.datetime.now(tz=pytz.timezone('Etc/GMT+2')) + timedelta(seconds=5),
        )
    stream_task_id = task.task_id
    if not stream:
        res = await wait_for_task(task)
        return json.loads(res)
    else:
        def generate():
            # Получаем Redis-соединение
            redis = celery_app.backend.client
            stream_key = f"stream:{stream_task_id}"
            
            # Ожидаем начала выполнения задачи
            time.sleep(0.1)
            
            # Последний обработанный чанк
            last_processed = -1
            
            # Максимальное время ожидания (30 секунд)
            max_wait_time = 30
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                # Проверяем статус
                status = redis.hget(stream_key, "status")
                
                if not status:
                    # Ключ еще не создан, ждем
                    time.sleep(0.1)
                    continue
                
                status = status.decode('utf-8')
                
                if status == "FAILED":
                    error = redis.hget(stream_key, "error")
                    print(f"Streaming failed: {error}")
                    break
                
                # Получаем индекс последнего чанка
                last_chunk_str = redis.hget(stream_key, "last_chunk")
                if last_chunk_str:
                    last_chunk = int(last_chunk_str.decode('utf-8'))
                    
                    # Обрабатываем новые чанки
                    for i in range(last_processed + 1, last_chunk + 1):
                        chunk_data = redis.hget(stream_key, f"chunk:{i}")
                        if chunk_data:
                            chunk_json = chunk_data.decode('utf-8')
                            #print(f"Sending chunk: {chunk_json}")
                            yield f"data: {chunk_json}\n\n"
                            #yield f"data: {chunk_data.decode('utf-8')}\n\n"
                    
                    # Обновляем последний обработанный чанк
                    last_processed = last_chunk
                
                # Если задача завершена и все чанки обработаны
                if status == "COMPLETED" and last_processed == last_chunk:
                    break
                
                time.sleep(0.05)
            
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

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

class LoggingIterator:
    def __init__(self, iterator, request_dict, user_id):
        self.iterator = iterator
        self.data = ''
        self.request_dict = request_dict
        self.user_id = user_id
        self.model_name = None
        self.response_dict = None
        self.last_finish_reason = None
        self.usage = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            isgen = False
            chunk = await self.iterator.__anext__()
            chunk_str = chunk.decode('utf-8', errors='replace')
            try:
                if chunk_str.startswith('data: '):
                    if chunk_str.strip() != 'data: [DONE]':
                        msg = json.loads(chunk_str[len('data: '):])
                        isgen = 'choices' in msg
                        data_key = 'delta'
                        self.usage += int(isgen)
                else:
                    msg = json.loads(chunk_str)
                    isgen = 'choices' in msg
                    data_key = 'message'

                if isgen:
                    self.data += msg['choices'][0][data_key]['content']
                    self.last_finish_reason = msg['choices'][0]['finish_reason']
                    if 'model' in msg:
                        if self.model_name is None:
                            self.model_name = msg['model']

                        if self.response_dict is None:
                            self.response_dict = msg
            except Exception as e:
                print('what:', chunk_str)
                print(e)
                pass
            return chunk
        except StopAsyncIteration:
            try:
                if self.response_dict is not None:
                    if 'delta' in self.response_dict['choices'][0]:
                        assert len(self.response_dict['choices']) == 1
                        self.response_dict['choices'][0]['message'] = copy.deepcopy(self.response_dict['choices'][0]['delta'])
                        del self.response_dict['choices'][0]['delta']
                        self.response_dict['choices'][0]['message']['content'] = self.data
                        self.response_dict['choices'][0]['finish_reason'] = self.last_finish_reason
                        self.response_dict['usage'] = {'completion_tokens': self.usage}

                    api_db.save_response(self.request_dict, self.response_dict, self.user_id, self.model_name)
            except Exception as e:
                print(e)
                pass
            raise

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
    if isinstance(response, StreamingResponse) or isinstance(response, _StreamingResponse):
        response.body_iterator = LoggingIterator(response.body_iterator, request_dict, user_id)
    else:
        ## not really happends ever?
        response = await to_fastapi_response(response)
        response_dict = json.loads(response.body)
        if 'status_code' not in response_dict and request_dict['body'] is not None:
            model_name = request_dict['body'].get('model', None)
            api_db.save_response(request_dict, response_dict, user_id, model_name)

    return response