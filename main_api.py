import time
import logging
from fastapi.responses import JSONResponse, Response, StreamingResponse
from src.utils import load_global_config, make_error, to_fastapi_response, extract_request_details, with_cancellation
from src.rate_limiter import RateLimiter
from celery_tasks import send_vllm_request
import json
import asyncio
from http import HTTPStatus
from src.api_database import Database
import copy
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import _StreamingResponse
import redis.asyncio as aioredis

from celery_tasks import celery_app

logger = logging.getLogger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds

db_path = "./database/generic.db"
api_db = Database(db_path)
router = APIRouter()
CFG_CELERY = load_global_config()['celery_config']
rate_limiter = RateLimiter(redis_url=f"redis://localhost:{CFG_CELERY['redis_port']}")

TIME_TO_EXPIRE = 600


async def wait_for_task(task):
    timeout = time.time() + TIME_TO_EXPIRE
    while time.time() < timeout:
        await asyncio.sleep(1)
        if task.state == 'SUCCESS':
            return task.result
        if task.state == 'FAILURE':
            return make_error(f"Task failed: {task.result}")
        if task.state == 'REVOKED':
            return make_error("Task was revoked")
    return make_error(f"Task timed out with state {task.state}")


async def post_to_queue(raw_request, command):
    token = extract_auth_token(raw_request)
    _, _, priority, _, _, _, _ = api_db.check_user_key(token)
    stream = False
    try:
        raw_body = await raw_request.body()
        if raw_body:
            request_json = await raw_request.json()
            stream = request_json.get('stream', False)
        else:
            request_json = {}
    except json.decoder.JSONDecodeError:
        return JSONResponse(
            content={"error": {"message": "Invalid JSON in request body"}},
            status_code=400,
        )
    request_json = {"command": command, "args": request_json}
    task = send_vllm_request.apply_async(
        args=[request_json], priority=priority,
        expires=TIME_TO_EXPIRE,
    )
    stream_task_id = task.task_id
    if not stream:
        res = await wait_for_task(task)
        res = json.loads(res)

        return res
    else:
        async def generate():
            redis_url = f"redis://localhost:{CFG_CELERY['redis_port']}"
            redis = aioredis.from_url(redis_url)
            stream_key = f"stream:{stream_task_id}"
            had_error = False

            try:
                await asyncio.sleep(0.1)
                last_processed = -1

                max_wait_time = TIME_TO_EXPIRE
                start_time = time.time()

                status = "PENDING"
                while time.time() - start_time < max_wait_time:
                    status = await redis.hget(stream_key, "status")

                    if not status:
                        await asyncio.sleep(0.1)
                        continue

                    status = status.decode('utf-8')

                    if status == "FAILED":
                        error_data = await redis.hget(stream_key, "error")
                        error = error_data.decode('utf-8') if error_data else "Unknown error"
                        logger.warning("Streaming failed: %s", error)
                        had_error = True
                        error_json = json.dumps(
                            {"error": {'message': error}}
                        )
                        yield f"data: {error_json}\n\n"
                        break

                    last_chunk_str = await redis.hget(stream_key, "last_chunk")
                    if last_chunk_str:
                        last_chunk = int(last_chunk_str.decode('utf-8'))

                        for i in range(last_processed + 1, last_chunk + 1):
                            chunk_data = await redis.hget(stream_key, f"chunk:{i}")
                            if chunk_data:
                                chunk_json = chunk_data.decode('utf-8')
                                yield f"data: {chunk_json}\n\n"

                        last_processed = last_chunk

                    if status == "COMPLETED" and last_processed == last_chunk:
                        break

                    await asyncio.sleep(0.05)

                if not had_error:
                    yield "data: [DONE]\n\n"
            finally:
                await redis.delete(stream_key)
                await redis.aclose()

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
    return JSONResponse(content={"error": "Not Implemented"}, status_code=501)


@router.post("/detokenize")
async def detokenize(request, raw_request: Request):
    return JSONResponse(content={"error": "Not Implemented"}, status_code=501)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    token = extract_auth_token(raw_request)
    _, _, _, allowed_models, _, _, _ = api_db.check_user_key(token)
    result = await post_to_queue(raw_request, 'model_list')
    if allowed_models is not None:
        allowed = json.loads(allowed_models) if isinstance(allowed_models, str) else allowed_models
        if isinstance(allowed, list) and len(allowed) > 0:
            if isinstance(result, dict) and 'data' in result:
                result['data'] = [m for m in result['data'] if m.get('id') in allowed]
            elif isinstance(result, StreamingResponse):
                pass
    return result


@router.get("/version")
async def show_version():
    return JSONResponse(content={"error": "Not Implemented"}, status_code=501)


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
    return JSONResponse(content={"error": "Not Implemented"}, status_code=501)


app = FastAPI()
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    return JSONResponse(
        content={"error": {"message": str(exc)}},
        status_code=HTTPStatus.BAD_REQUEST,
    )


@app.post("/internal/clear-cache")
async def internal_clear_cache(request: Request):
    data = await request.json()
    user_key = data.get("user_key")
    if user_key:
        api_db.clear_user_key_cache_by_key(user_key)
    return {"status": "ok"}


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
        self.had_error = False
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None

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
                        if 'error' in msg:
                            self.had_error = True
                        else:
                            isgen = 'choices' in msg
                            data_key = 'delta'
                            self.usage += int(isgen)
                            if 'usage' in msg and msg['usage']:
                                self.prompt_tokens = msg['usage'].get('prompt_tokens')
                                self.completion_tokens = msg['usage'].get('completion_tokens')
                                self.total_tokens = msg['usage'].get('total_tokens')
                else:
                    msg = json.loads(chunk_str)
                    isgen = 'choices' in msg
                    data_key = 'message'

                if isgen:
                    self.data += msg['choices'][0][data_key].get('content') or ''
                    self.last_finish_reason = msg['choices'][0]['finish_reason']
                    if 'model' in msg:
                        if self.model_name is None:
                            self.model_name = msg['model']

                        if self.response_dict is None:
                            self.response_dict = msg
            except Exception as e:
                logger.warning("Failed to parse streaming chunk: %s | chunk: %s", e, chunk_str, exc_info=True)
            return chunk
        except StopAsyncIteration:
            try:
                if not self.had_error and self.response_dict is not None:
                    if 'delta' in self.response_dict['choices'][0]:
                        assert len(self.response_dict['choices']) == 1
                        self.response_dict['choices'][0]['message'] = copy.deepcopy(
                            self.response_dict['choices'][0]['delta'])
                        del self.response_dict['choices'][0]['delta']
                        self.response_dict['choices'][0]['message']['content'] = self.data
                        self.response_dict['choices'][0]['finish_reason'] = self.last_finish_reason
                        self.response_dict['usage'] = {'completion_tokens': self.usage}

                    # Fallback: if model didn't report usage in streaming, use our counter
                    effective_completion = self.completion_tokens or self.usage
                    effective_total = self.total_tokens or effective_completion

                    api_db.save_response(
                        self.request_dict, self.response_dict, self.user_id, self.model_name,
                        prompt_tokens=self.prompt_tokens, completion_tokens=effective_completion,
                        total_tokens=effective_total,
                    )
                    if effective_total:
                        api_db.increment_token_usage(self.user_id, effective_total)
                        await rate_limiter.increment(self.user_id, effective_total)
                    api_db.increment_request_count(self.user_id)
            except Exception as e:
                logger.warning("Failed to save response log: %s", e, exc_info=True)
            raise


@app.middleware("http")
async def authentication(request: Request, call_next):
    request_dict = await extract_request_details(request)
    if request.method == "OPTIONS" or request.url.path.startswith("/internal/"):
        return await call_next(request)

    token = extract_auth_token(request)
    exists, user_id, priority, allowed_models, token_budget, total_tokens_used, rate_limits = api_db.check_user_key(token)
    if not exists:
        return JSONResponse(content={"error": "Unauthorized"},
                            status_code=401)

    # Enforce token budget (F4)
    if token_budget is not None and total_tokens_used >= token_budget:
        return JSONResponse(
            content={"error": {"message": "Token budget exhausted", "used": total_tokens_used, "budget": token_budget}},
            status_code=429,
        )

    # Enforce rate limits (F5)
    rate_ok, rate_msg = await rate_limiter.check(user_id, rate_limits)
    if not rate_ok:
        return JSONResponse(
            content={"error": {"message": rate_msg}},
            status_code=429,
            headers={"Retry-After": "60"},
        )

    # Enforce allowed_models restriction
    if allowed_models is not None:
        allowed = json.loads(allowed_models) if isinstance(allowed_models, str) else allowed_models
        if isinstance(allowed, list) and len(allowed) > 0:
            body = request_dict.get('body')
            if body is not None:
                requested_model = body.get('model')
                if requested_model and requested_model not in allowed:
                    return JSONResponse(
                        content={"error": {"message": f"Model '{requested_model}' is not allowed for this token"}},
                        status_code=403,
                    )

    response = await call_next(request)
    stream = False
    if request_dict.get('body', None) is not None:
        stream = request_dict['body'].get('stream', False)

    if stream and (isinstance(response, StreamingResponse) or isinstance(response, _StreamingResponse)):
        response.body_iterator = LoggingIterator(response.body_iterator, request_dict, user_id)
    else:
        response = await to_fastapi_response(response)
        response_dict = json.loads(response.body)
        if response_dict is not None:
            if "status_code" in response_dict and response_dict["status_code"] >= 400:
                return JSONResponse(
                    content=response_dict['content']['error'],
                    status_code=response_dict["status_code"]
                )
            if 'status_code' not in response_dict and request_dict['body'] is not None:
                model_name = request_dict['body'].get('model', None)
                usage = response_dict.get('usage')
                prompt_t = completion_t = total_t = None
                if usage:
                    prompt_t = usage.get('prompt_tokens')
                    completion_t = usage.get('completion_tokens')
                    total_t = usage.get('total_tokens')
                api_db.save_response(
                    request_dict, response_dict, user_id, model_name,
                    prompt_tokens=prompt_t, completion_tokens=completion_t, total_tokens=total_t,
                )
                if total_t:
                    api_db.increment_token_usage(user_id, total_t)
                    await rate_limiter.increment(user_id, total_t)
                api_db.increment_request_count(user_id)

    return response
