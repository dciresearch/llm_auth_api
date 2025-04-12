import re
from http import HTTPStatus
from functools import lru_cache
import time
from fastapi import Request
import functools
import asyncio
from fastapi import Response
import random
import hashlib
import yaml
import json


def load_global_config():
    with open("config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def get_key_hash(k):
    return hashlib.sha256(k.encode('utf-8')).hexdigest()


def shuffle_string(text, seed=1234):
    tmp = list(text)
    random.seed(seed)
    random.shuffle(tmp)
    return "".join(tmp)


OPENAI_ERROR_PATTERN = re.compile("Error code: (?:\d{3,4}) - (.+)")


def extract_openai_error(text):
    if isinstance(text, str) and text.startswith("Error code"):
        text = OPENAI_ERROR_PATTERN.findall(text)[0]
        if text.startswith("{"):
            text = text.replace("{'", '{"').replace("':", '":').replace(
                ", '", ', "').replace(": '", ': "').replace("', \"", '", "').replace(": None", ": null")
            text = json.loads(text)["message"]
    return text


def make_error(msg, code=HTTPStatus.BAD_REQUEST):
    msg = extract_openai_error(msg)
    json_dict = {
        'content': {"error": msg},
        'status_code': code
    }
    return json.dumps(json_dict, ensure_ascii=False)


async def get_streaming_response_body(response) -> bytes:
    return b"".join([chunk async for chunk in response.body_iterator])


async def to_fastapi_response(response):
    return Response(
        content=await get_streaming_response_body(response),
        status_code=response.status_code,
        headers=response.headers,
        media_type=response.media_type,
        background=response.background
    )


async def extract_request_details(request):
    raw_body = await request.body()
    body = None
    if raw_body:
        body = json.loads(raw_body)
    return {"url": request.url._url, "body": body}


def get_ttl_hash(seconds=3600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


def ttl_classcache(ttl=5):
    def ttl_cacher(func):
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "__ttl_cache"):
                self.__ttl_cache = {}
            fn = func.__name__
            if fn not in self.__ttl_cache or time.time() - self.__ttl_cache[fn][0] > ttl:
                cache = func(self, *args, **kwargs)
                self.__ttl_cache[fn] = (time.time(), cache)
                return cache
            else:
                return self.__ttl_cache[fn][1]
        return wrapper
    return ttl_cacher


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from 
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):

        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                           return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper
