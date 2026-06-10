from src.utils import ttl_classcache, get_url
from openai import APIConnectionError, OpenAIError
import requests
import json
from fastapi.responses import JSONResponse
from inspect import signature
from celery import Celery, Task
from time import sleep
from openai import OpenAI
from kombu import Exchange, Queue
from src.utils import load_global_config, make_error, extract_openai_error
import logging
import httpx

logger = logging.getLogger(__name__)

CFG = load_global_config()['celery_config']
MANAGER_SECRET = load_global_config()['manager_config'].get('manager_secret', '')

broker_url = f"amqp://localhost:{CFG['rabitmq_port']}"
redis_url = f"redis://localhost:{CFG['redis_port']}"
celery_app = Celery('vllm_queue', broker=broker_url, backend=redis_url)

celery_app.conf.task_queues = [
    Queue('tasks', Exchange('tasks'), routing_key='tasks',
          queue_arguments={'x-max-priority': 10}),
]
celery_app.conf.task_queue_max_priority = 10
celery_app.conf.task_default_priority = 5
celery_app.conf.task_default_queue = 'tasks'
celery_app.conf.task_acks_late = True
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.update(
    timezone='GMT',
)
celery_app.control.rate_limit('celery_tasks.send_vllm_request', '1000/s')

logger = logging.getLogger(__name__)


def fetch_client_by_url(url, api_key=None):
    openai_api_key = api_key
    if openai_api_key is None:
        openai_api_key = "EMPTY"
    openai_api_base = f"{url}/v1"
    timeout = httpx.Timeout(
        timeout=1800,
        connect=5.0
    )
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, max_retries=0, timeout=timeout)
    try:
        model_list = client.models.list()
    except (APIConnectionError, OpenAIError):
        return None, None
    if len(model_list.data) == 1:
        client_name = model_list.data[0].id
    else:
        client_name = None
    return client, client_name


class VllmTask(Task):
    def __init__(self):
        self._clients = None
        self._models = []
        self.manager_port = CFG['manager_port']
        self.manager_host = CFG['manager_host']
        if self.manager_host is None:
            self.manager_host = "localhost"
        if self.manager_host.startswith("http"):
            self.manager_host = self.manager_host.split("//", maxsplit=1)[1]
        self.manager_url = get_url(self.manager_host, self.manager_port)

    def query_manager(self, query_type, **kwargs):
        url = f"{self.manager_url}/{query_type}"
        headers = {}
        if MANAGER_SECRET:
            headers["Authorization"] = f"Bearer {MANAGER_SECRET}"
        res = requests.get(url, params=kwargs, headers=headers, timeout=660).json()
        return res

    def check_health(self):
        try:
            res = self.query_manager('library')
            models = res.get('models', [])
            status = []
            for m in models:
                name, _, state = m
                status.append({"model": name, "status": state})
            return json.dumps(status)
        except Exception as e:
            return make_error(f"Health check failed: {e}")

    @property
    def clients(self):
        return self._clients

    # @ttl_classcache(ttl=30)
    def fetch_client(self, model_alias):
        res = self.query_manager('models', model_alias=model_alias)
        if res['url'] is None:
            logger.warning("Model %s unavailable: %s", model_alias, res.get("message"))
            return None, None

        # Since we get raw response the local ports of docker manager
        # would be passed as the local ports of this celery machine
        # which won't be correct if docker manager is remote
        res['url'] = res['url'].replace("localhost", self.manager_host)

        client, c_name = fetch_client_by_url(res['url'], res['key'])
        return client, c_name

    def get_client(self, request_json):
        """Get OpenAI client by fetching appropriate port as well
        as fix the request_json to match client model.
        """
        model_alias = request_json['model']
        client, c_name = self.fetch_client(model_alias)

        # res = self.query_manager('models', model_alias=model_alias)
        # if res['url'] is None:
        #     return make_error(res["message"])

        # # Since we get raw response the local ports of docker manager
        # # would be passed as the local ports of this celery machine
        # # which won't be correct if docker manager is remote
        # res['url'] = res['url'].replace("localhost", self.manager_host)

        # client, c_name = fetch_client_by_url(res['url'], res['key'])

        if client is None:
            return make_error("Client may be respawning or remote url is not available. Please repeat your request later.")
        if c_name is not None:
            request_json['model'] = c_name
        return client

    def any_completion(self, request_json, interface_type='chat'):
        client = self.get_client(request_json)
        stream = request_json.get('stream', False)

        if isinstance(client, str):
            if stream:
                self.make_streaming_error(client)
            return client

        if interface_type == 'chat':
            generation_func = client.chat.completions.create
        else:
            generation_func = client.completions.create

        valid_args = dict(signature(generation_func).parameters)
        extra_body = {k: v for k, v in request_json.items() if k not in valid_args}
        request_json = {k: v for k, v in request_json.items() if k in valid_args}
        request_json['extra_body'] = extra_body

        try:
            if not stream:
                res = generation_func(**request_json)
                res = res.json()
            else:
                return self.process_streaming_chunk(generation_func, request_json)
        except Exception as e:
            logger.exception("Error in any_completion")
            res = make_error(str(e))
        return res

    def make_streaming_error(self, error):
        redis = celery_app.backend.client
        stream_key = f"stream:{self.request.id}"
        error = {"message": error}
        chunk_data = json.dumps(error)
        redis.hset(stream_key, "error", str(chunk_data))
        redis.hset(stream_key, "status", "FAILED")

    def process_streaming_chunk(self, generation_func, request_json):
        redis = celery_app.backend.client

        # Ключ для хранения чанков в Redis
        stream_key = f"stream:{self.request.id}"

        # Очищаем предыдущие данные, если они есть
        redis.delete(stream_key)

        # Устанавливаем начальное состояние
        redis.hset(stream_key, "status", "STARTED")

        # Счетчик для чанков
        chunk_index = 0
        try:
            for chunk in generation_func(**request_json):
                chunk_data = json.dumps(chunk.model_dump())

                # Сохраняем чанк в Redis
                redis.hset(stream_key, f"chunk:{chunk_index}", str(chunk_data))
                redis.hset(stream_key, "last_chunk", str(chunk_index))
                redis.hset(stream_key, "status", "PROGRESS")

                # Обновляем TTL ключа (10 минут)
                redis.expire(stream_key, 600)

                chunk_index += 1

            # Устанавливаем статус завершения
            redis.hset(stream_key, "status", "COMPLETED")
            logger.info("Streaming task completed successfully")
            return {"status": "COMPLETED", "chunks": chunk_index}
        except Exception as e:
            chunk_data = extract_openai_error(str(e))
            redis.hset(stream_key, "error", str(chunk_data))
            redis.hset(stream_key, "status", "FAILED")
            return {"status": "FAILED", "chunks": chunk_index}

    def chat_completion(self, request_json):
        return self.any_completion(request_json, interface_type='chat')

    def completion(self, request_json):
        return self.any_completion(request_json, interface_type='')

    def model_list(self, request_json=None):
        models = self.query_manager('library',)["models"]
        models = [{'id': m[0], 'max_model_len': m[1], "status": m[2]} for m in models]
        models = {'data': models}
        return json.dumps(models)


@celery_app.task(base=VllmTask, bind=True, acks_late=True)
def send_vllm_request(self, request_json):
    # logger.info(request_json)
    command = request_json['command']
    request_json = request_json.get('args', None)

    func = getattr(self, command, None)
    if func is None:
        return make_error(f"Unable to process command {command}")
    res = func(request_json)

    return res
