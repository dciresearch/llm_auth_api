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

CFG = load_global_config()['celery_config']

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
celery_app.control.rate_limit('celery_tasks.send_vllm_request', '100/s')

logger = logging.getLogger(__name__)


def fetch_client_by_url(url, api_key=None):
    openai_api_key = api_key
    if openai_api_key is None:
        openai_api_key = "EMPTY"
    openai_api_base = f"{url}/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    try:
        model_list = client.models.list()
    except (APIConnectionError, OpenAIError):
        return None, None
    client_name = model_list.data[0].id
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
        print(url)
        res = requests.get(url, params=kwargs).json()
        return res

    def check_health(self):
        status = []
        for name, client, _ in self.clients.items():
            try:
                client.models.list()
                status.append((name, "online"))
            except APIConnectionError:
                status.append((name, "offline"))
        return json.dumps(status)

    @property
    def clients(self):
        return self._clients

    def get_client(self, request_json):
        """Get OpenAI client by fetching appropriate port as well
        as fix the request_json to match client model.
        """
        model_alias = request_json['model']
        res = self.query_manager('models', model_alias=model_alias)
        if res['url'] is None:
            return make_error(res["message"])

        # Since we get raw response the local ports of docker manager
        # would be passed as the local ports of this celery machine
        # which won't be correct if docker manager is remote
        res['url'] = res['url'].replace("localhost", self.manager_host)

        client, c_name = fetch_client_by_url(res['url'], res['key'])
        if client is None:
            return make_error("Client may be respawning or remote url is not available. Please repeat your request later.")
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
            print(e)
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
            print("Streaming task completed successfully")
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
