from openai import APIConnectionError
import requests
import json
from fastapi.responses import JSONResponse
from inspect import signature
from celery import Celery, Task
from time import sleep
from openai import OpenAI
from kombu import Exchange, Queue
from src.utils import load_global_config, make_error
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


def fetch_client_by_port(port, api_key=None):
    openai_api_key = api_key
    if openai_api_key is None:
        openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    try:
        model_list = client.models.list()
    except APIConnectionError:
        return None, None
    client_name = model_list.data[0].id
    return client, client_name


class VllmTask(Task):
    def __init__(self):
        self._clients = None
        self._models = []
        self.manager_port = CFG['manager_port']

    def query_manager(self, query_type, **kwargs):
        url = f"http://localhost:{self.manager_port}/{query_type}"
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
        model_name = request_json['model']
        res = self.query_manager('models', model_name=model_name)
        if res['port'] is None:
            return make_error(res["message"])
        client, c_name = fetch_client_by_port(res['port'], res['key'])
        if client is None:
            return make_error("Client may be respawning. Please repeat your request later.")
        request_json['model'] = c_name
        return client

    def any_completion(self, request_json, interface_type='chat'):
        client = self.get_client(request_json)
        if isinstance(client, str):
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
            res = generation_func(**request_json)
            res = res.json()
        except:
            res = make_error("Unexpected server error occured")
        return res

    def chat_completion(self, request_json):
        return self.any_completion(request_json, interface_type='chat')

    def completion(self, request_json):
        return self.any_completion(request_json, interface_type='')

    def model_list(self, request_json=None):
        models = self.query_manager('library',)["models"]
        models = [{'id': m[0], 'max_model_len': m[1]} for m in models]
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
