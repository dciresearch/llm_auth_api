from docker.models.containers import Container
from typing import Union, Dict, Any, List, Tuple
import time
from docker.errors import NotFound, APIError
import requests
import re
import logging
from src.utils import get_port_from_url

logger = logging.getLogger(__name__)


class DockerInstance:
    def __init__(
            self, name_id: str, url: str = None,
            api_key: str = None, container: Container = None, max_idle_time: int = 0,
            gpu_ids: list = None
    ):
        self.name_id = name_id
        self.api_url = url
        self.api_key = api_key
        self.container = container
        self._max_idle_time = max_idle_time
        self._gpu_ids = gpu_ids or []

        self._time_created = time.time()
        self._last_accessed = time.time()

    def reset_access_timer(self):
        self._last_accessed = time.time()

    def check_api_health(self):
        return True

    @property
    def port(self):
        return get_port_from_url(self.api_url)

    @property
    def is_virtual(self):
        return self.container is None

    def container_exists(self):
        if self.is_virtual:
            return True
        try:
            self.container.reload()
            return True
        except NotFound:
            return False

    def check_container_health(self):
        if self.is_virtual:
            return True
        if self.container_exists():
            return self.container.status == 'running'
        return False

    def check_health(self):
        # print(self.container.status)
        if self.check_container_health():
            return self.check_api_health()
        return False

    @property
    def url(self) -> str:
        return self.api_url

    @property
    def key(self) -> str:
        return self.api_key

    @property
    def max_idle_time(self):
        return self._max_idle_time

    @property
    def gpu_ids(self):
        return self._gpu_ids

    def get_time_idle(self):
        return (time.time()-self._last_accessed) // 60

    def expired(self):
        if self.is_virtual:
            return False
        return self.get_time_idle() > self.max_idle_time

    def stop_container(self):
        if self.container is None:
            return
        name = self.name_id
        logger.info("[%s] stopping container", name)
        # Kill the process (fast path). An exited container raises APIError on
        # kill — that is fine, we still need to remove it to free the name.
        try:
            self.container.kill()
        except (NotFound, APIError) as e:
            logger.debug("[%s] kill skipped (container already stopped/gone): %s", name, e)
        except Exception as e:
            logger.warning("[%s] kill failed: %s", name, e)
        # Always attempt remove so the container name is freed.
        try:
            self.container.remove(force=True)
            logger.info("[%s] container removed", name)
        except (NotFound, APIError):
            pass
        except Exception as e:
            logger.error("[%s] remove failed: %s", name, e)


def is_vllm_up(url=None):
    try:
        r = requests.get(url=f'{url}/health', timeout=10)
        return r.status_code == 200
    except Exception:
        return False


class VllmInstance(DockerInstance):
    def check_api_health(self):
        return is_vllm_up(self.url)


instance_types = {
    "vllm": VllmInstance
}
