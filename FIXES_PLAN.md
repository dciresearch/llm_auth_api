# Аудит кодовой базы: spawn/unload логика — ИСПРАВЛЕННЫЙ ПЛАН

## Найденные проблемы

### P1 — Double spawn remote-моделей

**Где:** `docker_manager/docker_store.py`, `try_spawn_by_alias()` строки 244–247

**Суть:**
Для remote-конфигов (`config.remote_url is not None`) спавн происходит **вне `spawner_lock`** и без какой-либо проверки на уже идущий спавн:

```python
if config.remote_url is not None:
    spawned = await self.spawn_docker(config)  # нет lock, нет guard
    return spawned
```

При двух параллельных запросах к одной remote-модели оба пройдут через эту ветку, оба вызовут `spawn_docker` → `track_new_instance`. Второй вызов перезапишет запись в `_store` и продублирует `save_container_key` в SQLite.

**Контекст:** для локальных моделей double-spawn **уже защищён** — `spawner_lock` + double-check `if instance_alias in self._store` на строке 255 внутри lock гарантируют, что `spawn_docker` для локальной модели вызовется не более одного раза.

**Последствия:**
- Дублирование записи API-ключа в БД для remote-модели.
- Лишний `check_health()` к внешнему URL при каждом параллельном запросе.

---

### P2 — Коллизия портов при параллельном спавне разных моделей

**Где:** `docker_manager/docker_store.py`, `get_vacant_ports()` (строки 156–160) + `spawn_docker()` (строки 418–419)

**Суть:**
`get_vacant_ports()` вычисляет свободные порты как `self._known_ports - self.get_allocated_ports()` и вызывается внутри `spawn_docker()` (строка 418) — **уже вне `spawner_lock`**. Два параллельных спавна разных моделей могут получить одинаковый порт до того, как первый из них завершится и занесёт порт в `_store` через `track_new_instance`. Второй `docker run` падает с `Bind for 0.0.0.0:XXXX failed: port is already allocated`.

Также в `get_vacant_ports` есть **необработанный `ValueError`**: если свободных портов меньше `n`, `random.sample` кидает исключение, которое не перехватывается.

**Последствия:**
- Один из параллельных спавнов падает с ошибкой порта → пользователь получает 500.
- GPU, зарезервированные через `_inflight_gpus`, освобождаются в `finally`, но коллизия порта при retry повторится.

---

### P3 — `fetch_instance_url`: ложная ошибка при race condition между спавном и чтением `_store`

**Где:** `docker_manager/docker_store.py`, `fetch_instance_url()` строки 288–301

**Суть:**
```python
spawned = await self.try_spawn_by_alias(alias)  # строка 288 — содержит await
...
if alias not in self._store:                    # строка 291
    error = f"{alias} can't be deployed at this time."
...
i = self._store[alias]                          # строка 301 — достижима только если error is None
```

`KeyError` на строке 301 невозможен: строка 291 уже проверяет отсутствие alias в `_store` и при отсутствии выставляет `error`, после чего строка 295 делает `return` до строки 301.

Реальная проблема другая: между строкой 288 (`try_spawn_by_alias`) и строкой 291 (`if alias not in self._store`) event loop может передать управление другой корутине — например, обработчику `/spawned` → `fetch_spawned_models()` → `remove_idle_or_crashed_instances(remove_idle=True)`. Если этот обработчик успеет убить только что поднятый контейнер (по истечении `max_idle_time`, который мог быть маленьким), строка 291 обнаружит отсутствие alias в `_store` и клиент получит ошибку **`"can't be deployed at this time"`** — вместо успешного ответа с URL.

Таким образом, баг проявляется как ложная ошибка «модель недоступна», а не как 500 с необработанным исключением. На практике маловероятно, но архитектурно незащищено.

---

### P4 — `_evict_one_idle_instance` вызывает синхронный Docker API в async-контексте

**Где:** `docker_manager/docker_store.py`, `_evict_one_idle_instance()` (строки 329–343), вызов из `try_spawn_by_alias` (строка 266) внутри `async with spawner_lock`

**Суть:**
`_evict_one_idle_instance()` → `_remove_instance()` → `stop_container()` → `container.kill()` + `container.remove(force=True)` — синхронные блокирующие Docker SDK-вызовы. Вызываются из корутины (`try_spawn_by_alias` — `async def`) без `await` и без `run_in_executor`, поэтому **блокируют event loop напрямую** — не из-за самого `asyncio.Lock()`, а потому что синхронный I/O выполняется в теле корутины, не отдавая управление.

Дополнительно: вызов происходит внутри `async with spawner_lock`, поэтому lock удерживается на всё время синхронного I/O — другие спавны не могут даже начать проверку GPU, пока эвикция не завершится.

**Аналогичная проблема:** `remove_idle_or_crashed_instances(remove_idle=False)` тоже вызывается под lock (строка 258) и для каждого живого контейнера делает синхронный `check_health()` → `requests.get()`. При N контейнерах — N синхронных HTTP-запросов к vLLM под lock. Это отдельная подпроблема (см. P10).

**Последствия:**
- Во время эвикции event loop заморожен: не обрабатываются ни другие спавны, ни endpoint'ы manager.
- В худшем случае (зависший Docker daemon) `kill()`/`remove()` могут висеть десятки секунд.

---

### P5 — Module-level `spawner_lock` и `_inflight_gpus`

**Где:** `docker_manager/docker_store.py`, строки 61–62

**Суть:**
```python
spawner_lock = asyncio.Lock()
_inflight_gpus = set()
```
Объявлены на уровне модуля. Если создать несколько экземпляров `InstanceManager` (например, в тестах или при расширении архитектуры), они будут разделять один lock и один набор inflight GPU, хотя управляют независимыми контейнерами.

---

### P6 — Celery worker thread блокируется на время спавна

**Где:** `celery_tasks.py`, `query_manager()` строка 79, `fetch_client()` строки 99–111

**Суть:**
Воркер делает синхронный `requests.get(..., timeout=660)` к manager. Manager при `/models` выполняет `fetch_instance_url` → `try_spawn_by_alias` — ждёт `docker run` + health-check. Health-check в `spawn_docker` (строки 453–456): exponential backoff с `max_delay=30` сек, `retry_count=35` итераций — максимальное время ожидания **порядка 15–18 минут**. Всё это время HTTP-соединение открыто, Celery thread заблокирован.

`timeout=660` (11 минут) недостаточен — может не покрыть весь сценарий спавна. Либо manager отвечает быстро (модель уже поднята), либо никак не отвечает до таймаута.

**Риск:** при concurrency Celery меньше числа одновременных спавнов — очередь задач перестаёт сдвигаться.

**Дополнительный блокирующий вызов:** после `query_manager` воркер вызывает `fetch_client_by_url` → `client.models.list()` (синхронный OpenAI SDK вызов с httpx timeout=1800 сек). Итого в худшем случае один воркер-thread заблокирован на: `query_manager` (~15 мин, время спавна) + `models.list` (~5 сек, connect timeout) = порядка 15+ минут на один запрос. Это усиливает риск исчерпания concurrency Celery.

**Корень проблемы:** архитектура `/models` endpoint'а синхронная — manager держит соединение открытым на всё время спавна. S6 (retry loop) лишь маскирует симптом, но не решает корень: при `timeout=30` сек к manager, если спавн занимает минуты, manager не успеет ответить за это время и каждый запрос будет падать по timeout. Полное решение требует сделать спавн асинхронным на стороне manager: сразу возвращать `{"status": "spawning"}`, а Celery делать retry. Это архитектурное изменение, S6 лишь улучшает текущую схему в её рамках.

---

### P7 — Manager endpoints блокируют event loop синхронными health-check'ами

**Где:** `manager_server.py`, `/spawned` (строка 49), `/library` (строка 44), `/instances` (строки 60–77)

**Суть:**
- `/spawned` → `fetch_spawned_models()` → `remove_idle_or_crashed_instances()` → для каждого контейнера `check_health()` → `is_vllm_up()` → `requests.get()` — всё синхронно прямо в async handler'е.
- `/library` → `fetch_known_models()` (объявлена как `async def`, но внутри вызывает тот же синхронный `remove_idle_or_crashed_instances()`).
- `/instances` — `remove_idle_or_crashed_instances()` вызывается синхронно (строка 60), `inst.check_health()` тоже синхронный (строка 71), хотя `get_gpu_info` уже корректно вынесен в `run_in_executor`.

N контейнеров = N sync HTTP-запросов в async handler → event loop заблокирован на всё это время.

---

### P8 — `nvidia-smi` блокирует event loop изнутри `spawner_lock`

**Где:** `docker_manager/docker_store.py`, `get_gpu_ids()` → `find_gpu_ids()` → `get_gpu_memory()` → `subprocess.check_output()` (строка 27), вызов из `try_spawn_by_alias` (строка 260) внутри `async with spawner_lock`

**Суть:**
`subprocess.check_output` — синхронный блокирующий вызов. Выполняется внутри `async with spawner_lock`, блокируя event loop на время выполнения `nvidia-smi` (~10–50 мс в норме, дольше при нагрузке).

---

### P9 — `post_to_queue` молча глотает malformed JSON

**Где:** `main_api.py`, `post_to_queue()` строки 50–55

**Суть:**
```python
except json.decoder.JSONDecodeError:
    request_json = {}
request_json = {"command": command, "args": request_json}
```
Malformed JSON тела запроса не возвращает 400. Вместо этого в Celery передаётся `{"command": command, "args": {}}`, что приводит к непонятным downstream-ошибкам в воркере (например, `KeyError: 'model'`).

---

### P10 — `remove_idle_or_crashed_instances` под lock делает синхронные health-check'и

**Где:** `docker_manager/docker_store.py`, `try_spawn_by_alias()` строка 258, вызов `self.remove_idle_or_crashed_instances(remove_idle=False)` внутри `async with spawner_lock`

**Суть:**
`remove_idle_or_crashed_instances()` проходит по всем элементам `_store` и для каждого вызывает `v.check_health()` → синхронный `requests.get()` к vLLM (строки 306–318). При N живых контейнерах = N синхронных HTTP-запросов — всё внутри `spawner_lock`, блокируя event loop. Аналогично P4/P8, но для другой операции.

**Последствия:**
- При каждом вызове `try_spawn_by_alias` event loop замораживается на время N * RTT к vLLM.
- Масштабируется линейно с числом запущенных контейнеров.

**Решение (S2-смежное):** вынести `remove_idle_or_crashed_instances` из-под lock: сначала под lock снять список кандидатов на удаление (только чтение `_store`), затем вне lock проверить их health и удалить найденные мёртвые. Либо кешировать результат `check_health` с небольшим TTL, чтобы не дёргать HTTP при каждом спавне.

---

## Предлагаемые решения

### S1 — Instance-level lock + `_inflight_ports` + исправление remote double-spawn

**Патч:** `docker_manager/docker_store.py`

**Решает:** P1, P2, P5 (частично)

#### 1. Перенос `spawner_lock` и `_inflight_gpus` на уровень экземпляра, добавление `_inflight_ports`

Строки 61–62 удалить. В `__init__` добавить:

```python
self._spawner_lock = asyncio.Lock()
self._inflight_gpus: set = set()
self._inflight_ports: set = set()
```

Все обращения `spawner_lock` → `self._spawner_lock`, `_inflight_gpus` → `self._inflight_gpus`.

#### 2. Резервирование портов до выхода из lock

Вынести `get_vacant_ports()` из `spawn_docker()` в `try_spawn_by_alias()` — выполнять **под lock** вместе с GPU-резервированием, передавать в `spawn_docker()` явным аргументом. Внутри `spawn_docker()` убрать вызов `get_vacant_ports()` на строке 418 — порты всегда передаются снаружи.

**Важно:** в `spawn_docker()` также есть fallback на строках 411–415:
```python
if gpu_ids is None:
    gpu_ids = self.get_gpu_ids(config.gpu_needed)
```
После внедрения S1 этот fallback должен быть удалён (или заменён на `assert gpu_ids is not None`). Иначе при вызове `spawn_docker` напрямую (например, из тестов или будущих расширений) `get_gpu_ids` снова вызовет незащищённый `nvidia-smi` вне lock и без резервирования в `_inflight_gpus`.

Обновить `get_vacant_ports()` для учёта `_inflight_ports` и защиты от `ValueError`:

```python
def get_vacant_ports(self, n=1):
    available = list(
        self._known_ports - self.get_allocated_ports() - self._inflight_ports
    )
    if len(available) < n:
        return None  # вместо ValueError из random.sample
    return random.sample(available, k=n)
```

В `try_spawn_by_alias()` внутри `async with self._spawner_lock`, сразу после выбора `gpu_ids`:

```python
ports = self.get_vacant_ports(config.ports_needed)
if ports is None:
    return (False, "No ports available.")
self._inflight_gpus.update(gpu_ids)
self._inflight_ports.update(ports)
# break — выходим из lock
```

В `finally`-блоке спавна освобождать оба набора:

```python
finally:
    self._inflight_gpus.difference_update(gpu_ids)
    self._inflight_ports.difference_update(ports)
```

#### 3. Guard для remote-моделей

Добавить `_inflight_aliases: Dict[str, asyncio.Event]` в `__init__` — словарь alias → `asyncio.Event`. Когда спавн идёт, в словаре хранится незавершённый Event; когда спавн закончен — Event выставляется и запись удаляется.

```python
self._inflight_aliases: dict = {}  # alias -> asyncio.Event
```

Логика ветки remote:

```python
if config.remote_url is not None:
    while True:
        async with self._spawner_lock:
            if instance_alias in self._store:
                return True
            if instance_alias not in self._inflight_aliases:
                # Мы первые — занять слот и выйти из lock
                event = asyncio.Event()
                self._inflight_aliases[instance_alias] = event
                break
            # Кто-то уже спавнит — взять Event под lock и ждать вне lock
            event = self._inflight_aliases[instance_alias]
        # Вне lock — ждём завершения параллельного спавна
        await event.wait()
        # После пробуждения: либо модель появилась в _store, либо спавн упал
        async with self._spawner_lock:
            return instance_alias in self._store

    # Мы выиграли слот — спавним
    try:
        spawned = await self.spawn_docker(config)
        return spawned
    finally:
        async with self._spawner_lock:
            self._inflight_aliases.pop(instance_alias, None)
        event.set()  # разбудить всех ожидающих
```

**Почему нельзя просто вернуть `True` при `alias in _inflight_aliases`:** спавн ещё не завершился, `_store` ещё не содержит alias. Вызывающий код в `fetch_instance_url` сразу после `try_spawn_by_alias` обратится к `_store` и получит ложную ошибку `"can't be deployed"`. Нужно дождаться реального завершения спавна через `asyncio.Event.wait()`, а затем проверить `_store` — если спавн прошёл успешно, alias будет там; если нет — вернуть `False`.

---

### S2 — Вынос Docker API из-под lock при эвикции

**Патч:** `docker_manager/docker_store.py`, `_evict_one_idle_instance()`

**Решает:** P4

Разделить метод на два шага:
1. **Под lock** — только идентификация жертвы и её `pop` из `_store`.
2. **Вне lock** — `stop_container()` и `delete_container_key()` (Docker API + SQLite).

```python
async def _evict_one_idle_instance(self):
    """Kill the most-idle expired instance. Returns True if evicted."""
    async with self._spawner_lock:
        expired = [
            (k, v) for k, v in self._store.items()
            if not v.is_virtual and v.expired()
        ]
        if not expired:
            return False
        k, v = max(expired, key=lambda kv: kv[1].get_time_idle())
        self._store.pop(k)

    # Outside lock — Docker API may take seconds
    try:
        v.stop_container()
    finally:
        if self._api_db is not None:
            self._api_db.delete_container_key(k)
    return True
```

`delete_container_key` вынесен в `finally`: если `stop_container()` выбросит исключение (зависший Docker daemon и т.п.), запись в БД всё равно будет удалена и не останется «мёртвой» строки.

Метод становится `async`. В `try_spawn_by_alias` while-цикл нужно реструктурировать: сейчас эвикция происходит **внутри** `async with self._spawner_lock`, и при успешной эвикции цикл продолжается через `asyncio.sleep(3)` для ожидания освобождения GPU-памяти. После выноса эвикции эта логика должна сохраниться:

```python
while True:
    async with self._spawner_lock:
        if instance_alias in self._store:
            return True
        self.remove_idle_or_crashed_instances(remove_idle=False)
        gpu_ids = self.get_gpu_ids(config.gpu_needed, exclude=self._inflight_gpus)
        if gpu_ids:
            ports = self.get_vacant_ports(config.ports_needed)
            if ports is None:
                return (False, "No ports available.")
            self._inflight_gpus.update(gpu_ids)
            self._inflight_ports.update(ports)
            break
        # Найти жертву под lock — только pop из _store
        expired = [(k, v) for k, v in self._store.items() if not v.is_virtual and v.expired()]
        if not expired:
            return (False, "Not enough vacant GPU to spawn Container at this time.")
        k, v = max(expired, key=lambda kv: kv[1].get_time_idle())
        self._store.pop(k)

    # Вне lock — Docker API (может занять секунды)
    try:
        await asyncio.to_thread(v.stop_container)
    finally:
        if self._api_db is not None:
            self._api_db.delete_container_key(k)
    # Ждём освобождения GPU-памяти
    await asyncio.sleep(3)
```

Таким образом, Docker API (`stop_container`) выполняется вне lock, но логика цикла (эвикция → ожидание → повторная проверка GPU) сохраняется. Вместо `run_in_executor` предпочтительно `asyncio.to_thread` (Python 3.9+, соответствует Python 3.10 из Dockerfile).

**Важно:** `remove_idle_or_crashed_instances` остаётся вызываться под lock — это тоже содержит синхронные `check_health()`. Эта подпроблема вынесена в P10.

---

### S3 — Defensive check в `fetch_instance_url`

**Патч:** `docker_manager/docker_store.py`, `fetch_instance_url()` строки 299–302

**Решает:** P3

Существующая проверка на строке 291 (`if alias not in self._store`) уже предотвращает `KeyError` на строке 301. Однако проверка и последующее обращение `self._store[alias]` — это два отдельных чтения, между которыми в теории может произойти удаление ключа (хотя в однопоточном event loop без `await` между строками 291 и 301 это невозможно). Для дополнительной надёжности и явной семантики заменить прямой доступ на `.get()`:

```python
i = self._store.get(alias)
if i is None:
    return (make_server_error(f"{alias} was unloaded during request."), None, None)
i.reset_access_timer()
```

Это также даёт более точное сообщение об ошибке: `"was unloaded during request"` вместо общего `"can't be deployed at this time"`.

---

### S4 — `nvidia-smi` через `asyncio.to_thread`

**Патч:** `docker_manager/docker_store.py`, `try_spawn_by_alias()`

**Решает:** P8

`get_gpu_ids()` вызывает `subprocess.check_output` синхронно. Нельзя просто вынести вызов за пределы lock: после `nvidia-smi` нужно атомарно проверить и зарезервировать GPU. Поэтому требуется двухфазный подход:

**Фаза 1 (под lock):** взять snapshot `_inflight_gpus` и выйти из lock.  
**Фаза 2 (вне lock):** выполнить `nvidia-smi` через `asyncio.to_thread`.  
**Фаза 3 (под lock снова):** double-check — убедиться, что alias всё ещё не в `_store` и `_inflight_gpus` не изменился настолько, чтобы результат был невалиден, затем зарезервировать GPU.

```python
while True:
    # Фаза 1: snapshot под lock
    async with self._spawner_lock:
        if instance_alias in self._store:
            return True
        inflight_snapshot = frozenset(self._inflight_gpus)

    # Фаза 2: nvidia-smi вне lock
    gpu_ids = await asyncio.to_thread(
        self.get_gpu_ids, config.gpu_needed, inflight_snapshot
    )

    # Фаза 3: double-check и резервирование под lock
    async with self._spawner_lock:
        if instance_alias in self._store:
            return True
        # Проверяем, что найденные GPU не заняты пока мы ждали
        if gpu_ids and set(gpu_ids) & self._inflight_gpus:
            # Коллизия — начать сначала
            continue
        if gpu_ids:
            ports = self.get_vacant_ports(config.ports_needed)
            if ports is None:
                return (False, "No ports available.")
            self._inflight_gpus.update(gpu_ids)
            self._inflight_ports.update(ports)
            break
        # Нет GPU — попробовать эвикцию (аналогично S2)
        expired = [(k, v) for k, v in self._store.items() if not v.is_virtual and v.expired()]
        if not expired:
            return (False, "Not enough vacant GPU to spawn Container at this time.")
        k, v = max(expired, key=lambda kv: kv[1].get_time_idle())
        self._store.pop(k)

    # Вне lock — Docker API
    try:
        await asyncio.to_thread(v.stop_container)
    finally:
        if self._api_db is not None:
            self._api_db.delete_container_key(k)
    # Ждём освобождения GPU-памяти, затем цикл повторится с nvidia-smi
    await asyncio.sleep(3)
```

**Важно:** использовать `asyncio.to_thread` (Python 3.10+, есть в Dockerfile), а не устаревший `loop.run_in_executor`. Если GPU, найденные в фазе 2, пересекаются с `_inflight_gpus` в фазе 3 (кто-то занял их пока мы ждали) — нужно начать цикл заново, а не возвращать ошибку. Ветка «нет GPU» в фазе 3 — аналог логики эвикции из S2 — pop из `_store` под lock, `stop_container` вне lock с `try/finally`.

---

### S5 — Non-blocking manager endpoints

**Патч:** `manager_server.py`

**Решает:** P7

**Предупреждение о thread-safety:** `InstanceManager._store` — обычный `dict`, который читается и изменяется из event loop. Передача методов `manager` в `run_in_executor`/`asyncio.to_thread` создаёт concurrent-доступ из потока threadpool. Хотя в CPython операции с `dict` атомарны на уровне GIL, это не является гарантией стандарта языка и не защищает от логических гонок (например, `_store` меняется между итерациями). **Вместо `run_in_executor` для методов, обращающихся к `_store`, нужно снимать snapshot данных в event loop, а вне loop передавать только snapshot (неизменяемые данные).**

Для `/spawned`: `fetch_spawned_models()` вызывает `remove_idle_or_crashed_instances()` → `check_health()` → синхронный HTTP. Нужно разделить: мутирующую часть (`remove_idle_or_crashed_instances`) оставить в event loop, а HTTP health-check'и вынести в `asyncio.to_thread`. Либо принять компромисс: `check_health()` вызывается только под стражей (не меняет `_store` при `remove_idle=False`) — тогда его безопасно выполнять в thread.

Для `/library`: `fetch_known_models` объявлена как `async def` — это мешает вызову через `run_in_executor`. Нужно либо:
- Переименовать в `fetch_known_models_sync` и сделать обычным `def` (рекомендуется), вызывая через `asyncio.to_thread` только часть без мутации `_store`;
- Либо оставить `async` и заменить внутренние sync-вызовы на `await asyncio.to_thread(...)`.

Для `/instances`: текущий код уже корректно использует `run_in_executor` для `get_gpu_info`. `remove_idle_or_crashed_instances()` (строка 60) и `inst.check_health()` (строка 71) вызывать в thread нельзя без snapshot — они читают `_store`. **Правильный подход:** вызвать `remove_idle_or_crashed_instances()` в event loop (это быстро без network I/O если `remove_idle=False`), затем снять `snapshot = list(manager._store.items())` и передать его в `asyncio.to_thread` для health-check'ов и сборки отчёта.

**Ограничение S5:** `remove_idle_or_crashed_instances(remove_idle=False)` остаётся вызываться синхронно в event loop. Этот вызов сам по себе содержит `check_health()` → `requests.get()` (P10), поэтому S5 решает P7 **только частично** — блокировка от проверки здоровья при `/instances` устраняется, но аналогичная блокировка при каждом вызове `remove_idle_or_crashed_instances` в event loop (в т.ч. внутри `spawner_lock`) остаётся. Полное решение P7 требует совместной реализации S5 + решения P10.

```python
@router.get("/instances")
async def fetch_instances():
    manager.remove_idle_or_crashed_instances(remove_idle=False)
    snapshot = list(manager._store.items())  # snapshot под event loop

    def build_report(items):
        result = []
        for name, inst in items:
            result.append({
                "name": name,
                "url": inst.api_url,
                "is_virtual": inst.is_virtual,
                "health": inst.check_health(),    # sync HTTP теперь в thread
                "idle_minutes": int(inst.get_time_idle()),
                "max_idle_minutes": inst.max_idle_time,
                "expired": inst.expired(),
                "gpu_ids": inst.gpu_ids,
                "gpu_info": manager.get_gpu_info(inst.gpu_ids),
            })
        return result

    result = await asyncio.to_thread(build_report, snapshot)
    return {"instances": result}
```

---

### S6 — Увеличить timeout и добавить retry для терминальных ошибок

**Патч:** `celery_tasks.py`, `query_manager()` строка 79, `fetch_client()` строки 99–111

**Решает:** P6 (частично — в рамках текущей синхронной архитектуры)

**Ограничение:** timeout=30 сек к manager не работает если спавн занимает минуты — manager не успеет ответить. Корень проблемы (синхронный блокирующий спавн на стороне manager) требует отдельного архитектурного изменения (асинхронный `/models` с `{"status": "spawning"}`). S6 — улучшение в рамках текущей схемы.

Конкретные правки в рамках текущей схемы:

1. **Увеличить `timeout`** в `query_manager` с 660 до значения, покрывающего максимальное время спавна (~20 минут = 1200 сек):

```python
res = requests.get(url, params=kwargs, headers=headers, timeout=kwargs.pop('timeout', 1200)).json()
```

2. **Добавить ранний выход** в `fetch_client` при терминальных ошибках (сейчас при `url=None` сразу возвращается `None, None` без retry — это уже верно для некоторых случаев, но только если manager вернул ответ):

```python
def fetch_client(self, model_alias):
    res = self.query_manager('models', model_alias=model_alias)
    if res.get('url') is None:
        msg = res.get("message", "").lower()
        logger.warning("Model %s unavailable: %s", model_alias, msg)
        return None, None
    res['url'] = res['url'].replace("localhost", self.manager_host)
    client, c_name = fetch_client_by_url(res['url'], res['key'])
    return client, c_name
```

**Полное решение (архитектурное, опционально):** сделать `/models` неблокирующим — manager сразу возвращает `{"status": "spawning", "url": null}`, Celery делает retry с задержкой. Это устраняет корень проблемы — thread не блокируется, concurrency Celery не расходуется впустую.

---

### S7 — `post_to_queue` malformed JSON → 400

**Патч:** `main_api.py`, строки 50–55

**Решает:** P9

```python
try:
    request_json = await raw_request.json()
    stream = request_json.get('stream', False)
except json.decoder.JSONDecodeError:
    return JSONResponse(
        content={"error": {"message": "Invalid JSON in request body"}},
        status_code=400,
    )
```

---

## Приоритеты реализации

| Приоритет | Проблема | Решение | Оценка усилий | Зависимости |
|---|---|---|---|---|
| **P0** | P1 remote double-spawn | S1 (`_inflight_aliases` + `asyncio.Event` под lock) | Низкая | — |
| **P0** | P2 port collision + ValueError | S1 (`_inflight_ports` + `get_vacant_ports` fix) | Низкая | — |
| **P0** | P5 module-level lock | S1 (instance-level) | Минимальная | — |
| **P0** | P4 eviction: sync Docker API в корутине | S2 (async eviction вне lock, `asyncio.to_thread`) | Низкая | S1 |
| **P1** | P3 race condition → ложная ошибка в `fetch_instance_url` | S3 (defensive `.get()`) | Минимальная | — |
| **P1** | P9 malformed JSON → 400 | S7 | Минимальная | — |
| **P2** | P10 `remove_idle_or_crashed_instances` под lock | S2-смежное (вынос health-check из-под lock или TTL-кеш) | Средняя | S1 |
| **P2** | P8 nvidia-smi блокирует event loop | S4 (двухфазный: snapshot → `asyncio.to_thread` → re-lock) | Средняя | S1 |
| **P2** | P7 manager endpoints блокируют loop | S5 + решение P10 (полное решение требует обоих) | Средняя | — |
| **P2** | P6 Celery thread висит | S6 (увеличить timeout, ранний выход + опц. async спавн) | Средняя | — |

**Порядок реализации: S1 → S2 → S3 → S7 → S4 → S5 → S6.**

S1 и S2 — атомарная группа изменений в `docker_store.py`, реализуются вместе. S3 и S7 — независимые однострочные правки, можно в любой момент. S4 зависит от S1 (нужны `_inflight_gpus` на уровне экземпляра). S5 зависит от решения P10 для полного эффекта. S6 — независим.

---

## Проверки после реализации (попросить пользователя это провести, так как тесты происходят на другой машине.)

1. Два параллельных запроса `/models?model_alias=<remote>` → в `_store` одна запись, в SQLite один ключ.
2. Пять параллельных спавнов разных локальных моделей → порты уникальны, ни одного `port already allocated`.
3. Спавн при нехватке GPU: эвикция idle-контейнера происходит **вне lock** — `/library` и `/instances` отвечают во время эвикции.
4. `/spawned` при 5+ контейнерах → endpoint возвращается быстро, event loop не блокируется.
5. Celery-запрос на спавн: thread не висит > 30 сек, между retry `sleep ≥ 5 сек`.
6. Malformed JSON в `/v1/chat/completions` → ответ 400 без задержки.
7. Race condition в `fetch_instance_url` при быстрой эвикции → клиент получает понятное сообщение об ошибке, не 500.
8. 10 параллельных запросов `/models?model_alias=<remote>` к remote-модели во время спавна → ровно один `spawn_docker` вызов; остальные 9 ждут завершения через `asyncio.Event` и получают корректный результат (URL, не ошибку).
