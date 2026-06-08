# AGENTS.md

## What This Is

An authenticated API gateway in front of dynamically-spawned vLLM Docker containers. Requests go through FastAPI -> Celery (RabbitMQ broker) -> vLLM. The manager server auto-spawns/kills GPU containers based on demand and idle timeout. SQLite stores auth keys and request logs.

## Architecture

```
run_api.py  (orchestrator, spawns 4 subprocesses)
  ├─ uvicorn main_api:app     (port from config.yaml: app_port, default 1234)
  ├─ uvicorn manager_server:app (port: manager_port, default 6333)
  ├─ celery worker             (celery_tasks, thread pool, concurrency=10000)
  └─ sqlite_web                (port: dbapi_port, default 2235)
```

- `main_api.py` — public-facing FastAPI app with auth middleware, proxies requests to Celery queue
- `manager_server.py` — internal FastAPI app that manages vLLM Docker container lifecycle
- `celery_tasks.py` — Celery task that calls vLLM via OpenAI SDK, handles streaming via Redis hashes
- `src/api_database.py` — SQLAlchemy models (`UserAuth`, `Requests`) + `Database` class
- `src/utils.py` — config loader (`config.yaml`), helpers, `with_cancellation` decorator
- `docker_manager/` — `InstanceManager` (docker_store.py), spawn logic, config parsing
- `llm_docker_configs/` — JSON model configs; rename to `.hidden` to disable a model

## Running Locally

No `requirements.txt` exists. Install deps from `docker_manager/Dockerfile`:

```bash
pip install fastapi docker uvicorn fire sqlalchemy pyyaml celery openai redis sqlite-web httpx
```

Start infrastructure (RabbitMQ, Redis, SQLite):

```bash
docker compose up -d
```

Start the full stack:

```bash
python run_api.py
```

Or run inside the manager Docker container (requires nvidia-docker):

```bash
bash run_main.sh
```

## Key Config

All runtime config is in `config.yaml` (loaded by `src/utils.py:load_global_config`). The loader uses a relative path `"config.yaml"` — the CWD must be the repo root.

Model configs live in `llm_docker_configs/`. Each JSON needs `config_type`, `model_alias`, `model_name`, `model_parent_dir`, `gpu_needed`. See `remote.jsonexample` for remote model proxying.

## Database

SQLite at `./database/generic.db` (created automatically). CLI:

```bash
python database_cli.py <method> [args]
```

e.g. `python database_cli.py register_new_user myuser 5`

## Conventions

- No test suite, no linter, no type checker, no formatter configured
- Python 3.10 (from Dockerfile)
- Config typo: `config.yaml` has `rabitmq_port` (one 'b')
- Auth token extracted from `Authorization: Bearer <token>` header; checked via `UserAuth` table
- Streaming responses use Redis hashes with `stream:{task_id}` keys (not SSE from the worker directly)
- Celery uses thread pool (`--pool=threads`), not prefork
- `__pycache__` and `database/` contents are gitignored
- `.hidden` suffix on config JSONs disables that model without deleting the file
