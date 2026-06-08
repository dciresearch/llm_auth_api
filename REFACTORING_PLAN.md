# Refactoring Plan: llm_auth_api

Based on deep audit conducted 2026-06-04, cross-verified against source code. Covers Critical, High, and Medium findings plus new feature requirements.

---

## Part 1: Critical Bugs (must fix first)

### C1. Manager API — unauthenticated spawn endpoint
**File:** `manager_server.py:32-45`
**Problem:** All endpoints lack auth. `/models?model_alias=X` triggers actual container spawn via `try_spawn_by_alias()` → `spawn_docker()`. Anyone with network access to port 6333 can spawn/kill GPU containers.
**Fix:** Add token-based auth middleware to `manager_server.py`. Use a shared secret between `main_api` and `manager_server` (config value in `config.yaml`).

### C2. SQLite Web exposed without auth
**File:** `run_api.py:18`
**Problem:** `sqlite_web` runs on `0.0.0.0:2235` without password. Direct DB access from network.
**Fix:** Either bind to `127.0.0.1` only, or add `--password-file` flag, or remove from production.

### C3. `flushdb()` kills all streaming data on import
**File:** `celery_tasks.py:34`
**Problem:** `celery_app.backend.client.flushdb()` runs at module import time. Starting a new worker wipes all in-progress streaming data from other workers.
**Fix:** Remove `flushdb()`. If cleanup is needed, do it once via a dedicated management command, not at import.

### C4. `lru_cache` on `check_user_key` never invalidates
**File:** `src/api_database.py:45-53`
**Problem:** `@lru_cache(1000)` with no TTL. Deleted/revoked users retain access until process restart.
**Fix:** Replace with TTL-based cache (e.g. 5-minute TTL). See utils.py `ttl_classcache` — adapt or use `cachetools.TTLCache`.

### C5. Duplicate FastAPI app — `build_app()` never used
**File:** `main_api.py:182-235`
**Problem:** `build_app()` configures CORS, middleware, docs, error handlers. But the actual app created at line 235 (`app = FastAPI()`) has none of that. CORS, `root_path`, validation handler — all lost.
**Fix:** Either use `build_app()` properly or configure the module-level `app` directly. Remove dead code.

### C6. Non-existent attribute in validation handler
**File:** `main_api.py:202`
**Problem:** `app.state.openai_serving_chat` is never defined. Will raise `AttributeError` on any `RequestValidationError`.
**Fix:** Replace with proper error response.

### C7. `wait_for_task` ignores FAILURE/REVOKED states
**File:** `main_api.py:39-46`
**Problem:** Only checks for `SUCCESS`. Tasks in `FAILURE`/`REVOKED` spin for full `TIME_TO_EXPIRE` (600s) before returning generic error.
**Fix:** Add explicit handling for `FAILURE`, `REVOKED`, and timeout states.

### C8. `AttributeError` when streaming fails
**File:** `main_api.py:99`
**Problem:** `redis.hget(stream_key, "error")` may return `None`. Calling `.decode('utf-8')` on `None` crashes.
**Fix:** Add null check before `.decode()`.

---

## Part 2: High — Reliability & Architecture

### H1. Blocking Redis calls in async context
**File:** `main_api.py:72,88,108,114`
**Problem:** Synchronous `redis.hget()`/`redis.hset()` inside async generator blocks the FastAPI event loop.
**Fix:** Use `redis.asyncio` client for the streaming generator.

### H2. Endpoints returning None
**File:** `main_api.py:146,151,161,178`
**Problem:** `/tokenize`, `/detokenize`, `/version`, `/v1/embeddings` return `None`. OpenAI SDK clients expect proper responses.
**Fix:** Either implement or return explicit 501 Not Implemented.

### H3. Redis memory leak in streaming
**File:** `celery_tasks.py:187-192`
**Problem:** Chunks accumulate in Redis with 600s TTL but are never explicitly cleaned after completion. Under load = gigabytes.
**Fix:** Delete `stream:{task_id}` hash after consumer finishes processing (in `main_api.py`'s `generate()`).

### H4. `stop_container` infinite loop
**File:** `docker_manager/instances.py:82-86`
**Problem:** `while True` with `container.kill()`. If kill() doesn't raise but container doesn't die — infinite loop.
**Fix:** Add retry limit (e.g. 3 attempts) and fallback to `container.remove(force=True)`.

### H5. Port selection not future-proof
**File:** `docker_manager/docker_store.py:90-94`
**Problem:** `random.sample` from available ports. Currently mitigated because `get_vacant_ports()` is called inside `spawn_docker()` (line 226) which runs under `spawner_lock`. However, if `get_vacant_ports` is ever called outside the lock, two concurrent spawns could get the same port.
**Fix:** Reserve port atomically inside the spawner lock explicitly, or document the lock dependency.

### H6. Global lock serializes all spawns
**File:** `docker_manager/docker_store.py:55`
**Problem:** Single `asyncio.Lock()` for all spawn operations. Spawning an 8-GPU model blocks spawning a 1-GPU model.
**Fix:** Per-GPU-group or per-model locks. Or use a semaphore with reasonable concurrency.

### H7. Container health check uses fixed sleep intervals
**File:** `docker_manager/docker_store.py:265-267`
**Problem:** Fixed `sleep(startup_time)` (20s) between health check attempts. The loop does exit early on success (the `while` condition checks `not instance.check_api_health()`), but fixed 20s intervals are wasteful — a container that's ready in 2s still waits 20s before being detected.
**Fix:** Use exponential backoff (e.g. start at 1s, double each attempt) with health endpoint polling.

### H8. `purge_instance` may leave orphan containers
**File:** `docker_manager/docker_store.py:210-212`
**Problem:** `pop` + `del` relies on `__del__` which isn't guaranteed to call `stop_container()`.
**Fix:** Call `stop_container()` explicitly before removing from store.

### H9. Blocking HTTP in Celery worker
**File:** `celery_tasks.py:75`
**Problem:** `requests.get()` is blocking. With thread pool concurrency=10000, this means 10000 blocking threads.
**Fix:** Use `httpx` (already imported) or `requests` with connection pooling. Consider async Celery worker if throughput demands it.

### H10. `VllmInstance.check_api_health` always True for remote
**File:** `docker_manager/instances.py:101-104`
**Problem:** Virtual (remote) instances always return `True` from health check, even if remote URL is down.
**Fix:** Actually ping the remote health endpoint for virtual instances.

---

## Part 3: New Features

### F1. Per-token model access control
**Requirement:** Restrict which models each token can view and interact with.
**Schema changes in `src/api_database.py`:**
```
UserAuth table — add:
  - allowed_models: Optional[str]  (JSON array of model aliases, NULL = all models)
```
**Behavior:**
- `NULL` or empty → token can access all models (backward compatible with existing tokens)
- JSON array `["qwen3-4b-instruct", "tpro"]` → token can only see/use these models
- Enforce in `main_api.py` authentication middleware: check `model` field from request body against `allowed_models`
- Enforce in `main_api.py` `/v1/models` endpoint: filter returned models by `allowed_models`
- `database_cli.py`: add `set_user_models <user_id> <model1,model2,...>` command

### F2. Token expiration (TTL)
**Requirement:** Each token has a lifetime (default 6 months). Editable. Expired tokens rejected.
**Schema changes in `src/api_database.py`:**
```
UserAuth table — add:
  - created_at: int  (unix timestamp, auto-set on creation)
  - expires_at: int  (unix timestamp, default = created_at + 6 months)
```
**Behavior:**
- `check_user_key` must verify `expires_at > now` — reject expired tokens with clear message
- Invalidate `lru_cache` for expired tokens (related to C4 — TTL cache fixes this)
- `database_cli.py`: add `extend_token <user_id> <new_expiry_iso_or_duration>` command
- `database_cli.py`: add `set_expiry <user_id> <duration>` command (e.g. "6m", "1y", "2026-12-31")
- Return `expires_at` in any admin-facing user info

### F3. Bulk token revocation
**Requirement:** Ability to disable ALL existing tokens at once.
**Implementation:**
- `database_cli.py`: add `revoke_all_tokens` command
- Approach: add `is_active: bool` column (default `True`) to `UserAuth`
- `revoke_all_tokens` sets `is_active = False` for all rows
- `check_user_key` must check `is_active == True`
- Alternative (simpler): rotate a global salt used in key generation — but this is destructive and non-recoverable
- **Chosen approach:** `is_active` column — reversible, auditable, clear
- `database_cli.py`: add `reactivate_user <user_id>` for selective re-enabling
- `database_cli.py`: add `revoke_user <user_id>` for single token revocation

### Combined schema migration for F1+F2+F3
```python
class UserAuth(Base):
    __tablename__ = "user_auth_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_key: Mapped[str] = mapped_column(String, index=True)
    user_name: Mapped[str]
    priority: Mapped[int]
    allowed_models: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array or NULL
    created_at: Mapped[int] = mapped_column(Integer, default=get_current_ts)
    expires_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # NULL = never expires
    is_active: Mapped[bool] = mapped_column(Integer, default=1)                 # SQLite bool
```

### F4. Token billing (usage tracking)
**Requirement:** Track how many tokens each user consumes. Store per-request usage (prompt_tokens, completion_tokens, total_tokens). Provide summary stats per user.

**Schema changes in `src/api_database.py`:**
```
Requests table — add:
  - prompt_tokens: Mapped[Optional[int]]       # tokens in the prompt
  - completion_tokens: Mapped[Optional[int]]    # tokens in the completion
  - total_tokens: Mapped[Optional[int]]         # prompt + completion

UserAuth table — add:
  - total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)  # running total
  - token_budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # NULL = unlimited
```

**Behavior:**
- Extract `usage` from OpenAI response (`response.usage.prompt_tokens`, `.completion_tokens`, `.total_tokens`) in `LoggingIterator.__anext__` (for streaming) and `authentication` middleware (for non-streaming)
- Save token counts alongside request in `save_response()`
- Increment `UserAuth.total_tokens_used` on each request
- Check `token_budget` before processing request: if `total_tokens_used >= token_budget`, return 429 with clear message "Token budget exhausted"
- `token_budget = NULL` → unlimited (backward compatible)
- `database_cli.py`: add `set_token_budget <user_id> <budget>` command
- `database_cli.py`: add `reset_token_usage <user_id>` command
- `database_cli.py`: add `get_usage <user_id>` command — returns total_tokens_used, breakdown by model, request count
- `database_cli.py`: add `get_usage_all` command — summary table for all users

**Enforcement point:** In `main_api.py` `authentication` middleware — check budget before `call_next(request)`. Reject with 429 if exceeded.

### F5. Token generation limits (rate limiting)
**Requirement:** Limit how many tokens a user can generate per time window (e.g. per minute, per hour, per day). Prevents abuse and controls costs.

**Schema changes in `src/api_database.py`:**
```
UserAuth table — add:
  - rate_limit_tokens_per_min: Mapped[Optional[int]]  = mapped_column(Integer, nullable=True)
  - rate_limit_tokens_per_hour: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
  - rate_limit_tokens_per_day: Mapped[Optional[int]]  = mapped_column(Integer, nullable=True)
  - rate_limit_requests_per_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
```
`NULL` = no limit for that window (backward compatible).

**Implementation approach:** Use Redis counters (already available via Celery backend) for real-time rate limiting. SQLite is too slow for per-request counter updates under load.
```
Redis keys:
  - rate:{user_id}:tokens:min:{minute_bucket}   → int (token count, TTL 120s)
  - rate:{user_id}:tokens:hour:{hour_bucket}     → int (token count, TTL 7200s)
  - rate:{user_id}:tokens:day:{day_bucket}       → int (token count, TTL 172800s)
  - rate:{user_id}:requests:min:{minute_bucket}  → int (request count, TTL 120s)
```

**Behavior:**
- After each request completes, increment Redis counters by `total_tokens` used
- Before processing a request (in auth middleware), check all applicable limits:
  - If any limit would be exceeded → return 429 with `Retry-After` header and clear message
  - Message format: `{"error": {"message": "Rate limit exceeded: X tokens/min (limit: Y)", "retry_after": Z}}`
- For streaming requests: estimate tokens as they arrive; if limit mid-stream, terminate with error chunk
- `database_cli.py`: add `set_rate_limit <user_id> <field> <value>` command (e.g. `set_rate_limit 5 rate_limit_tokens_per_hour 100000`)
- `database_cli.py`: add `clear_rate_limits <user_id>` command — sets all limits to NULL
- Expose current usage in `/v1/models` or a new admin endpoint (optional)

**Enforcement point:** In `main_api.py` `authentication` middleware — check limits before `call_next(request)`. After response, increment counters. Use Redis `INCRBY` + `EXPIRE` for atomicity.

### Combined schema migration for F4+F5
```python
class UserAuth(Base):
    # ... existing columns from F1+F2+F3 ...
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    token_budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_tokens_per_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_tokens_per_hour: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_tokens_per_day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_requests_per_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

class Requests(Base):
    # ... existing columns ...
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
```

---

### F6. Admin Panel (web UI)
**Requirement:** A separate web-based admin service for managing all tokens visually. View all tokens, their fields, edit inline, create new tokens, revoke/activate.

**Implementation:**
- New file: `admin_panel.py` — standalone FastAPI app on separate port (default 6334, configurable via `admin_config.admin_port` in `config.yaml`)
- Protected by `admin_config.admin_secret` password (Bearer token or cookie)
- **UI:** Server-side rendered HTML with dark theme, embedded in `admin_panel.py` (no separate template files)
- **Features:**
  - Dashboard stats: total tokens, active/inactive count, total requests
  - Full token table with all fields: ID, Name, Key, Priority, Active, Created, Expires, Allowed Models, Tokens Used, Budget, Rate Limits (per min/hour/day, requests/min)
  - Inline editing of all fields via input fields
  - Save button per row (PATCH `/api/users/{id}`)
  - Create new token row at bottom (POST `/api/users`)
  - Revoke/Activate toggle per token
  - Reset usage counter per token
- **API endpoints:**
  - `GET /` — HTML admin panel
  - `GET /api/users` — JSON list of all users
  - `POST /api/users` — create new user (body: `{user_name, priority, key?}`)
  - `PATCH /api/users/{id}` — update any field
  - `POST /api/users/{id}/reset-usage` — reset token usage counter
- **Dependencies:** `jinja2` for HTML templating
- **Config:** `admin_config.admin_port` (default 6334), `admin_config.admin_secret`
- **Process management:** Added to `run_api.py` as `uvicorn admin_panel:app`

---

## Part 4: Refactoring & Cleanup (tied to fixes above)

### R1. Fix or replace `build_app()` in main_api.py
Remove dead code. Configure the single module-level `app` with CORS, middleware, docs settings directly.

### R2. Remove dead file `src/openai_protocol.py`
966 lines, unused, references non-existent imports. Delete entirely.

### R3. Replace print() with logging
Configure `logging` module. Replace ~30 `print()` calls across all files.

### R4. Add requirements.txt
Pin all dependencies from `docker_manager/Dockerfile` plus `httpx`, `cachetools`.

### R5. Fix `config.yaml` loading path
`src/utils.py:16` — use `Path(__file__).parent.parent / "config.yaml"` instead of relative `"config.yaml"`.

### R6. Fix `AutoConfig` classmethod signatures
`docker_manager/config_patterns.py:78,83` — change `self` to `cls`.

### R7. Fix mutable defaults in dataclasses
`docker_manager/config_patterns.py:26,54` — `lambda: []` and `lambda: {}` are not valid dataclass defaults. Use `dataclasses.field(default_factory=...)`.

### R8. Remove unused imports
`main_api.py`: `importlib`, `inspect`, `tempfile`, `argparse.Namespace`
Note: `signature` from `inspect` in `celery_tasks.py` is USED (line 144) — do NOT remove.

### R9. Remove `--reload` from production commands
`run_api.py:15,17` — remove `--reload` flag or make it configurable via config.

---

## Part 5: Medium — Missed by Initial Audit (found during cross-verification)

### M1. `check_health()` guaranteed crash — `clients` is always `None`
**File:** `celery_tasks.py:78-90`
**Problem:** `self._clients` is initialized to `None` in `__init__` (line 62) and never populated. `check_health()` at line 80 calls `self.clients.items()` which raises `AttributeError` on `None`. This method is dead code but reachable via `send_vllm_request` with `command='check_health'`.
**Fix:** Either implement client tracking or remove `check_health()` and return an appropriate response.

### M2. Bare `except:` catches `KeyboardInterrupt`
**File:** `src/utils.py:108`
**Problem:** `get_port_from_url` uses bare `except:` which catches `KeyboardInterrupt`, `SystemExit`, etc.
**Fix:** Change to `except Exception:` or `except (ValueError, IndexError):`.

### M3. Bare `except:` in `is_vllm_up`
**File:** `docker_manager/instances.py:96`
**Problem:** Same issue — bare `except:` catches everything.
**Fix:** Change to `except Exception:`.

### M4. Wrong type hint on `url` property
**File:** `docker_manager/instances.py:60`
**Problem:** `-> Tuple[str, int]` but returns `self.api_url` which is `str`.
**Fix:** Change to `-> str`.

### M5. Wrong type hint on `key` property
**File:** `docker_manager/instances.py:64`
**Problem:** `-> int` but returns `self.api_key` which is `str`.
**Fix:** Change to `-> str`.

### M6. `LoggingIterator` silently swallows parse errors
**File:** `main_api.py:289-292`
**Problem:** `except Exception as e: print(e); pass` — all streaming chunk parse errors are silently lost. No logging, no metrics, no way to debug data corruption.
**Fix:** Use `logging.warning()` with exc_info. Consider whether certain errors should set `had_error = True`.

### M7. `extract_request_details` no error handling for malformed JSON
**File:** `src/utils.py:72`
**Problem:** `json.loads(raw_body)` raises on non-JSON bodies with no try/except. Any non-JSON request (e.g. form data) crashes the middleware.
**Fix:** Wrap in try/except, return `body=None` on parse failure.

### M8. `GenericDockerConfig.__init__` does nothing
**File:** `docker_manager/config_patterns.py:31-32`
**Problem:** `__init__(self, **kwargs): pass` — all kwargs are silently ignored. Base class fields are never set. Works only because `VllmConfig` overrides `__init__` and sets fields explicitly.
**Fix:** Set fields from kwargs in the base class, or remove the `__init__` override and let the dataclass default `__init__` work (requires removing `init=False`).

### M9. `AutoConfig.__init__` error message truncated
**File:** `docker_manager/config_patterns.py:75`
**Problem:** `"is designed to be instantiated "` — missing the word "not".
**Fix:** `"is not designed to be instantiated directly"`.

### M10. `remove_possible_orphans` kills ALL matching containers
**File:** `docker_manager/docker_store.py:79-85`
**Problem:** On startup, kills any container whose name starts with `dockermanaged_vllm`. If multiple manager instances run (e.g. during deploy), they kill each other's containers.
**Fix:** Include a unique instance ID in container names, or check container labels before killing.

### M11. `generate()` always yields `[DONE]` after errors
**File:** `main_api.py:131`
**Problem:** `yield "data: [DONE]\n\n"` is outside the while loop but inside the function. It executes after error chunks too, which may confuse clients that expect `[DONE]` only after successful completion.
**Fix:** Only yield `[DONE]` if no error occurred (track error state in the generator).

### M12. `post_to_queue` silently wraps JSONDecodeError
**File:** `main_api.py:56-58`
**Problem:** On `JSONDecodeError`, `request_json` is set to `None`, then immediately overwritten at line 58 with `{"command": command, "args": None}`. The error is swallowed; the task receives `args=None` and will likely fail downstream with a confusing error.
**Fix:** Return a 400 error immediately on malformed JSON instead of forwarding `None` to the task.

### M13. `spawn_logic` mutates config object as side effect
**File:** `docker_manager/spawn_logic.py:37`
**Problem:** `config.gpu_needed = 0` for remote configs. This mutates the config object in-place. If the same config is reused (e.g. after a remote URL changes to local), `gpu_needed` stays 0.
**Fix:** Don't mutate the config; handle `remote_url is not None` separately in the caller.

### M14. Typo in constant name
**File:** `docker_manager/docker_store.py:48`
**Problem:** `SERVER_ERORR_PATTERN` — missing second 'R' in "ERROR".
**Fix:** Rename to `SERVER_ERROR_PATTERN`.

---

## Execution Order

### Phase 1 — Schema & DB (F1, F2, F3)

#### Step 1. Add new columns to UserAuth (allowed_models, created_at, expires_at, is_active)
**What done:** Extended `UserAuth` model in `src/api_database.py` with four new columns: `allowed_models` (Text, nullable), `created_at` (Integer, default=now), `expires_at` (Integer, nullable), `is_active` (Integer, default=1). Added `SIX_MONTHS_SECONDS` constant. Added `_run_migrations()` method to `Database` that uses `ALTER TABLE` to add missing columns to existing databases automatically on startup.

#### Step 2. Migration: set created_at=now, expires_at=now+6m for existing rows, is_active=1
**What done:** Migration is handled by `_run_migrations()` in `Database.__init__()`. Uses `ALTER TABLE ADD COLUMN` with `DEFAULT` values — `created_at` defaults to current timestamp, `expires_at` defaults to now+6 months, `is_active` defaults to 1. This ensures backward compatibility with existing databases without data loss.

#### Step 3. Update check_user_key to validate is_active + expires_at + return allowed_models
**What done:** `check_user_key` now returns a 4-tuple `(exists, user_id, priority, allowed_models)`. Added validation: returns `(False, ...)` if `is_active == 0` or if `expires_at < now`. All callers updated (`main_api.py` auth middleware, `post_to_queue`, `register_new_user`).

#### Step 4. Update database_cli.py with new commands
**What done:** Added new methods to `Database` class (auto-exposed via `fire.Fire`): `set_user_models(user_id, models)`, `set_expiry(user_id, duration)`, `extend_token(user_id, duration)`, `revoke_user(user_id)`, `reactivate_user(user_id)`, `revoke_all_tokens()`. Added `_parse_duration()` helper supporting formats: `6m`, `1y`, `30d`, `2026-12-31`.

---

### Phase 2 — Auth & Security (C1, C2, C4, C5, C6)

#### Step 5. Add manager_server auth middleware
**What done:** Added `manager_secret` to `config.yaml` under `manager_config`. Added `@app.middleware("http")` to `manager_server.py` that checks `Authorization: Bearer <secret>` header. Empty secret disables auth (backward compatible). Updated `celery_tasks.py` `query_manager()` to pass the auth header in all manager requests.

#### Step 6. Fix main_api.py app configuration (remove build_app duplication)
**What done:** Removed dead `build_app()` function and all its unused imports (`importlib`, `inspect`, `tempfile`, `argparse.Namespace`). Configured module-level `app` directly with CORS middleware (allow_origins=["*"]) and proper `RequestValidationError` handler that returns `{"error": {"message": ...}}` instead of referencing non-existent `app.state.openai_serving_chat`.

#### Step 7. Replace lru_cache with TTL cache on check_user_key
**What done:** Replaced `@lru_cache(1000)` with `cachetools.TTLCache(maxsize=1000, ttl=300)` (5-minute TTL). Falls back to `lru_cache` if `cachetools` is not installed. Cache is checked manually in `check_user_key` to support the new validation logic.

#### Step 8. Fix validation exception handler
**What done:** Replaced handler that referenced `app.state.openai_serving_chat` (always `None`) with a simple handler that returns `{"error": {"message": str(exc)}}` with status 400. Done as part of Step 6.

#### Step 9. Bind sqlite_web to 127.0.0.1 or add auth
**What done:** Changed `run_api.py` `dbapi_command` from `--host 0.0.0.0` to `--host 127.0.0.1`, restricting SQLite Web access to localhost only.

---

### Phase 3 — Reliability (C3, C7, C8, H1-H10)

#### Step 10. Remove flushdb() from celery_tasks.py
**What done:** Removed `celery_app.backend.client.flushdb()` line from `celery_tasks.py:34`. Starting a new worker no longer wipes in-progress streaming data from other workers.

#### Step 11. Fix wait_for_task error handling
**What done:** Added explicit checks for `FAILURE` and `REVOKED` task states in `wait_for_task()`. These now return proper error messages instead of spinning for the full 600s timeout.

#### Step 12. Fix streaming error handling (null checks) + yield [DONE] only on success (M11)
**What done:** In `generate()`, `redis.hget(stream_key, "error")` now checks for `None` before calling `.decode()`. Added `had_error` flag — `[DONE]` is only yielded if no error occurred. Error messages are logged via `logger.warning`.

#### Step 13. Fix post_to_queue JSON error handling — return 400 on malformed JSON (M12)
**What done:** On `JSONDecodeError`, `post_to_queue` now returns `JSONResponse(content={"error": {"message": "Invalid JSON in request body"}}, status_code=400)` instead of silently setting `request_json = None` and forwarding to the task.

#### Step 14. Switch to redis.asyncio for streaming
**What done:** Replaced synchronous `celery_app.backend.client` with `redis.asyncio.from_url()` in the `generate()` async generator. All `redis.hget()` calls replaced with `await redis.hget()`. Added proper cleanup with `await redis.delete(stream_key)` and `await redis.aclose()` in a `finally` block.

#### Step 15. Implement missing endpoints or return 501
**What done:** `/tokenize`, `/detokenize`, `/version`, `/v1/embeddings` now return `JSONResponse(content={"error": "Not Implemented"}, status_code=501)` instead of `None`.

#### Step 16. Fix stop_container infinite loop (add retry limit)
**What done:** Replaced `while True` loop with a `for _ in range(3)` loop. Falls back to `container.remove(force=True)` after 3 failed kill attempts. Handles all exceptions gracefully.

#### Step 17. Fix container spawn health check (exponential backoff)
**What done:** Replaced fixed `sleep(startup_time)` (20s) with exponential backoff: starts at 1s, doubles each attempt, capped at `startup_time`. Container ready in 2s is now detected in ~3s instead of ~20s.

#### Step 18. Fix purge_instance — call stop_container() explicitly before pop/del
**What done:** Changed `purge_instance` to call `v.stop_container()` explicitly before removing from store. Uses `pop(k, None)` to handle missing keys safely. Removed reliance on `__del__`.

#### Step 19. Fix remote health check for virtual instances
**What done:** `VllmInstance.check_api_health()` for virtual instances now calls `is_vllm_up(self.url)` instead of returning `True` unconditionally. Remote URLs are actually pinged.

#### Step 20. Clean up Redis stream data after consumption
**What done:** In `generate()`, added `finally` block that calls `await redis.delete(stream_key)` and `await redis.aclose()`. Stream data is cleaned up immediately after consumption instead of waiting for 600s TTL.

#### Step 21. Fix check_health() crash — clients is always None (M1)
**What done:** Replaced broken `check_health()` that iterated over `self.clients` (always `None`) with a working implementation that queries the manager's `/library` endpoint and returns model statuses.

---

### Phase 4 — Features integration

#### Step 22. Enforce allowed_models in auth middleware + /v1/models filtering
**What done:** In auth middleware: if `allowed_models` is set (not None), parses JSON array and checks `model` field from request body against it. Returns 403 if model not allowed. `/v1/models` endpoint filters the response data to only include models in the user's `allowed_models` list.

#### Step 23. Enforce token expiration in auth middleware
**What done:** Handled in `check_user_key` (Step 3). Returns `(False, ...)` if `expires_at < now`. TTL cache (Step 7) ensures expired tokens are re-checked within 5 minutes.

#### Step 24. Enforce is_active check in auth middleware
**What done:** Handled in `check_user_key` (Step 3). Returns `(False, ...)` if `is_active == 0`. TTL cache (Step 7) ensures revoked tokens are re-checked within 5 minutes.

#### Step 25. Wire up all new database_cli commands
**What done:** All new methods (`set_user_models`, `set_expiry`, `extend_token`, `revoke_user`, `reactivate_user`, `revoke_all_tokens`) are automatically exposed via `fire.Fire(api_db)` in `database_cli.py`. No additional wiring needed.

---

### Phase 5 — Cleanup (R1-R9, M1-M14)

#### Step 26. Remove dead code (openai_protocol.py, unused imports, build_app)
**What done:** Deleted `src/openai_protocol.py` (966 lines, unused). Removed unused imports from `main_api.py`: `importlib`, `inspect`, `tempfile`, `argparse.Namespace`, `AsyncIterator`, `Set`. Removed `_running_tasks` variable (defined but never used).

#### Step 27. Replace print with logging
**What done:** Added `logging.getLogger(__name__)` to all files. Replaced `print()` calls with appropriate `logger.info()`, `logger.debug()`, `logger.warning()`, `logger.error()`, `logger.exception()` calls across: `celery_tasks.py`, `docker_manager/docker_store.py`, `docker_manager/config_patterns.py`, `src/api_database.py`, `main_api.py`. CLI methods (`database_cli.py`) retain `print()` for stdout output.

#### Step 28. Add requirements.txt
**What done:** Created `requirements.txt` with all dependencies: `fastapi`, `uvicorn`, `docker`, `pyyaml`, `celery`, `openai`, `redis`, `httpx`, `sqlalchemy`, `fire`, `cachetools`.

#### Step 29. Fix config path, dataclass defaults, classmethod signatures
**What done:**
- **R5:** `src/utils.py` — changed `open("config.yaml")` to `open(Path(__file__).parent.parent / "config.yaml")` for reliable path resolution regardless of CWD.
- **R7:** `docker_manager/config_patterns.py` — changed `lambda: []` to `dataclasses.field(default_factory=list)` and `lambda: {}` to `dataclasses.field(default_factory=dict)`.
- **R6:** Fixed `AutoConfig.from_config(self, config)` → `from_config(cls, config)` and `AutoConfig.from_path(self, path)` → `from_path(cls, path)`.

#### Step 30. Make --reload configurable
**What done:** Added `debug` key to `config.yaml`. In `run_api.py`, `--reload` flag is only added when `debug: true` is set. Default is production mode (no reload).

#### Step 31. Fix bare except clauses (M2, M3)
**What done:**
- **M2:** `src/utils.py` `get_port_from_url` — changed `except:` to `except (ValueError, IndexError, AttributeError):`.
- **M3:** `docker_manager/instances.py` `is_vllm_up` — changed `except:` to `except Exception:`.

#### Step 32. Fix type hints on url/key properties (M4, M5)
**What done:**
- **M4:** `DockerInstance.url` — changed return type from `Tuple[str, int]` to `str`.
- **M5:** `DockerInstance.key` — changed return type from `int` to `str`.

#### Step 33. Improve LoggingIterator error handling (M6)
**What done:** Replaced `print('what:', chunk_str); print(e); pass` with `logger.warning("Failed to parse streaming chunk: %s | chunk: %s", e, chunk_str, exc_info=True)`. Similarly for the save response error handler.

#### Step 34. Fix extract_request_details JSON error handling (M7)
**What done:** Wrapped `json.loads(raw_body)` in try/except `(json.JSONDecodeError, ValueError)`, returning `body=None` on parse failure instead of crashing.

#### Step 35. Fix GenericDockerConfig.__init__ and AutoConfig error message (M8, M9)
**What done:**
- **M8:** `GenericDockerConfig.__init__` now iterates over `dataclasses.fields(self)` and sets attributes from kwargs, matching `VllmConfig`'s behavior.
- **M9:** Fixed error message from `"is designed to be instantiated "` to `"is not designed to be instantiated directly"`.

#### Step 36. Fix remove_possible_orphans to be multi-instance safe (M10)
**What done:** Added `instance_id` (UUID-based) to `InstanceManager`. Containers are now created with a `labels={"manager_instance_id": self.instance_id}` label. `remove_possible_orphans` only kills containers matching this instance's ID, preventing conflicts between multiple manager instances.

#### Step 37. Fix spawn_logic config mutation side effect (M13)
**What done:** Removed `config.gpu_needed = 0` from `get_vllm_docker_spawn_args`. Updated `try_spawn_by_alias` to skip GPU checks when `config.remote_url is not None` (remote configs don't need local GPUs).

#### Step 38. Fix SERVER_ERORR_PATTERN typo (M14)
**What done:** Renamed `SERVER_ERORR_PATTERN` to `SERVER_ERROR_PATTERN` in `docker_manager/docker_store.py`.

---

### Phase 6 — Billing & Rate Limiting (F4, F5)

#### Step 39. Add token usage columns to Requests table (F4)
**What done:** Added `prompt_tokens`, `completion_tokens`, `total_tokens` (Integer, nullable) columns to `Requests` model in `src/api_database.py`. Added migrations in `_run_migrations()` that add these columns to existing `requests` tables via `ALTER TABLE`.

#### Step 40. Add billing columns to UserAuth table (F4)
**What done:** Added `total_tokens_used` (Integer, default=0) and `token_budget` (Integer, nullable=NULL) columns to `UserAuth` model. Added migrations in `_run_migrations()`. NULL budget = unlimited (backward compatible).

#### Step 41. Extract usage from responses and save to DB (F4)
**What done:** Updated `save_response()` to accept `prompt_tokens`, `completion_tokens`, `total_tokens` kwargs and store them in `Requests`. Added `increment_token_usage(user_id, tokens)` method that does `UPDATE user_auth_keys SET total_tokens_used = total_tokens_used + ?`. In `main_api.py` auth middleware: extracts `usage` from non-streaming OpenAI responses and passes to `save_response()`. In `LoggingIterator`: extracts `usage` from streaming chunks (when present in final chunk) and passes to `save_response()` on `StopAsyncIteration`. Both paths call `increment_token_usage()` after saving.

#### Step 42. Enforce token budget before request processing (F4)
**What done:** In `main_api.py` `authentication` middleware, after auth check, verifies `total_tokens_used < token_budget` (or `token_budget IS NULL`). If exceeded, returns 429 with `{"error": {"message": "Token budget exhausted", "used": ..., "budget": ...}}`. `check_user_key` now returns a 7-tuple including `token_budget` and `total_tokens_used`.

#### Step 43. Add billing CLI commands (F4)
**What done:** Added to `Database` class: `set_token_budget(user_id, budget)` — sets budget, 0/unlimited; `reset_token_usage(user_id)` — resets counter to 0; `get_usage(user_id)` — prints detailed stats including breakdown by model; `get_usage_all()` — prints summary table for all users. All auto-exposed via `fire.Fire`.

#### Step 44. Add rate limit columns to UserAuth table (F5)
**What done:** Added `rate_limit_tokens_per_min`, `rate_limit_tokens_per_hour`, `rate_limit_tokens_per_day`, `rate_limit_requests_per_min` (Integer, nullable=NULL) to `UserAuth` model. Added migrations in `_run_migrations()`. NULL = no limit (backward compatible). `check_user_key` now returns rate limits as a dict in the 7-tuple.

#### Step 45. Implement Redis-based rate limiter (F5)
**What done:** Created `src/rate_limiter.py` with `RateLimiter` class. Uses `redis.asyncio` client. Methods: `check(user_id, rate_limits) → (allowed, message)` — reads current counters from Redis and compares against limits; `increment(user_id, tokens_used)` — atomically increments counters using Redis pipeline (`INCRBY` + `EXPIRE`). Redis keys: `rate:{user_id}:{metric}:{bucket}` with TTLs (120s for min, 7200s for hour, 172800s for day).

#### Step 46. Enforce rate limits in auth middleware (F5)
**What done:** In `main_api.py` `authentication` middleware, after auth and budget checks, calls `rate_limiter.check(user_id, rate_limits)`. If not allowed, returns 429 with `Retry-After: 60` header and error message. After successful request, calls `rate_limiter.increment(user_id, total_tokens)` to update counters. Rate limiter initialized with Redis URL from config.

#### Step 47. Add rate limit CLI commands (F5)
**What done:** Added to `Database` class: `set_rate_limit(user_id, field, value)` — sets a specific rate limit field (validates field name); `clear_rate_limits(user_id)` — sets all four rate limit fields to NULL. All auto-exposed via `fire.Fire`.

---

### Phase 7 — Admin Panel (F6)

#### Step 48. Create admin panel service (F6)
**What done:** Created `admin_panel.py` — standalone FastAPI app with server-side rendered HTML admin UI. Features: dashboard stats (total/active/inactive tokens, total requests); full token table with all fields editable inline; create new token form; revoke/activate toggle; reset usage button; dark theme. Protected by `admin_config.admin_secret` (Bearer token or cookie). API endpoints: `GET /` (HTML), `GET /api/users`, `POST /api/users`, `PATCH /api/users/{id}`, `POST /api/users/{id}/reset-usage`. Added `admin_config` section to `config.yaml` with `admin_port` (default 6334) and `admin_secret`. Added admin process to `run_api.py`. Added `jinja2` to `requirements.txt`.
