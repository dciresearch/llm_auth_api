import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader
from src.api_database import Database, UserAuth
from src.utils import load_global_config
import httpx

logger = logging.getLogger(__name__)

CFG = load_global_config()
ADMIN_SECRET = CFG.get('admin_config', {}).get('admin_secret', '')
ADMIN_PORT = CFG.get('admin_config', {}).get('admin_port', 6334)
APP_PORT = CFG.get('celery_config', {}).get('app_port', 1234)
db_path = "./database/generic.db"
api_db = Database(db_path)

app = FastAPI(title="LLM Auth API — Admin Panel")

TEMPLATES_DIR = Path(__file__).parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(request: Request):
    if not ADMIN_SECRET:
        return True
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    cookie_token = request.cookies.get("admin_token", "")
    if token == ADMIN_SECRET or cookie_token == ADMIN_SECRET:
        return True
    raise HTTPException(status_code=401, detail="Unauthorized")


def _ts_to_str(ts):
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _str_to_ts(s):
    if not s or s.strip() == "":
        return None
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        return int(s) if s.isdigit() else None


def _user_to_dict(u):
    return {
        "id": u.id,
        "user_name": u.user_name,
        "user_key": u.user_key,
        "priority": u.priority,
        "allowed_models": u.allowed_models,
        "created_at": u.created_at,
        "created_at_str": _ts_to_str(u.created_at),
        "expires_at": u.expires_at,
        "expires_at_input": datetime.fromtimestamp(u.expires_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M") if u.expires_at else "",
        "is_active": u.is_active,
        "total_tokens_used": u.total_tokens_used or 0,
        "token_budget": u.token_budget,
        "rate_limit_tokens_per_min": u.rate_limit_tokens_per_min,
        "rate_limit_tokens_per_hour": u.rate_limit_tokens_per_hour,
        "rate_limit_tokens_per_day": u.rate_limit_tokens_per_day,
        "rate_limit_requests_per_min": u.rate_limit_requests_per_min,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, _=Depends(_check_auth)):
    from sqlalchemy import func as sa_func
    from src.api_database import Requests

    with api_db.Session() as session:
        users = session.query(UserAuth).order_by(UserAuth.id).all()
        user_dicts = [_user_to_dict(u) for u in users]
        total_users = len(user_dicts)
        active_users = sum(1 for u in user_dicts if u["is_active"])
        total_requests = session.query(sa_func.count(Requests.id)).scalar() or 0

    tmpl = jinja_env.get_template("admin.html")
    return tmpl.render(
        users=user_dicts,
        total_users=total_users,
        active_users=active_users,
        inactive_users=total_users - active_users,
        total_requests=total_requests,
    )


@app.get("/api/users")
async def api_list_users(_=Depends(_check_auth)):
    with api_db.Session() as session:
        users = session.query(UserAuth).order_by(UserAuth.id).all()
        return [_user_to_dict(u) for u in users]


@app.post("/api/users")
async def api_create_user(request: Request, _=Depends(_check_auth)):
    data = await request.json()
    name = data.get("user_name")
    priority = data.get("priority", 5)
    key = data.get("key")
    if not name:
        raise HTTPException(400, "user_name is required")
    user_name, priority, new_key = api_db.register_new_user(name, priority, key)
    return {"user_name": user_name, "priority": priority, "key": new_key}


@app.patch("/api/users/{user_id}")
async def api_update_user(user_id: int, request: Request, _=Depends(_check_auth)):
    data = await request.json()
    with api_db.Session() as session:
        user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
        if not user:
            raise HTTPException(404, "User not found")

        if "user_name" in data:
            user.user_name = data["user_name"]
        if "priority" in data:
            user.priority = data["priority"]
        if "is_active" in data:
            user.is_active = data["is_active"]
        if "allowed_models" in data:
            v = data["allowed_models"]
            user.allowed_models = None if v in (None, "", "all", "null") else v
        if "expires_at" in data:
            v = data["expires_at"]
            if v is None or v == "":
                user.expires_at = None
            elif isinstance(v, str):
                user.expires_at = _str_to_ts(v)
            elif isinstance(v, (int, float)):
                user.expires_at = int(v)
        if "token_budget" in data:
            v = data["token_budget"]
            user.token_budget = None if v in (None, "", 0) else int(v)
        if "rate_limit_tokens_per_min" in data:
            v = data["rate_limit_tokens_per_min"]
            user.rate_limit_tokens_per_min = None if v in (None, "", 0) else int(v)
        if "rate_limit_tokens_per_hour" in data:
            v = data["rate_limit_tokens_per_hour"]
            user.rate_limit_tokens_per_hour = None if v in (None, "", 0) else int(v)
        if "rate_limit_tokens_per_day" in data:
            v = data["rate_limit_tokens_per_day"]
            user.rate_limit_tokens_per_day = None if v in (None, "", 0) else int(v)
        if "rate_limit_requests_per_min" in data:
            v = data["rate_limit_requests_per_min"]
            user.rate_limit_requests_per_min = None if v in (None, "", 0) else int(v)

        session.commit()
        return _user_to_dict(user)


@app.post("/api/users/{user_id}/reset-usage")
async def api_reset_usage(user_id: int, _=Depends(_check_auth)):
    api_db.reset_token_usage(user_id)
    return {"status": "ok"}


@app.get("/api/playground/models")
async def playground_models(token: str, _=Depends(_check_auth)):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(
                f"http://localhost:{APP_PORT}/v1/models",
                headers={"Authorization": f"Bearer {token}"}
            )
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except httpx.ConnectError:
            return JSONResponse(content={"error": {"message": "Cannot connect to API server"}}, status_code=502)


@app.post("/api/playground/chat")
async def playground_chat(request: Request, _=Depends(_check_auth)):
    data = await request.json()
    token = data.pop("token", None)
    if not token:
        raise HTTPException(400, "token is required")

    model = data.get("model")
    messages = data.get("messages", [])
    if not model or not messages:
        raise HTTPException(400, "model and messages are required")

    stream = data.get("stream", False)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": data.get("temperature", 0.7),
        "max_tokens": data.get("max_tokens", 512),
        "top_p": data.get("top_p", 1.0),
        "stream": stream,
    }

    if "chat_template_kwargs" in data:
        payload["chat_template_kwargs"] = data["chat_template_kwargs"]

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if stream:
        async def stream_proxy():
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
                try:
                    async with client.stream(
                        "POST",
                        f"http://localhost:{APP_PORT}/v1/chat/completions",
                        json=payload,
                        headers=headers,
                    ) as r:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                except httpx.ConnectError:
                    yield b'data: {"error": {"message": "Cannot connect to API server"}}\n\n'

        return StreamingResponse(stream_proxy(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10)) as client:
            try:
                r = await client.post(
                    f"http://localhost:{APP_PORT}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                return JSONResponse(content=r.json(), status_code=r.status_code)
            except httpx.ConnectError:
                return JSONResponse(content={"error": {"message": "Cannot connect to API server"}}, status_code=502)
