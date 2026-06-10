import os
import pytest
import httpx
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_config()

API_URL = os.environ.get("API_URL", f"http://localhost:{_cfg['celery_config']['app_port']}")
ADMIN_URL = os.environ.get("ADMIN_URL", f"http://localhost:{_cfg['admin_config']['admin_port']}")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", _cfg["admin_config"]["admin_secret"])


@pytest.fixture
def api_url():
    return API_URL


@pytest.fixture
async def admin_client():
    headers = {}
    if ADMIN_SECRET:
        headers["Authorization"] = f"Bearer {ADMIN_SECRET}"
    async with httpx.AsyncClient(
        base_url=ADMIN_URL,
        headers=headers,
        timeout=30,
    ) as c:
        yield c


@pytest.fixture
async def available_models(admin_client):
    r = await admin_client.get("/api/available-models")
    r.raise_for_status()
    return r.json()["models"]


@pytest.fixture
def remote_model():
    return "Qwen3-235B-A22B-Instruct-2507"


@pytest.fixture
def local_model():
    return "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture
async def make_token(admin_client):
    created = []

    async def _factory(**kwargs):
        import uuid
        name = f"test_{uuid.uuid4().hex[:8]}"
        r = await admin_client.post("/api/users", json={"user_name": name, "priority": 5})
        r.raise_for_status()
        api_key = r.json()["key"]

        r2 = await admin_client.get("/api/users")
        r2.raise_for_status()
        user = next(u for u in r2.json() if u["user_key"] == api_key)
        user_id = user["id"]
        created.append(user_id)

        patch = {}
        for field in [
            "is_active", "expires_at", "allowed_models",
            "token_budget", "rate_limit_tokens_per_min",
            "rate_limit_tokens_per_hour", "rate_limit_tokens_per_day",
            "rate_limit_requests_per_min",
        ]:
            if field in kwargs:
                patch[field] = kwargs[field]
        if patch:
            rp = await admin_client.patch(f"/api/users/{user_id}", json=patch)
            rp.raise_for_status()

        return user_id, api_key

    yield _factory

    for uid in created:
        try:
            await admin_client.patch(f"/api/users/{uid}", json={"is_active": 0})
        except Exception:
            pass
