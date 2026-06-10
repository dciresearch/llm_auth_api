import httpx
import pytest


async def _chat(api_url, api_key, model, prompt="hi"):
    async with httpx.AsyncClient(
        base_url=api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    ) as c:
        return await c.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )


async def _models(api_url, api_key):
    async with httpx.AsyncClient(
        base_url=api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    ) as c:
        return await c.get("/v1/models")


# ── Auth ──────────────────────────────────────────────────────────────



async def test_01_invalid_token(api_url):
    r = await _models(api_url, "invalid_fake_key_12345")
    assert r.status_code == 401



async def test_02_missing_auth_header(api_url):
    async with httpx.AsyncClient(base_url=api_url, timeout=30) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 401



async def test_03_inactive_token(api_url, make_token):
    _, api_key = await make_token(is_active=0)
    r = await _models(api_url, api_key)
    assert r.status_code == 401



async def test_04_expired_token(api_url, make_token):
    _, api_key = await make_token(expires_at=1577836800)  # 2020-01-01
    r = await _models(api_url, api_key)
    assert r.status_code == 401


# ── Allowed models ────────────────────────────────────────────────────



async def test_05_allowed_models_filters_list(
    api_url, make_token, remote_model, local_model
):
    _, api_key = await make_token(
        allowed_models=f'["{remote_model}"]'
    )
    r = await _models(api_url, api_key)
    assert r.status_code == 200
    model_ids = [m["id"] for m in r.json()["data"]]
    assert remote_model in model_ids
    assert local_model not in model_ids



async def test_06_disallowed_model_returns_403(
    api_url, make_token, remote_model, local_model
):
    _, api_key = await make_token(
        allowed_models=f'["{remote_model}"]'
    )
    r = await _chat(api_url, api_key, local_model)
    assert r.status_code == 403
    assert "not allowed" in r.json()["error"]["message"]


# ── Token budget ──────────────────────────────────────────────────────



async def test_07_token_budget_exhausted(
    api_url, make_token, remote_model
):
    _, api_key = await make_token(token_budget=5)

    got_429 = False
    for _ in range(10):
        r = await _chat(api_url, api_key, remote_model)
        if r.status_code == 429:
            assert "budget exhausted" in r.json()["error"]["message"].lower()
            got_429 = True
            break
    assert got_429, "Budget was never exhausted after 10 requests"


# ── Rate limits ───────────────────────────────────────────────────────



async def test_08_rate_limit_requests_per_min(
    api_url, make_token, remote_model
):
    _, api_key = await make_token(rate_limit_requests_per_min=5)

    got_429 = False
    for _ in range(10):
        r = await _chat(api_url, api_key, remote_model)
        if r.status_code == 429:
            assert "Rate limit exceeded" in r.json()["error"]["message"]
            got_429 = True
            break
    assert got_429, "Rate limit never hit after 10 sequential requests"



async def test_09_rate_limit_tokens_per_min(
    api_url, make_token, remote_model
):
    _, api_key = await make_token(rate_limit_tokens_per_min=50)

    r1 = await _chat(api_url, api_key, remote_model, prompt="explain quantum physics in detail")
    assert r1.status_code == 200

    r2 = await _chat(api_url, api_key, remote_model)
    assert r2.status_code == 429
    assert "Rate limit exceeded" in r2.json()["error"]["message"]



async def test_10_rate_limit_tokens_per_hour(
    api_url, make_token, remote_model
):
    _, api_key = await make_token(rate_limit_tokens_per_hour=10)

    r1 = await _chat(api_url, api_key, remote_model)
    assert r1.status_code == 200

    r2 = await _chat(api_url, api_key, remote_model)
    assert r2.status_code == 429
    assert "Rate limit exceeded" in r2.json()["error"]["message"]



async def test_11_rate_limit_tokens_per_day(
    api_url, make_token, remote_model
):
    _, api_key = await make_token(rate_limit_tokens_per_day=10)

    r1 = await _chat(api_url, api_key, remote_model)
    assert r1.status_code == 200

    r2 = await _chat(api_url, api_key, remote_model)
    assert r2.status_code == 429
    assert "Rate limit exceeded" in r2.json()["error"]["message"]


# ── Cache invalidation ────────────────────────────────────────────────



async def test_12_cache_invalidation_immediate(
    api_url, admin_client, make_token, remote_model, local_model
):
    user_id, api_key = await make_token(allowed_models=None)

    r1 = await _chat(api_url, api_key, remote_model)
    assert r1.status_code == 200

    await admin_client.patch(
        f"/api/users/{user_id}",
        json={"allowed_models": f'["{remote_model}"]'},
    )

    r2 = await _chat(api_url, api_key, local_model)
    assert r2.status_code == 403, "Cache invalidation failed: local model still allowed"

    r3 = await _chat(api_url, api_key, remote_model)
    assert r3.status_code == 200, "Remote model should still be allowed"
