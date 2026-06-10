# Test Plan: LLM Auth API Limits & Restrictions

## Restrictions Tested

| # | Restriction | DB Field | HTTP Code | Error Pattern |
|---|---|---|---|---|
| 1 | Invalid token | `user_key` | 401 | `"Unauthorized"` |
| 2 | Missing auth header | — | 401 | `"Unauthorized"` |
| 3 | Inactive token | `is_active=0` | 401 | `"Unauthorized"` |
| 4 | Expired token | `expires_at` | 401 | `"Unauthorized"` |
| 5 | Allowed models filters /v1/models | `allowed_models` | 200 | only listed models in response |
| 6 | Disallowed model blocked | `allowed_models` | 403 | `"not allowed for this token"` |
| 7 | Token budget exhausted | `token_budget` | 429 | `"Token budget exhausted"` |
| 8 | Rate limit: requests/min | `rate_limit_requests_per_min` | 429 | `"Rate limit exceeded ... (min)"` |
| 9 | Rate limit: tokens/min | `rate_limit_tokens_per_min` | 429 | `"Rate limit exceeded ... (min)"` |
| 10 | Rate limit: tokens/hour | `rate_limit_tokens_per_hour` | 429 | `"Rate limit exceeded ... (hour)"` |
| 11 | Rate limit: tokens/day | `rate_limit_tokens_per_day` | 429 | `"Rate limit exceeded ... (day)"` |
| 12 | Cache invalidation immediate | — | 403→200 | change takes effect without delay |

## Configuration

Reads from `config.yaml` (project root). Overridable via env vars:

| Env var | Default |
|---|---|
| `API_URL` | `http://localhost:1234` |
| `ADMIN_URL` | `http://localhost:6334` |
| `ADMIN_SECRET` | value from `config.yaml → admin_config.admin_secret` |

## Running

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Models Used

- **Remote (always available):** `Qwen3-235B-A22B-Instruct-2507`
- **Local (needs spawning):** `Qwen/Qwen3-4B-Instruct-2507`
