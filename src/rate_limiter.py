import time
import logging
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

MSK_OFFSET = 3 * 3600  # UTC+3 (Moscow)


class RateLimiter:
    """Redis-based rate limiter for token and request quotas."""

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis = None

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url)
        return self._redis

    async def check(self, user_id: int, rate_limits: dict) -> tuple:
        """Check if the request is within rate limits.

        Args:
            user_id: The user ID
            rate_limits: dict with keys 'tokens_per_min', 'tokens_per_hour',
                         'tokens_per_day', 'requests_per_min'

        Returns:
            (allowed: bool, message: str or None)
        """
        if rate_limits is None:
            return True, None

        now = time.time() + MSK_OFFSET
        minute_bucket = int(now // 60)
        hour_bucket = int(now // 3600)
        day_bucket = int(now // 86400)

        redis = await self._get_redis()

        checks = [
            ('requests_per_min', f"rate:{user_id}:req:min:{minute_bucket}", 120),
            ('tokens_per_min', f"rate:{user_id}:tok:min:{minute_bucket}", 120),
            ('tokens_per_hour', f"rate:{user_id}:tok:hour:{hour_bucket}", 7200),
            ('tokens_per_day', f"rate:{user_id}:tok:day:{day_bucket}", 172800),
        ]

        for limit_key, redis_key, ttl in checks:
            limit = rate_limits.get(limit_key)
            if limit is None:
                continue
            current = await redis.get(redis_key)
            current = int(current) if current else 0
            if current >= limit:
                window = limit_key.split('_')[-1]
                msg = f"Rate limit exceeded: {current}/{limit} ({window})"
                logger.warning("User %d: %s", user_id, msg)
                return False, msg

        return True, None

    async def increment(self, user_id: int, tokens_used: int) -> None:
        """Increment rate limit counters after a successful request."""
        now = time.time() + MSK_OFFSET
        minute_bucket = int(now // 60)
        hour_bucket = int(now // 3600)
        day_bucket = int(now // 86400)

        redis = await self._get_redis()

        keys = [
            (f"rate:{user_id}:req:min:{minute_bucket}", 1, 120),
            (f"rate:{user_id}:tok:min:{minute_bucket}", tokens_used, 120),
            (f"rate:{user_id}:tok:hour:{hour_bucket}", tokens_used, 7200),
            (f"rate:{user_id}:tok:day:{day_bucket}", tokens_used, 172800),
        ]

        for key, increment, ttl in keys:
            pipe = redis.pipeline()
            pipe.incrby(key, increment)
            pipe.expire(key, ttl)
            await pipe.execute()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
