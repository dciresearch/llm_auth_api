
import json
import logging
from functools import lru_cache
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, Integer, String, Text, MetaData, func, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped
from typing import Optional, List, Any, Dict, Union
from .utils import get_key_hash, shuffle_string

try:
    from cachetools import TTLCache
    _check_key_cache = TTLCache(maxsize=1000, ttl=300)
except ImportError:
    from functools import lru_cache as _lru_fallback
    _check_key_cache = None

logger = logging.getLogger(__name__)

SIX_MONTHS_SECONDS = 6 * 30 * 24 * 3600

metadata = MetaData()


class Base(DeclarativeBase):
    pass


class Requests(Base):
    __tablename__ = "requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]]
    request: Mapped[str] = mapped_column(Text, nullable=True)
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[Optional[int]]
    model: Mapped[Optional[str]]
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class UserAuth(Base):
    __tablename__ = "user_auth_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_key: Mapped[str] = mapped_column(String, index=True)
    user_name: Mapped[str]
    priority: Mapped[int]
    allowed_models: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(datetime.now(timezone.utc).timestamp()))
    expires_at: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_active: Mapped[int] = mapped_column(Integer, default=1)
    total_tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    token_budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_tokens_per_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_tokens_per_hour: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_tokens_per_day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rate_limit_requests_per_min: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class ContainerKey(Base):
    __tablename__ = "container_keys"
    model_alias: Mapped[str] = mapped_column(String, primary_key=True)
    api_key: Mapped[str] = mapped_column(String)
    created_at: Mapped[int] = mapped_column(Integer, default=lambda: int(datetime.now(timezone.utc).timestamp()))


class Database:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}?charset=utf8")
        self._run_migrations()
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _run_migrations(self):
        """Add missing columns to existing tables for backward compatibility."""
        from sqlalchemy import inspect as sa_inspect, text
        inspector = sa_inspect(self.engine)

        migrations = []

        # UserAuth migrations
        existing_columns = {
            col['name'] for col in inspector.get_columns('user_auth_keys')
        } if 'user_auth_keys' in inspector.get_table_names() else set()

        now_ts = int(datetime.now(timezone.utc).timestamp())
        six_months = 6 * 30 * 24 * 3600

        userauth_migrations = {
            'allowed_models': "ALTER TABLE user_auth_keys ADD COLUMN allowed_models TEXT",
            'created_at': f"ALTER TABLE user_auth_keys ADD COLUMN created_at INTEGER DEFAULT {now_ts}",
            'expires_at': f"ALTER TABLE user_auth_keys ADD COLUMN expires_at INTEGER DEFAULT {now_ts + six_months}",
            'is_active': "ALTER TABLE user_auth_keys ADD COLUMN is_active INTEGER DEFAULT 1",
            'total_tokens_used': "ALTER TABLE user_auth_keys ADD COLUMN total_tokens_used INTEGER DEFAULT 0",
            'total_requests': "ALTER TABLE user_auth_keys ADD COLUMN total_requests INTEGER DEFAULT 0",
            'token_budget': "ALTER TABLE user_auth_keys ADD COLUMN token_budget INTEGER",
            'rate_limit_tokens_per_min': "ALTER TABLE user_auth_keys ADD COLUMN rate_limit_tokens_per_min INTEGER",
            'rate_limit_tokens_per_hour': "ALTER TABLE user_auth_keys ADD COLUMN rate_limit_tokens_per_hour INTEGER",
            'rate_limit_tokens_per_day': "ALTER TABLE user_auth_keys ADD COLUMN rate_limit_tokens_per_day INTEGER",
            'rate_limit_requests_per_min': "ALTER TABLE user_auth_keys ADD COLUMN rate_limit_requests_per_min INTEGER",
        }
        for col, sql in userauth_migrations.items():
            if col not in existing_columns and existing_columns:
                migrations.append(sql)

        # Requests migrations
        req_columns = {
            col['name'] for col in inspector.get_columns('requests')
        } if 'requests' in inspector.get_table_names() else set()

        req_migrations = {
            'prompt_tokens': "ALTER TABLE requests ADD COLUMN prompt_tokens INTEGER",
            'completion_tokens': "ALTER TABLE requests ADD COLUMN completion_tokens INTEGER",
            'total_tokens': "ALTER TABLE requests ADD COLUMN total_tokens INTEGER",
        }
        for col, sql in req_migrations.items():
            if col not in req_columns and req_columns:
                migrations.append(sql)

        if migrations:
            with self.engine.connect() as conn:
                for m in migrations:
                    conn.execute(text(m))
                conn.commit()
            logger.info("Applied %d schema migration(s)", len(migrations))

    @staticmethod
    def get_current_ts() -> int:
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

    def check_user_key(self, auth_key: str) -> List[Any]:
        cached = _check_key_cache.get(auth_key) if _check_key_cache is not None else None
        if cached is not None:
            return cached
        with self.Session() as session:
            keys = session.query(UserAuth).filter(UserAuth.user_key == auth_key).all()
            if not keys:
                result = (False, None, None, None, None, None, None)
            else:
                user = keys[0]
                if not user.is_active:
                    result = (False, user.id, None, None, None, None, None)
                elif user.expires_at is not None and user.expires_at < self.get_current_ts():
                    result = (False, user.id, None, None, None, None, None)
                else:
                    rate_limits = {
                        'tokens_per_min': user.rate_limit_tokens_per_min,
                        'tokens_per_hour': user.rate_limit_tokens_per_hour,
                        'tokens_per_day': user.rate_limit_tokens_per_day,
                        'requests_per_min': user.rate_limit_requests_per_min,
                    }
                    result = (
                        True, user.id, user.priority, user.allowed_models,
                        user.token_budget, user.total_tokens_used, rate_limits,
                    )
        if _check_key_cache is not None:
            _check_key_cache[auth_key] = result
        return result

    def list_users(self):
        with self.Session() as session:
            rows = session.query(UserAuth).all()
            for r in rows:
                logger.info("User: %s", r.__dict__)

    def generate_user_key(self, user_name, priority):
        # TODO make better hashing
        new_key = f"{user_name}+{priority}+{id(self)}"
        new_key = shuffle_string(new_key)
        new_key = get_key_hash(new_key)
        return new_key

    def register_new_user(self, user_name: str, priority: int, key: Optional[str] = None):
        new_key = key
        if new_key is None:
            new_key = self.generate_user_key(user_name, priority)
        exists, *_ = self.check_user_key(new_key)
        if not exists:
            now_ts = self.get_current_ts()
            with self.Session() as session:
                new_user = UserAuth(
                    user_key=new_key,
                    user_name=user_name,
                    priority=priority,
                    created_at=now_ts,
                    expires_at=now_ts + SIX_MONTHS_SECONDS,
                    is_active=1,
                )
                session.add(new_user)
                session.commit()
        return user_name, priority, new_key

    def set_user_models(self, user_id: int, models: str):
        """Set allowed models for a user. Pass empty string or 'all' to allow all."""
        allowed = None if models in ('', 'all', 'null') else models
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            user.allowed_models = allowed
            session.commit()
            print(f"User {user_id} allowed_models set to: {allowed}")

    def set_expiry(self, user_id: int, duration: str):
        """Set token expiry. Duration examples: '6m', '1y', '30d', '2026-12-31'."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            expires_at = self._parse_duration(duration)
            user.expires_at = expires_at
            session.commit()
            print(f"User {user_id} expires_at set to {expires_at}")

    def extend_token(self, user_id: int, duration: str):
        """Extend token expiry from now by duration."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            expires_at = self._parse_duration(duration)
            user.expires_at = expires_at
            session.commit()
            print(f"User {user_id} expires_at extended to {expires_at}")

    def revoke_user(self, user_id: int):
        """Deactivate a single user token."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            user.is_active = 0
            session.commit()
            print(f"User {user_id} revoked")

    def reactivate_user(self, user_id: int):
        """Reactivate a single user token."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            user.is_active = 1
            session.commit()
            print(f"User {user_id} reactivated")

    def revoke_all_tokens(self):
        """Deactivate ALL user tokens."""
        with self.Session() as session:
            count = session.query(UserAuth).update({UserAuth.is_active: 0})
            session.commit()
            print(f"Revoked {count} tokens")

    @staticmethod
    def _parse_duration(duration: str) -> int:
        """Parse duration string to unix timestamp.
        Supports: '6m' (months), '1y' (years), '30d' (days), ISO date '2026-12-31'.
        """
        now = int(datetime.now(timezone.utc).timestamp())
        if '-' in duration and len(duration) >= 8:
            dt = datetime.fromisoformat(duration).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        unit = duration[-1]
        value = int(duration[:-1])
        if unit == 'd':
            return now + value * 86400
        elif unit == 'm':
            return now + value * 30 * 86400
        elif unit == 'y':
            return now + value * 365 * 86400
        else:
            raise ValueError(f"Unknown duration format: {duration}. Use '6m', '1y', '30d', or '2026-12-31'.")

    def save_response(
        self,
        request: dict,
        response: dict,
        user_id: int,
        model_name: str = None,
        prompt_tokens: int = None,
        completion_tokens: int = None,
        total_tokens: int = None,
    ) -> None:
        with self.Session() as session:
            new_response = Requests(
                request=self._serialize_content(request),
                response=self._serialize_content(response),
                user_id=user_id,
                timestamp=self.get_current_ts(),
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            session.add(new_response)
            session.commit()

    def increment_token_usage(self, user_id: int, tokens: int) -> None:
        """Increment the total_tokens_used counter for a user."""
        with self.Session() as session:
            session.query(UserAuth).filter(UserAuth.id == user_id).update(
                {UserAuth.total_tokens_used: UserAuth.total_tokens_used + tokens}
            )
            session.commit()

    def increment_request_count(self, user_id: int) -> None:
        """Increment the total_requests counter for a user."""
        with self.Session() as session:
            session.query(UserAuth).filter(UserAuth.id == user_id).update(
                {UserAuth.total_requests: UserAuth.total_requests + 1}
            )
            session.commit()

    def _serialize_content(self, content: Union[None, str, List[Dict[str, Any]]]) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    # --- Billing CLI commands (F4) ---

    def set_token_budget(self, user_id: int, budget: int):
        """Set token budget for a user. 0 or negative = unlimited."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            user.token_budget = None if budget <= 0 else budget
            session.commit()
            print(f"User {user_id} token_budget set to: {user.token_budget}")

    def reset_token_usage(self, user_id: int):
        """Reset the token usage counter for a user."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            user.total_tokens_used = 0
            user.total_requests = 0
            session.commit()
            print(f"User {user_id} total_tokens_used and total_requests reset to 0")

    def get_usage(self, user_id: int):
        """Get usage stats for a user."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            total = session.query(func.sum(Requests.total_tokens)).filter(
                Requests.user_id == user_id
            ).scalar() or 0
            count = session.query(func.count(Requests.id)).filter(
                Requests.user_id == user_id
            ).scalar() or 0
            by_model = session.query(
                Requests.model, func.sum(Requests.total_tokens), func.count(Requests.id)
            ).filter(Requests.user_id == user_id).group_by(Requests.model).all()

            print(f"User {user_id} ({user.user_name}):")
            print(f"  Total tokens used (counter): {user.total_tokens_used}")
            print(f"  Token budget: {user.token_budget or 'unlimited'}")
            print(f"  Total requests: {count}")
            print(f"  Total tokens (from requests): {total}")
            if by_model:
                print("  By model:")
                for model, tokens, req_count in by_model:
                    print(f"    {model}: {tokens} tokens, {req_count} requests")

    def get_usage_all(self):
        """Get usage summary for all users."""
        with self.Session() as session:
            users = session.query(UserAuth).all()
            print(f"{'ID':<5} {'Name':<20} {'Tokens Used':<15} {'Budget':<15} {'Requests':<10}")
            print("-" * 65)
            for u in users:
                req_count = session.query(func.count(Requests.id)).filter(
                    Requests.user_id == u.id
                ).scalar() or 0
                budget_str = str(u.token_budget) if u.token_budget else "unlimited"
                print(f"{u.id:<5} {u.user_name:<20} {u.total_tokens_used:<15} {budget_str:<15} {req_count:<10}")

    # --- Rate limit CLI commands (F5) ---

    def set_rate_limit(self, user_id: int, field: str, value: int):
        """Set a rate limit field. Fields: rate_limit_tokens_per_min/hour/day, rate_limit_requests_per_min."""
        valid_fields = {
            'rate_limit_tokens_per_min', 'rate_limit_tokens_per_hour',
            'rate_limit_tokens_per_day', 'rate_limit_requests_per_min',
        }
        if field not in valid_fields:
            print(f"Invalid field. Valid: {', '.join(sorted(valid_fields))}")
            return
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            setattr(user, field, None if value <= 0 else value)
            session.commit()
            print(f"User {user_id} {field} set to: {getattr(user, field)}")

    def clear_rate_limits(self, user_id: int):
        """Clear all rate limits for a user."""
        with self.Session() as session:
            user = session.query(UserAuth).filter(UserAuth.id == user_id).first()
            if user is None:
                print(f"User {user_id} not found")
                return
            user.rate_limit_tokens_per_min = None
            user.rate_limit_tokens_per_hour = None
            user.rate_limit_tokens_per_day = None
            user.rate_limit_requests_per_min = None
            session.commit()

    # --- Container key management ---

    def save_container_key(self, model_alias: str, api_key: str) -> None:
        """Save or update API key for a container model."""
        with self.Session() as session:
            existing = session.query(ContainerKey).filter(ContainerKey.model_alias == model_alias).first()
            if existing:
                existing.api_key = api_key
                existing.created_at = self.get_current_ts()
            else:
                session.add(ContainerKey(
                    model_alias=model_alias,
                    api_key=api_key,
                    created_at=self.get_current_ts(),
                ))
            session.commit()

    def get_container_key(self, model_alias: str) -> Optional[str]:
        """Get API key for a container model."""
        with self.Session() as session:
            row = session.query(ContainerKey).filter(ContainerKey.model_alias == model_alias).first()
            return row.api_key if row else None

    def delete_container_key(self, model_alias: str) -> None:
        """Delete API key for a container model."""
        with self.Session() as session:
            session.query(ContainerKey).filter(ContainerKey.model_alias == model_alias).delete()
            session.commit()

    def clear_container_keys(self) -> None:
        """Delete all container keys."""
        with self.Session() as session:
            session.query(ContainerKey).delete()
            session.commit()
