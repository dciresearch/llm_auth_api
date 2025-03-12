
import json
from functools import lru_cache
from datetime import datetime, timezone
from sqlalchemy import create_engine, Integer, String, Text, MetaData, func
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped
from typing import Optional, List, Any, Dict, Union
from .utils import get_key_hash, shuffle_string

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


class UserAuth(Base):
    __tablename__ = "user_auth_keys"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_key: Mapped[str] = mapped_column(String, index=True)
    user_name: Mapped[str]
    priority: Mapped[int]


class Database:
    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}?charset=utf8")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def get_current_ts() -> int:
        return int(datetime.now().replace(tzinfo=timezone.utc).timestamp())

    @lru_cache(1000)
    def check_user_key(self, auth_key: str) -> List[Any]:
        with self.Session() as session:
            keys = session.query(UserAuth).filter(UserAuth.user_key == auth_key).all()
            if not keys:
                return False, None, None
            priority = keys[0].priority
            user_id = keys[0].id
            return True, user_id, priority

    def list_users(self):
        with self.Session() as session:
            rows = session.query(UserAuth).all()
            for r in rows:
                print(r.__dict__)

    def generate_user_key(self, user_name, priority):
        # TODO make better hashing
        new_key = f"{user_name}+{priority}+{id(self)}"
        new_key = shuffle_string(new_key)
        new_key = get_key_hash(new_key)
        return new_key

    def register_new_user(self, user_name, priority):
        new_key = self.generate_user_key(user_name, priority)
        exists, _, _ = self.check_user_key(new_key)
        if not exists:
            with self.Session() as session:
                new_user = UserAuth(
                    user_key=new_key,
                    user_name=user_name,
                    priority=priority,
                )
                session.add(new_user)
                session.commit()
        return user_name, priority, new_key

    def save_response(
        self,
        request: dict,
        response: dict,
        user_id: int,
        model_name: str = None,
    ) -> None:
        with self.Session() as session:
            new_response = Requests(
                request=self._serialize_content(request),
                response=self._serialize_content(response),
                user_id=user_id,
                timestamp=self.get_current_ts(),
                model=model_name,
            )
            session.add(new_response)
            session.commit()

    def _serialize_content(self, content: Union[None, str, List[Dict[str, Any]]]) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
