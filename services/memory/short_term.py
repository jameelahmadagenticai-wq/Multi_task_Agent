import json
from datetime import datetime

from db.redis_config import RedisManager
from config.settings import settings


class ShortTermMemory:

    def __init__(self):
        self.redis = RedisManager.get_client()

    def _build_key(self, user_id: str, session_id: str):
        return f"session:{user_id}:{session_id}"

    async def add_turn(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        agent_used: str = None,
        tokens_used: int = 0
    ):

        key = self._build_key(user_id, session_id)

        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_used": agent_used,
            "tokens_used": tokens_used
        }

        await self.redis.rpush(key, json.dumps(turn))

        await self.redis.expire(
            key,
            settings.SHORT_TERM_MEMORY_TTL
        )

    async def get_history(
        self,
        user_id: str,
        session_id: str,
        last_n: int = 10
    ):

        key = self._build_key(user_id, session_id)

        history = await self.redis.lrange(
            key,
            -last_n,
            -1
        )

        return [json.loads(item) for item in history]

    async def clear_session(
        self,
        user_id: str,
        session_id: str
    ):

        key = self._build_key(user_id, session_id)

        await self.redis.delete(key)