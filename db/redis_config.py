import redis.asyncio as redis
from config.settings import settings


class RedisManager:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )

        return cls._client