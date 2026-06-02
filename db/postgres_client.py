from collections.abc import AsyncGenerator

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings


class PostgresManager:
    _pool = None
    _engine = None
    _session_factory = None

    @classmethod
    def database_url(cls) -> str:
        if settings.DATABASE_URL:
            return settings.DATABASE_URL
        return (
            "postgresql+asyncpg://"
            f"{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )

    @classmethod
    async def get_pool(cls):
        """Legacy asyncpg pool kept for modules that still need raw driver access."""
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
                min_size=2,
                max_size=10,
                timeout=3,
            )
        return cls._pool

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            cls._engine = create_async_engine(
                cls.database_url(),
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )
        return cls._engine

    @classmethod
    def get_session_factory(cls):
        if cls._session_factory is None:
            cls._session_factory = async_sessionmaker(
                cls.get_engine(),
                expire_on_commit=False,
                autoflush=False,
            )
        return cls._session_factory

    @classmethod
    async def session(cls) -> AsyncGenerator[AsyncSession, None]:
        async with cls.get_session_factory()() as db:
            yield db
