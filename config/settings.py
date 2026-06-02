import os


class Settings:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    SHORT_TERM_MEMORY_TTL = int(os.getenv("SHORT_TERM_MEMORY_TTL", "7200"))

    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "aiagent")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "aiagent")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "aiagent")
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        (
            "postgresql+asyncpg://"
            f"{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        ),
    )

    JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    API_BASE_URL = os.getenv("API_BASE_URL", "http://fastapi:8000")


settings = Settings()

OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
QDRANT_HOST = settings.QDRANT_HOST
QDRANT_PORT = settings.QDRANT_PORT
REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
SHORT_TERM_MEMORY_TTL = settings.SHORT_TERM_MEMORY_TTL
POSTGRES_HOST = settings.POSTGRES_HOST
POSTGRES_PORT = settings.POSTGRES_PORT
POSTGRES_USER = settings.POSTGRES_USER
POSTGRES_PASSWORD = settings.POSTGRES_PASSWORD
POSTGRES_DB = settings.POSTGRES_DB
DATABASE_URL = settings.DATABASE_URL
JWT_SECRET = settings.JWT_SECRET
JWT_ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
API_BASE_URL = settings.API_BASE_URL
