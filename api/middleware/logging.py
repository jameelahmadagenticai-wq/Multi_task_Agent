import logging
import asyncio

from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import settings
from db.models import AuditLog
from db.postgres_client import PostgresManager


logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        asyncio.create_task(self._write_audit_log(request, response.status_code))
        return response

    async def _write_audit_log(self, request, response_code: int):
        user_id = self._extract_user_id(request)
        try:
            async with PostgresManager.get_session_factory()() as session:
                session.add(
                    AuditLog(
                        user_id=user_id,
                        endpoint=request.url.path,
                        response_code=response_code,
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.warning("audit_log_write_failed path=%s error=%s", request.url.path, exc)

    def _extract_user_id(self, request):
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return request.headers.get("x-user-id")
        token = auth_header.split(" ", 1)[1]
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )
            return payload.get("sub") or payload.get("user_id")
        except JWTError:
            return None
