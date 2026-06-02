from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config.settings import settings


security = HTTPBearer(auto_error=False)


class AuthManager:
    @staticmethod
    def create_access_token(data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            settings.JWT_SECRET,
            algorithm=settings.JWT_ALGORITHM,
        )

    @staticmethod
    def verify_token(token: str):
        try:
            return jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except JWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid token") from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    payload = AuthManager.verify_token(credentials.credentials)
    return {
        "user_id": payload.get("sub") or payload.get("user_id") or "anonymous",
        "username": payload.get("username", "anonymous"),
        "role": payload.get("role", "viewer"),
    }
