from collections.abc import Callable

from fastapi import Depends, HTTPException

from api.middleware.auth import get_current_user


ROLE_PERMISSIONS = {
    "admin": {"can_upload": True, "can_query": True, "can_admin": True},
    "user": {"can_upload": True, "can_query": True, "can_admin": False},
    "viewer": {"can_upload": False, "can_query": True, "can_admin": False},
}


class RBACChecker:
    @staticmethod
    def require_permission(user: dict, permission: str):
        allowed = ROLE_PERMISSIONS.get(user.get("role"), {}).get(permission, False)
        if not allowed:
            raise HTTPException(status_code=403, detail="Permission denied")
        return user


def require_permission(permission: str) -> Callable:
    async def dependency(user: dict = Depends(get_current_user)):
        return RBACChecker.require_permission(user, permission)

    return dependency
