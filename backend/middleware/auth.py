from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from services.auth_service import AuthService

security = HTTPBearer()


async def get_auth_service() -> AuthService:
    from main import auth_service
    return auth_service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth: AuthService = Depends(get_auth_service),
) -> dict:
    token = credentials.credentials
    user_id = auth.verify_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证令牌无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = auth.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被禁用",
        )
    return {"user_id": user_id, "user": user}


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    auth: AuthService = Depends(get_auth_service),
) -> Optional[dict]:
    if credentials is None:
        return None
    token = credentials.credentials
    user_id = auth.verify_token(token)
    if user_id is None:
        return None
    user = auth.get_user(user_id)
    if not user:
        return None
    return {"user_id": user_id, "user": user}