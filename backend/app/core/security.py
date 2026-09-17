from datetime import datetime, timedelta, timezone
from typing import Callable, List

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.user import User

ALGORITHM = "HS256"
bearer_scheme = HTTPBearer(auto_error=False)


def _secret() -> str:
    if not settings.JWT_SECRET or len(settings.JWT_SECRET) < 32:
        raise RuntimeError("JWT_SECRET must be configured with at least 32 characters")
    return settings.JWT_SECRET


def validate_security_settings() -> None:
    _secret()


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("Password must be at most 72 UTF-8 bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return False


def create_access_token(username: str, role: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": username, "role": role, "exp": expires}
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    token = (
        credentials.credentials
        if credentials
        else request.cookies.get("ibvap_access_token")
    )
    if not token:
        token = request.query_params.get("token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return _decode_user(token, db)


def _decode_user(token: str, db: Session) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
    )
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not isinstance(username, str) or not username:
            raise credentials_error
    except (JWTError, RuntimeError) as exc:
        raise credentials_error from exc
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_error
    return user


def require_roles(*roles: str) -> Callable:
    allowed_roles: List[str] = [role.upper() for role in roles]

    def dependency(user: User = Depends(get_current_user)) -> User:
        if user.role.upper() not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges",
            )
        return user

    return dependency


def bootstrap_admin(db: Session) -> None:
    def seed_user(username: str, password: str, role: str) -> None:
        if len(password) < 8:
            raise RuntimeError(f"{role}_PASSWORD must be at least 8 characters")
        existing = db.query(User).filter(User.username == username).first()
        if existing is None:
            db.add(
                User(
                    username=username,
                    hashed_password=hash_password(password),
                    role=role,
                )
            )

    username = settings.ADMIN_USERNAME
    password = settings.ADMIN_PASSWORD
    if bool(username) != bool(password):
        raise RuntimeError(
            "ADMIN_USERNAME and ADMIN_PASSWORD must be configured together"
        )
    if username:
        seed_user(username, password, "ADMIN")

    if settings.DEMO_MODE:
        if not settings.DEMO_PASSWORD:
            raise RuntimeError("DEMO_PASSWORD is required when DEMO_MODE is enabled")
        seed_user(settings.DEMO_USERNAME, settings.DEMO_PASSWORD, "OPERATOR")

    db.commit()
