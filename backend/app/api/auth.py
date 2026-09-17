from fastapi import APIRouter, Depends, HTTPException, Response, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import logging
import threading
import time

import redis

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.core.security import (
    create_access_token,
    get_current_user,
    verify_password,
)
from backend.app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("Auth")

MAX_ATTEMPTS = 5
LOCKOUT_WINDOW = 60  # seconds

# ── R-07: Redis circuit-breaker ────────────────────────────────────────────────
# _redis_available acts as an open/closed circuit flag.  When Redis is down
# the rate-limiter degrades gracefully (login is still allowed, just un-throttled).
# A background probe re-closes the circuit when Redis recovers.
_redis_client: redis.Redis | None = None
_redis_available: bool = False
_redis_lock = threading.Lock()


def _make_redis_client() -> redis.Redis:
    return redis.from_url(
        settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2
    )


def _open_redis_circuit() -> None:
    """R-07: Open the circuit (mark Redis unavailable) from any scope."""
    global _redis_available
    with _redis_lock:
        _redis_available = False


def _probe_redis():
    """R-07: Background probe that re-enables Redis rate-limiting after recovery."""
    global _redis_client, _redis_available
    while True:
        time.sleep(30)
        if not _redis_available:
            try:
                client = _make_redis_client()
                client.ping()
                with _redis_lock:
                    _redis_client = client
                    _redis_available = True
                logger.info("R-07: Redis circuit re-closed — rate limiting restored.")
            except Exception:
                pass  # still down; keep probing


# Initial connection attempt
try:
    _redis_client = _make_redis_client()
    _redis_client.ping()
    _redis_available = True
    logger.info("Redis rate-limiter connected.")
except Exception as e:
    logger.warning(
        "R-07: Redis unavailable at startup — rate limiting degraded. Probe active. (%s)",
        e,
    )

# Start the recovery probe thread
threading.Thread(target=_probe_redis, daemon=True, name="RedisCircuitProbe").start()


@router.post("/login")
def login(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate_limit:login:{client_ip}"
    current_time = time.time()

    # R-07: Only enforce rate-limit when circuit is closed (Redis healthy).
    with _redis_lock:
        r = _redis_client if _redis_available else None

    if r:
        try:
            r.zremrangebyscore(key, 0, current_time - LOCKOUT_WINDOW)
            attempts = r.zcard(key)
            if attempts >= MAX_ATTEMPTS:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please try again later.",
                )
        except redis.RedisError as e:
            # R-07: Open the circuit on Redis error so subsequent requests degrade
            # gracefully rather than raising 500.
            _open_redis_circuit()
            logger.error("R-07: Redis rate-limit error — circuit opened: %s", e)

    user = db.query(User).filter(User.username == form_data.username).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        if r:
            try:
                r.zadd(key, {str(current_time): current_time})
                r.expire(key, LOCKOUT_WINDOW)
            except redis.RedisError:
                pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user.username, user.role)
    response.set_cookie(
        key="ibvap_access_token",
        value=token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        max_age=60 * 60 * 8,
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
    }


@router.get("/me")
def get_me(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response):
    response.delete_cookie("ibvap_access_token")
