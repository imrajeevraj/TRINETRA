import time
import json
import logging
import redis
from backend.app.core.config import settings

logger = logging.getLogger("EventsPubSub")

_circuit_open_until = 0.0

try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
except Exception as e:
    logger.error(f"Failed to connect to Redis for pub/sub: {e}")
    redis_client = None


def publish_event(event_type: str, data: dict):
    global _circuit_open_until
    now = time.time()
    if not redis_client or now < _circuit_open_until:
        return
    try:
        payload = json.dumps({"type": event_type, "data": data}, default=str)
        redis_client.publish("ibvap_events", payload)
    except Exception as e:
        logger.warning(f"Redis pub/sub unavailable, opening circuit for 30s: {e}")
        _circuit_open_until = now + 30.0
