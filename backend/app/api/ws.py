import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as redis
from backend.app.core.config import settings

router = APIRouter(prefix="/ws", tags=["websockets"])
logger = logging.getLogger("WebSocketManager")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.redis_client: redis.Redis | None = None
        self.pubsub = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total clients: {len(self.active_connections)}"
        )
        await start_ws_listener()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(
                f"WebSocket disconnected. Total clients: {len(self.active_connections)}"
            )

    async def broadcast(self, payload: dict):
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.REDIS_URL, decode_responses=True
            )
        await self.redis_client.publish("ibvap_events", json.dumps(payload))

    async def _redis_listener(self):
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.REDIS_URL, decode_responses=True
            )
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe("ibvap_events")
            logger.info("Subscribed to Redis channel: ibvap_events")

        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    payload = message["data"]
                    # Broadcast to all connected clients
                    disconnected = []
                    for connection in self.active_connections:
                        try:
                            await connection.send_text(payload)
                        except Exception as e:
                            logger.error(f"Error sending message to client: {e}")
                            disconnected.append(connection)

                    for conn in disconnected:
                        self.disconnect(conn)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis listener error: {e}")


manager = ConnectionManager()
listener_task = None


async def start_ws_listener():
    global listener_task
    if listener_task is None or listener_task.done():
        listener_task = asyncio.create_task(manager._redis_listener())


async def stop_ws_listener():
    global listener_task
    if listener_task:
        listener_task.cancel()
    if manager.pubsub:
        await manager.pubsub.unsubscribe("ibvap_events")
        await manager.pubsub.close()
    if manager.redis_client:
        await manager.redis_client.close()


@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket, token: str = None):
    from backend.app.core.security import _decode_user
    from backend.app.core.database import SessionLocal

    if not token:
        await websocket.close(code=1008, reason="Authentication required")
        return

    db = SessionLocal()
    try:
        user = _decode_user(token, db)
        logger.info("WebSocket connection authenticated for user: %s", user.username)
    except Exception:
        await websocket.close(code=1008, reason="Invalid or expired credentials")
        return
    finally:
        db.close()

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, ConnectionResetError, OSError):
        manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket client connection closed: %s", exc)
        manager.disconnect(websocket)
