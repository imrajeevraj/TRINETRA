import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.database import engine
from backend.app.core.config import settings
from backend.app.core.database import SessionLocal
from backend.app.core.security import bootstrap_admin, validate_security_settings
from backend.app.models import Base
from backend.app.services.camera_manager import camera_manager
from backend.app.api.cameras import router as cameras_router
from backend.app.api.events import router as events_router
from backend.app.api.auth import router as auth_router
from backend.app.api.anpr import router as anpr_router
from backend.app.api.system import router as system_router
from backend.app.api.demo import router as demo_router
from backend.app.api.ws import router as ws_router
from backend.app.api.metrics import router as metrics_router
from backend.app.api.ptz import router as ptz_router
from backend.app.api.incidents import router as incidents_router, entities_router
from backend.app.api.predictions import router as predictions_router, ptz_router as predictive_ptz_router
from backend.app.api.handovers import router as handovers_router, camera_res_router
from backend.app.api.edge_mesh import router as edge_mesh_router
from backend.app.api.models_governance import (
    router as models_governance_router,
    datasets_router,
    training_router,
)
from backend.app.api.ai_quality_api import (
    feedback_router as ai_feedback_router,
    quality_router as ai_quality_router,
    learning_router as ai_learning_router,
)
from backend.app.api.multimodal_api import (
    sensors_router,
    multimodal_router,
    thermal_router,
)
from backend.app.services.geo_service import geo_service

# ── O-01: Structured JSON logging ───────────────────────────────────────────
import json as _json


class _JSONFormatter(logging.Formatter):
    """O-01: Emit each log record as a single-line JSON object.

    Format::
        {"ts": "<ISO8601>", "level": "INFO", "logger": "Main", "msg": "..."}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return _json.dumps(payload, ensure_ascii=False)


_handler = logging.StreamHandler()
_handler.setFormatter(_JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger("Main")

_ENV = os.environ.get("ENV", "development").lower()
_IS_PRODUCTION = _ENV == "production"

# C-06: OpenAPI tags and structured metadata for command-center API docs
OPENAPI_TAGS = [
    {
        "name": "auth",
        "description": "Authentication, JWT tokens, session lifecycle, and role verification.",
    },
    {
        "name": "cameras",
        "description": "Surveillance camera feeds, RTSP ingestion, and live MJPEG streams.",
    },
    {
        "name": "events",
        "description": "Security incident detection, risk scoring, evidence retrieval, and operator audits.",
    },
    {
        "name": "anpr",
        "description": "Automatic Number Plate Recognition, high-speed vehicle tracking, and stolen vehicle checks.",
    },
    {
        "name": "system",
        "description": "Real-time edge hardware monitoring (CPU, GPU, memory, disk) and pipeline telemetry.",
    },
    {
        "name": "demo",
        "description": "Controlled reproducible simulation scenarios for demonstrations.",
    },
    {
        "name": "Observability",
        "description": "Prometheus metrics scraper endpoint and operational diagnostics.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    validate_security_settings()
    # C-03: create_all() bypasses Alembic and is removed from production startup.
    # Run `alembic upgrade head` in the deployment runbook before starting the app.
    if not _IS_PRODUCTION:
        try:
            Base.metadata.create_all(bind=engine)
        except Exception as e:
            logger.warning(f"Base.metadata.create_all warning (dev-only): {e}")
    db = SessionLocal()
    try:
        bootstrap_admin(db)
    finally:
        db.close()

    logger.info("Starting video ingestion streams...")
    camera_manager.start_all()

    from backend.app.api.ws import start_ws_listener, stop_ws_listener

    await start_ws_listener()

    # R-06: Fault-isolate geo_service startup — a missing GPS serial port must
    # not prevent the rest of the platform from starting.
    try:
        geo_service.start(loop=asyncio.get_running_loop())
    except Exception as e:
        logger.warning(
            "geo_service could not start (GPS unavailable?): %s — continuing in degraded mode",
            e,
        )

    yield

    # ── Shutdown (R-01: flush open evidence capture jobs before exit) ───────────
    logger.info("Stopping video ingestion streams...")
    camera_manager.stop_all()

    # R-01: Give in-progress EvidenceService CaptureJobs time to finalise clips
    # so no forensic evidence is lost on SIGTERM/container restart.
    from backend.app.services.evidence_service import evidence_service
    import time as _time

    shutdown_deadline = _time.monotonic() + 12  # max 12 s grace period
    while evidence_service.jobs and _time.monotonic() < shutdown_deadline:
        remaining = list(evidence_service.jobs.keys())
        logger.info(
            "Waiting for %d in-progress evidence job(s) to finish: %s",
            len(remaining),
            remaining,
        )
        await asyncio.sleep(0.5)
    if evidence_service.jobs:
        logger.warning(
            "Shutdown deadline reached; %d evidence job(s) were abandoned: %s",
            len(evidence_service.jobs),
            list(evidence_service.jobs.keys()),
        )

    geo_service.stop()
    await stop_ws_listener()
    # R-03: Stop the ConfigManager file-watcher thread.
    from backend.app.core.config import config_manager

    config_manager.stop_watching()


# S-04 / C-06: Swagger UI (/docs) and ReDoc (/redoc) are gated by ENV, with full OpenAPI tags
app = FastAPI(
    title="TRINETRA Command Center API",
    description="TRINETRA — Threat Recognition, Intelligent Networked Eyes & Tracking for Real-time Analysis (Three Eyes. One Secure Border. — SIH26187 / MHA & SSB)",
    version="2.0.0",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
    docs_url=None if _IS_PRODUCTION else "/docs",
    redoc_url=None if _IS_PRODUCTION else "/redoc",
    openapi_url=None if _IS_PRODUCTION else "/openapi.json",
)


# ── Security Headers Middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    # S-06: Allow WebSocket, MJPEG streams, and MapLibre geospatial tiles/workers
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "connect-src 'self' ws: wss: https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com https://demotiles.maplibre.org; "
        "worker-src 'self' blob:; "
        "child-src 'self' blob:; "
        "img-src 'self' data: blob: https://basemaps.cartocdn.com https://*.basemaps.cartocdn.com https://tile.openstreetmap.org https://*.tile.openstreetmap.org; "
        "media-src 'self' blob:;"
    )
    return response


# ── X-Request-ID Tracing Middleware (S-12 / O-05) ──────────────────────────────
@app.middleware("http")
async def request_id_middleware(request, call_next):
    """S-12 / O-05: Attach a trace ID to every request for end-to-end log correlation.

    The caller may supply their own X-Request-ID (e.g. from a load-balancer);
    if absent a UUID4 is generated. The ID is echoed in the response header
    and included in all structured telemetry.
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── CORS (S-05: scoped methods and headers — no wildcard in production) ────────
_CORS_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_CORS_HEADERS = ["Content-Type", "Authorization", "X-Requested-With", "X-Request-ID"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=_CORS_METHODS,
    allow_headers=_CORS_HEADERS,
)


# ── API Routers ────────────────────────────────────────────────────────────────
app.include_router(cameras_router, prefix="/api")
app.include_router(events_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(anpr_router, prefix="/api")
app.include_router(system_router, prefix="/api")
app.include_router(demo_router, prefix="/api")
app.include_router(ptz_router, prefix="/api")
app.include_router(incidents_router, prefix="/api")
app.include_router(entities_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")
app.include_router(predictive_ptz_router, prefix="/api")
app.include_router(handovers_router, prefix="/api")
app.include_router(camera_res_router, prefix="/api")
app.include_router(edge_mesh_router, prefix="/api")
app.include_router(models_governance_router)
app.include_router(datasets_router)
app.include_router(training_router)
app.include_router(ai_feedback_router)
app.include_router(ai_quality_router)
app.include_router(ai_learning_router)
app.include_router(sensors_router, prefix="/api")
app.include_router(multimodal_router, prefix="/api")
app.include_router(thermal_router, prefix="/api")
app.include_router(metrics_router)  # P-07: Exposes GET /metrics
app.include_router(ws_router)


@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "app": "TRINETRA — Threat Recognition, Intelligent Networked Eyes & Tracking for Real-time Analysis",
        "tagline": "Three Eyes. One Secure Border.",
        "version": "2.0.0",
    }


@app.get("/api/health")
def health_check():
    """O-02: Real liveness probe used by Docker healthcheck and load-balancers.

    Checks:
    - PostgreSQL connectivity (SELECT 1)
    - Redis connectivity (PING)

    Returns HTTP 200 only if all critical dependencies are reachable.
    Returns HTTP 503 if any dependency is degraded.
    """
    import redis as _redis
    from fastapi.responses import JSONResponse

    checks: dict = {"database": "ok", "redis": "ok"}
    healthy = True

    # DB check
    db = None
    try:
        db = SessionLocal()
        from sqlalchemy import text as _text

        db.execute(_text("SELECT 1"))
    except Exception as e:
        checks["database"] = f"degraded: {e}"
        healthy = False
    finally:
        if db:
            db.close()

    # Redis check
    try:
        _r = _redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        _r.ping()
    except Exception as e:
        checks["redis"] = f"degraded: {e}"
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "healthy" if healthy else "degraded", "checks": checks},
    )
