"""System health API endpoints.

Provides endpoints for monitoring system resources including
CPU, memory, disk, and GPU metrics.
"""

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from backend.app.core.security import get_current_user
from backend.app.core.database import SessionLocal
from backend.app.services.camera_manager import camera_manager
from backend.app.services.ai_scheduler import ai_scheduler
from backend.app.services.anpr.anpr_service import anpr_service
from backend.app.services.system_health_service import SystemHealthService
from backend.app.services.detection_service import detection_service
from backend.app.services.border_rules_service import border_rules_service

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health/detailed")
async def get_system_health(_user=Depends(get_current_user)):
    """Get detailed system health metrics.

    Returns:
        {
            "status": "HEALTHY" | "DEGRADED" | "CRITICAL",
            "cpu": {
                "percent": float (0-100),
                "warning": bool,
                "critical": bool
            },
            "memory": {
                "percent": float (0-100),
                "used_mb": float,
                "total_mb": float,
                "warning": bool,
                "critical": bool
            },
            "disk": {
                "percent": float (0-100),
                "free_gb": float,
                "warning": bool,
                "critical": bool
            },
            "gpu": {
                "available": bool,
                "percent": float | null,
                "memory_percent": float | null,
                "memory_used_mb": float | null,
                "memory_total_mb": float | null
            } | null
        }
    """
    health = SystemHealthService.get_system_health()
    cameras = list(camera_manager.streams.values())
    database_status = "HEALTHY"
    db = None
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
    except Exception:
        database_status = "DEGRADED"
    finally:
        if db is not None:
            db.close()

    return {
        "status": "DEGRADED"
        if database_status != "HEALTHY"
        else SystemHealthService.get_status_string(health),
        "cpu": {
            "percent": round(health.cpu_percent, 1),
            "warning": health.cpu_percent > SystemHealthService.CPU_WARNING,
            "critical": health.cpu_percent > SystemHealthService.CPU_CRITICAL,
        },
        "memory": {
            "percent": round(health.memory_percent, 1),
            "used_mb": round(health.memory_used_mb, 1),
            "total_mb": round(health.memory_total_mb, 1),
            "warning": health.memory_percent > SystemHealthService.MEMORY_WARNING,
            "critical": health.memory_percent > SystemHealthService.MEMORY_CRITICAL,
        },
        "disk": {
            "percent": round(health.disk_percent, 1),
            "free_gb": round(health.disk_free_gb, 1),
            "warning": health.disk_percent > SystemHealthService.DISK_WARNING,
            "critical": health.disk_percent > SystemHealthService.DISK_CRITICAL,
        },
        "gpu": {
            "available": health.gpu_available,
            "percent": round(health.gpu_percent, 1)
            if health.gpu_percent is not None
            else None,
            "memory_percent": round(health.gpu_memory_percent, 1)
            if health.gpu_memory_percent is not None
            else None,
            "memory_used_mb": round(health.gpu_memory_used_mb, 1)
            if health.gpu_memory_used_mb is not None
            else None,
            "memory_total_mb": round(health.gpu_memory_total_mb, 1)
            if health.gpu_memory_total_mb is not None
            else None,
        }
        if health.gpu_available
        else None,
        "database_status": database_status,
        "cameras": {
            "online": sum(1 for camera in cameras if camera.status == "ONLINE"),
            "total": len(cameras),
            "inference_fps": round(
                sum(
                    ai_scheduler.get_metrics(c.camera_id).get("detector_fps", 0.0)
                    for c in cameras
                ),
                1,
            ),
            "stale": [
                c.camera_id
                for c in cameras
                if (ai_scheduler.get_metrics(c.camera_id).get("age_seconds") or 0) > 5
            ],
        },
        "stage_profiling_ms": {
            "avg_inference_ms": round(
                sum(
                    ai_scheduler.get_metrics(c.camera_id).get("inference_ms", 0.0)
                    for c in cameras
                )
                / max(len(cameras), 1),
                2,
            ),
            "avg_tracking_ms": round(
                sum(
                    ai_scheduler.get_metrics(c.camera_id).get("tracking_ms", 0.0)
                    for c in cameras
                )
                / max(len(cameras), 1),
                2,
            ),
            "avg_zone_ms": round(
                sum(
                    ai_scheduler.get_metrics(c.camera_id).get("zone_ms", 0.0)
                    for c in cameras
                )
                / max(len(cameras), 1),
                2,
            ),
            "avg_anpr_ms": round(
                sum(
                    ai_scheduler.get_metrics(c.camera_id).get("anpr_ms", 0.0)
                    for c in cameras
                )
                / max(len(cameras), 1),
                2,
            ),
            "avg_total_pipeline_ms": round(
                sum(
                    ai_scheduler.get_metrics(c.camera_id).get("total_pipeline_ms", 0.0)
                    for c in cameras
                )
                / max(len(cameras), 1),
                2,
            ),
            "per_camera": {
                c.camera_id: ai_scheduler.get_metrics(c.camera_id) for c in cameras
            },
        },
        "anpr": {"queue_depth": anpr_service.queue_depth()},
        "ground_ai": {
            "status": detection_service.ground_status,
            "fps": detection_service.ground_fps,
            "persons": detection_service.ground_counts["person"],
            "vehicles": detection_service.ground_counts["vehicle"],
        },
        "air_ai": {
            "status": detection_service.airborne_status,
            "fps": detection_service.airborne_fps,
            "drones": detection_service.airborne_counts["drone"],
            "aircraft": detection_service.airborne_counts["aircraft"],
        },
        "security_item_ai": {
            "status": detection_service.security_item_status,
            "fps": detection_service.security_item_fps,
            "firearms": detection_service.security_item_counts["firearm"],
            "model": "YOLO11n",
            "version": "v1.0",
        },
        "virtual_fence": {
            "ground_zones": sum(
                len(c.get("restricted_zones", []))
                for c in border_rules_service.zones_config.values()
            ),
            "air_zones": sum(
                len(c.get("airborne_zones", []))
                for c in border_rules_service.zones_config.values()
            ),
            "tripwires": sum(
                len(c.get("virtual_fences", [])) + len(c.get("air_fences", []))
                for c in border_rules_service.zones_config.values()
            ),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/health/quick")
async def get_quick_health(_user=Depends(get_current_user)):
    """Get quick system status (HEALTHY/DEGRADED/CRITICAL only).

    Returns:
        {
            "status": "HEALTHY" | "DEGRADED" | "CRITICAL",
            "is_healthy": bool
        }
    """
    return {
        "status": SystemHealthService.get_status_string(),
        "is_healthy": SystemHealthService.is_healthy(),
    }
