"""Controlled, clearly labelled demo scenarios for a reproducible presentation."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.core.security import require_roles
from backend.app.models.event import EventAudit, PlateEvent, SecurityEvent
from backend.app.models.user import User
from backend.app.services.camera_manager import camera_manager
from backend.app.services.evidence_service import evidence_service
from backend.app.services.risk_engine import risk_engine


def verify_demo_mode():
    if not settings.DEMO_MODE:
        raise HTTPException(status_code=403, detail="Demo mode is disabled")


router = APIRouter(
    prefix="/demo", tags=["demo"], dependencies=[Depends(verify_demo_mode)]
)


def _camera_or_error(camera_id: str):
    frame = camera_manager.get_raw_frame(camera_id)
    if frame is None:
        raise HTTPException(
            status_code=409,
            detail="Camera is not ready; wait for its stream to become ONLINE",
        )
    return frame


@router.post("/seed-intrusion")
def seed_intrusion(
    camera_id: str = "CAM-002",
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("ADMIN", "OPERATOR")),
):
    """Create a labelled, evidence-backed intrusion to validate the full UI flow."""
    frame = _camera_or_error(camera_id)
    event_data = risk_engine.evaluate_event(
        {
            "event_type": "ZONE_ENTRY",
            "camera_id": camera_id,
            "zone_id": "zone_cam002_restricted",
            "track_id": "DEMO-P-001",
            "object_type": "PERSON",
            "direction": "INWARD",
            # Reproducible night-context scenario, visibly labelled in the audit.
            "timestamp": datetime.utcnow().replace(
                hour=2, minute=0, second=0, microsecond=0
            ),
            "confidence": 1.0,
            "operator_notes": "SIMULATED DEMO EVENT: controlled intrusion scenario",
        }
    )
    event = SecurityEvent(**event_data, data_origin="DEMO")
    db.add(event)
    db.flush()
    event.snapshot_path, _ = evidence_service.capture_event(camera_id, event.id, frame)
    db.add(
        EventAudit(
            event_id=event.id,
            action="SIMULATED",
            actor=user.username,
            notes="Controlled demo intrusion seeded",
        )
    )
    db.commit()
    return {
        "id": f"evt-{event.id}",
        "message": "Demo intrusion created; clip will be ready in about 7 seconds.",
    }


@router.post("/seed-watchlist")
def seed_watchlist_match(
    camera_id: str = "CAM-002",
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("ADMIN", "OPERATOR")),
):
    """Seed the configured test plate so ANPR can be demonstrated on demand."""
    _camera_or_error(camera_id)
    plate = "AB12CD34"
    plate_event = PlateEvent(
        camera_id=camera_id,
        track_id="DEMO-V-001",
        plate_number=plate,
        confidence=0.99,
        watchlist_status="WATCHLIST MATCH",
        data_origin="DEMO",
    )
    db.add(plate_event)
    event_data = risk_engine.evaluate_event(
        {
            "event_type": "WATCHLIST_MATCH",
            "camera_id": camera_id,
            "track_id": "DEMO-V-001",
            "object_type": "VEHICLE",
            "timestamp": datetime.utcnow(),
            "watchlist_status": "WATCHLIST MATCH",
            "operator_notes": "SIMULATED DEMO EVENT: configured ANPR watchlist match for AB12CD34",
        }
    )
    # Watchlist status is scoring context, not a SecurityEvent database field.
    event_data.pop("watchlist_status", None)
    event = SecurityEvent(**event_data, data_origin="DEMO")
    db.add(event)
    db.flush()
    db.add(
        EventAudit(
            event_id=event.id,
            action="SIMULATED",
            actor=user.username,
            notes="Controlled ANPR watchlist scenario seeded",
        )
    )
    db.commit()
    return {
        "plate_event_id": plate_event.id,
        "event_id": f"evt-{event.id}",
        "message": "Demo watchlist match created.",
    }
