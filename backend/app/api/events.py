from datetime import datetime
from typing import Literal
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import json

from backend.app.core.database import get_db
from backend.app.core.security import get_current_user, require_roles
from backend.app.models.user import User
from backend.app.models.event import EventAudit, SecurityEvent

router = APIRouter(prefix="/events", tags=["events"])


class EventDisposition(BaseModel):
    status: str = Field(pattern="^(ACKNOWLEDGED|DISMISSED|ESCALATED|CONFIRMED|REJECTED)$")
    operator_notes: str | None = Field(default=None, max_length=2000)


@router.get("/recent")
def get_recent_events(
    camera_id: str | None = Query(default=None, description="Filter by camera ID"),
    severity: str | None = Query(
        default=None, description="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)"
    ),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    track_id: str | None = Query(
        default=None, description="Filter by track ID (substring)"
    ),
    status: str | None = Query(
        default=None,
        description="Filter by status (NEW, ACKNOWLEDGED, ESCALATED, DISMISSED)",
    ),
    search: str | None = Query(
        default=None, description="Search across track, zone, camera, and notes"
    ),
    start_time: str | None = Query(
        default=None, description="Filter events after start timestamp (ISO)"
    ),
    end_time: str | None = Query(
        default=None, description="Filter events before end timestamp (ISO)"
    ),
    data_origin: Literal["LIVE", "DEMO", "TEST", "IMPORTED"] = "LIVE",
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    query = db.query(SecurityEvent).filter(SecurityEvent.data_origin == data_origin)

    if isinstance(camera_id, str) and camera_id.strip() and camera_id.upper() != "ALL":
        query = query.filter(SecurityEvent.camera_id == camera_id.strip())

    if isinstance(severity, str) and severity.strip() and severity.upper() != "ALL":
        query = query.filter(SecurityEvent.severity == severity.strip().upper())

    if (
        isinstance(event_type, str)
        and event_type.strip()
        and event_type.upper() != "ALL"
    ):
        query = query.filter(SecurityEvent.event_type == event_type.strip())

    if isinstance(status, str) and status.strip() and status.upper() != "ALL":
        query = query.filter(SecurityEvent.status == status.strip().upper())

    if isinstance(track_id, str) and track_id.strip():
        query = query.filter(SecurityEvent.track_id.ilike(f"%{track_id.strip()}%"))

    if isinstance(search, str) and search.strip():
        search_pattern = f"%{search.strip()}%"
        query = query.filter(
            (SecurityEvent.track_id.ilike(search_pattern))
            | (SecurityEvent.zone_id.ilike(search_pattern))
            | (SecurityEvent.camera_id.ilike(search_pattern))
            | (SecurityEvent.event_type.ilike(search_pattern))
            | (SecurityEvent.operator_notes.ilike(search_pattern))
        )

    if isinstance(start_time, str) and start_time.strip():
        try:
            st = datetime.fromisoformat(start_time.strip().replace("Z", "+00:00"))
            query = query.filter(SecurityEvent.timestamp >= st)
        except Exception:
            pass

    if isinstance(end_time, str) and end_time.strip():
        try:
            et = datetime.fromisoformat(end_time.strip().replace("Z", "+00:00"))
            query = query.filter(SecurityEvent.timestamp <= et)
        except Exception:
            pass

    max_limit = limit if isinstance(limit, int) else 50
    events = query.order_by(SecurityEvent.timestamp.desc()).limit(max_limit).all()

    formatted_events = []
    for evt in events:
        # Parse risk_reasons JSON into structured list
        reasons_list = []
        reasons_summary = ""
        try:
            if evt.risk_reasons:
                raw_reasons = json.loads(evt.risk_reasons)
                if isinstance(raw_reasons, list):
                    reasons_list = raw_reasons
                    reasons_summary = " + ".join(raw_reasons)
        except (TypeError, json.JSONDecodeError):
            reasons_summary = evt.risk_reasons or ""
            reasons_list = [reasons_summary] if reasons_summary else []

        formatted_events.append(
            {
                "id": f"evt-{evt.id}",
                "event_type": evt.event_type,
                "severity": evt.severity or "LOW",
                "camera_id": evt.camera_id,
                "track_id": evt.track_id,
                "zone_id": evt.zone_id,
                "object_type": evt.object_type,
                "direction": evt.direction,
                "timestamp": evt.timestamp.isoformat() + "Z",
                "risk_score": evt.risk_score or 0,
                "risk_reasons": reasons_list,
                "reasons_summary": reasons_summary,
                "status": evt.status or "NEW",
                "operator_notes": evt.operator_notes,
                "handled_by": evt.handled_by,
                "handled_at": evt.handled_at.isoformat() + "Z"
                if evt.handled_at
                else None,
                "snapshot_available": bool(evt.snapshot_path),
                "video_clip_available": bool(evt.video_clip_path),
                "snapshot_url": f"/api/events/{evt.id}/evidence/snapshot"
                if evt.snapshot_path
                else None,
                "video_clip_url": f"/api/events/{evt.id}/evidence/clip"
                if evt.video_clip_path
                else None,
                "data_origin": evt.data_origin,
            }
        )

    return formatted_events


@router.patch("/{event_id}/disposition")
def update_event_disposition(
    event_id: int,
    disposition: EventDisposition,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("ADMIN", "OPERATOR")),
):
    event = db.query(SecurityEvent).filter(SecurityEvent.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    event.status = disposition.status
    event.operator_notes = disposition.operator_notes
    event.handled_by = user.username
    event.handled_at = datetime.utcnow()
    db.add(
        EventAudit(
            event_id=event.id,
            action=disposition.status,
            actor=user.username,
            notes=disposition.operator_notes,
        )
    )
    db.commit()
    from backend.app.core.events_pubsub import publish_event

    publish_event("ALERT_UPDATED", {"id": f"evt-{event.id}", "status": event.status})
    return {
        "id": f"evt-{event.id}",
        "status": event.status,
        "handled_by": event.handled_by,
    }


@router.get("/{event_id}/audit")
def get_event_audit(
    event_id: int,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    return [
        {
            "action": row.action,
            "actor": row.actor,
            "notes": row.notes,
            "timestamp": row.timestamp.isoformat() + "Z",
        }
        for row in db.query(EventAudit)
        .filter(EventAudit.event_id == event_id)
        .order_by(EventAudit.timestamp.desc())
        .all()
    ]


@router.get("/{event_id}/evidence/{kind}")
def get_event_evidence(
    event_id: int,
    kind: str,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    event = db.query(SecurityEvent).filter(SecurityEvent.id == event_id).first()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    relative_path = (
        event.snapshot_path
        if kind == "snapshot"
        else event.video_clip_path
        if kind == "clip"
        else None
    )
    if not relative_path:
        raise HTTPException(status_code=404, detail="Evidence is not ready")
    root = Path(__file__).resolve().parents[3]
    evidence_path = (root / relative_path).resolve()
    if root not in evidence_path.parents or not evidence_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found")
    return FileResponse(evidence_path, filename=evidence_path.name)
