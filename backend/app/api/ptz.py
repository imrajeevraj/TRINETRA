"""
TRINETRA — PTZ Control & Human Review API Router (Phase VII)
Exposes authenticated endpoints for PTZ status, slew-to-cue, manual control,
zoom inspection, and human operator alert dispositions.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from backend.app.core.security import require_roles, get_current_user
from backend.app.models.user import User
from backend.app.services.ptz_cue_engine import ptz_cue_engine
from backend.app.services.temporal_threat_confirmation import temporal_threat_engine
from backend.app.core.events_pubsub import publish_event

router = APIRouter(prefix="/ptz", tags=["ptz"])


class ManualPTZCommand(BaseModel):
    pan: float = Field(..., ge=-180.0, le=180.0, description="Target Pan in degrees (-180 to 180)")
    tilt: float = Field(..., ge=-90.0, le=90.0, description="Target Tilt in degrees (-90 to 90)")
    zoom: float = Field(..., ge=1.0, le=30.0, description="Target Zoom level (1.0x to 30.0x)")
    speed: float = Field(default=1.0, ge=0.1, le=2.0, description="Movement speed multiplier")


class CueTrackRequest(BaseModel):
    track_id: str = Field(..., description="Active track ID to cue PTZ towards")
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="Bounding box [x1, y1, x2, y2]")
    reason: Optional[str] = Field(default="Manual Operator Cue", max_length=256)


class HumanDispositionRequest(BaseModel):
    track_id: str = Field(..., description="Track ID to record disposition for")
    disposition: str = Field(..., pattern="^(CONFIRMED|REJECTED|REQUEST_ZOOM)$")
    operator_notes: Optional[str] = Field(default=None, max_length=1000)


@router.get("/{camera_id}/status")
def get_ptz_status(
    camera_id: str,
    user: User = Depends(require_roles("ADMIN", "OPERATOR", "VIEWER"))
):
    """Retrieves current Pan, Tilt, Zoom, and stabilization state for camera."""
    ctrl = ptz_cue_engine.get_or_create_controller(camera_id)
    status_data = ctrl.get_status()
    active_target = ptz_cue_engine.active_targets.get(camera_id)

    return {
        "status": status_data,
        "active_target": {
            "track_id": active_target.track_id,
            "class_name": active_target.class_name,
            "risk_score": active_target.risk_score,
            "priority": active_target.priority.value,
            "cue_score": active_target.cue_score
        } if active_target else None
    }


@router.post("/{camera_id}/command")
def manual_ptz_command(
    camera_id: str,
    cmd: ManualPTZCommand,
    user: User = Depends(require_roles("ADMIN", "OPERATOR"))
):
    """Privileged manual PTZ slew control. Requires ADMIN or OPERATOR role."""
    ctrl = ptz_cue_engine.get_or_create_controller(camera_id)
    success = ctrl.slew_to(pan=cmd.pan, tilt=cmd.tilt, zoom=cmd.zoom, speed=cmd.speed)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera {camera_id} does not support PTZ commands (Hardware mode: Disabled)."
        )

    publish_event("ptz.command_issued", {
        "camera_id": camera_id,
        "operator": user.username,
        "pan": cmd.pan,
        "tilt": cmd.tilt,
        "zoom": cmd.zoom
    })
    return {"status": "SUCCESS", "message": f"PTZ command dispatched for {camera_id}"}


@router.post("/{camera_id}/cue")
def cue_ptz_to_track(
    camera_id: str,
    req: CueTrackRequest,
    user: User = Depends(require_roles("ADMIN", "OPERATOR"))
):
    """Directs PTZ camera to center and zoom in on specific bounding box."""
    ctrl = ptz_cue_engine.get_or_create_controller(camera_id)
    pan, tilt, zoom = ptz_cue_engine.bbox_to_ptz(req.bbox)
    success = ctrl.slew_to(pan=pan, tilt=tilt, zoom=zoom)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera {camera_id} is a fixed camera without PTZ."
        )

    publish_event("ptz.cue_requested", {
        "camera_id": camera_id,
        "track_id": req.track_id,
        "operator": user.username,
        "pan": pan,
        "tilt": tilt,
        "zoom": zoom,
        "reason": req.reason
    })
    return {"status": "SUCCESS", "cued_position": {"pan": pan, "tilt": tilt, "zoom": zoom}}


@router.post("/{camera_id}/home")
def return_ptz_home(
    camera_id: str,
    user: User = Depends(require_roles("ADMIN", "OPERATOR"))
):
    """Returns PTZ camera to default home wide-angle orientation."""
    ctrl = ptz_cue_engine.get_or_create_controller(camera_id)
    ctrl.home()
    ptz_cue_engine.active_targets.pop(camera_id, None)

    publish_event("ptz.home_command", {"camera_id": camera_id, "operator": user.username})
    return {"status": "SUCCESS", "message": f"Camera {camera_id} returning to home preset."}


@router.post("/{camera_id}/disposition")
def record_operator_disposition(
    camera_id: str,
    req: HumanDispositionRequest,
    user: User = Depends(require_roles("ADMIN", "OPERATOR"))
):
    """Records human-in-the-loop disposition for an active threat track."""
    rec = temporal_threat_engine.record_human_disposition(
        camera_id=camera_id,
        track_id=req.track_id,
        disposition=req.disposition,
        notes=f"Reviewed by {user.username}: {req.operator_notes or 'No notes'}"
    )
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Active track {req.track_id} on camera {camera_id} not found."
        )

    publish_event("alert.disposition_recorded", {
        "camera_id": camera_id,
        "track_id": req.track_id,
        "disposition": req.disposition,
        "operator": user.username,
        "review_status": rec.review_status.value
    })
    return {"status": "SUCCESS", "review_status": rec.review_status.value}
