from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List

from backend.app.core.database import get_db
from backend.app.core.security import require_roles
from backend.app.models.camera import Camera
from backend.app.models.user import User
from backend.app.schemas.camera import CameraResponse
from backend.app.services.camera_manager import camera_manager

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("/", response_model=List[CameraResponse])
def list_cameras(
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("ADMIN", "OPERATOR", "VIEWER")),
):
    cameras = db.query(Camera).filter(Camera.is_active.is_(True)).all()
    # Dynamic sync with live thread statuses
    for cam in cameras:
        status_info = camera_manager.get_camera_status(cam.id)
        cam.status = status_info["status"]
        cam.fps = int(status_info["fps"])
        cam.resolution = status_info["resolution"]
        cam.detections = status_info.get("detections", [])
    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("ADMIN", "OPERATOR", "VIEWER")),
):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    status_info = camera_manager.get_camera_status(camera_id)
    camera.status = status_info["status"]
    camera.fps = int(status_info["fps"])
    camera.resolution = status_info["resolution"]
    camera.detections = status_info.get("detections", [])
    return camera


@router.get("/{camera_id}/stream")
def stream_camera(
    camera_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("ADMIN", "OPERATOR", "VIEWER")),
):
    # Verify camera exists in database
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return StreamingResponse(
        camera_manager.generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/status")
def get_camera_status(
    camera_id: str,
    user: User = Depends(require_roles("ADMIN", "OPERATOR", "VIEWER")),
):
    return camera_manager.get_camera_status(camera_id)
