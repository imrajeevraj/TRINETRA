import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from backend.app.core.security import get_current_user, require_roles
from backend.app.models.user import User
from backend.app.services.audit_service import audit_service

router = APIRouter(prefix="/audit", tags=["audit"])

UPLOAD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/videos/uploads")
)


@router.post("/upload")
def upload_video_for_audit(
    file: UploadFile = File(...),
    user: User = Depends(require_roles("ADMIN", "OPERATOR")),
):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files are supported")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, f"{user.username}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = audit_service.start_audit(file_path, user.username)

    return {
        "job_id": job_id,
        "status": "PROCESSING",
        "message": "Video uploaded and audit started",
    }


@router.get("/jobs")
def get_audit_jobs(_user=Depends(get_current_user)):
    return audit_service.get_all_jobs()


@router.get("/jobs/{job_id}")
def get_audit_job_status(job_id: str, _user=Depends(get_current_user)):
    job = audit_service.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
