from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional


class CameraBase(BaseModel):
    id: str
    name: str
    source: str
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: int = 30
    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    is_active: Optional[bool] = None
    status: Optional[str] = None


class CameraResponse(BaseModel):
    id: str
    name: str
    location: Optional[str] = None
    resolution: Optional[str] = None
    fps: int = 30
    is_active: bool = True
    status: str
    last_seen: Optional[datetime] = None
    detections: Optional[list] = []
    inference_fps: float = 0.0
    inference_latency_ms: float = 0.0
    ai_result_age_seconds: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)
