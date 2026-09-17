from sqlalchemy import Column, String, Integer, Boolean, DateTime
from datetime import datetime
from backend.app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    source = Column(String, nullable=False)
    location = Column(String, nullable=True)
    resolution = Column(String, nullable=True)
    fps = Column(Integer, default=30)
    is_active = Column(Boolean, default=True)
    status = Column(String, default="OFFLINE")  # ONLINE, OFFLINE, DEGRADED, NO_SIGNAL
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
