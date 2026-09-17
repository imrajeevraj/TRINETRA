"""
TRINETRA — Normalized PTZ Handover Mesh & Chain Database Models (Phase X)
PostgreSQL schema supporting multi-hop camera handovers, resource reservations, and terrain telemetry.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class PTZHandoverChainRecord(Base):
    """Corridor-wide sequence of camera handovers for a global entity."""
    __tablename__ = "ptz_handover_chains"

    chain_id = Column(String(32), primary_key=True, index=True) # e.g. "CHAIN-2026-000001"
    entity_id = Column(String(64), nullable=False, index=True)
    incident_id = Column(String(32), nullable=True, index=True)
    chain_status = Column(String(32), default="ACTIVE", nullable=False) # ACTIVE, COMPLETED, PAUSED, CANCELLED, LOOP_TERMINATED
    max_depth = Column(Integer, default=4, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    handovers = relationship("PTZHandoverRecord", back_populates="chain", cascade="all, delete-orphan")


class PTZHandoverRecord(Base):
    """Individual camera-to-camera handover hop."""
    __tablename__ = "ptz_handovers"

    handover_id = Column(String(32), primary_key=True, index=True) # e.g. "HO-2026-000001"
    chain_id = Column(String(32), ForeignKey("ptz_handover_chains.chain_id", ondelete="CASCADE"), nullable=False, index=True)
    hop_index = Column(Integer, nullable=False)                    # 1, 2, 3, 4
    source_camera = Column(String(32), nullable=False)
    target_camera = Column(String(32), nullable=False)
    prediction_id = Column(String(32), nullable=False)
    prediction_confidence = Column(Float, nullable=False)
    association_confidence = Column(Float, nullable=True)
    handover_confidence = Column(String(32), default="UNCONFIRMED", nullable=False) # CONFIRMED, PROBABLE, UNCONFIRMED
    state = Column(String(32), default="PREDICTED", nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expected_eta_sec = Column(Float, nullable=False)
    actual_arrival_sec = Column(Float, nullable=True)
    lead_time_sec = Column(Float, nullable=True)
    source_track_id = Column(String(64), nullable=True)
    target_track_id = Column(String(64), nullable=True)
    ptz_action_id = Column(String(32), nullable=True)
    reservation_id = Column(String(32), nullable=True)
    reason = Column(String(256), nullable=True)

    chain = relationship("PTZHandoverChainRecord", back_populates="handovers")


class CameraReservationRecord(Base):
    """Camera asset reservation record."""
    __tablename__ = "camera_reservations"

    reservation_id = Column(String(32), primary_key=True, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    entity_id = Column(String(64), nullable=False)
    priority_score = Column(Float, nullable=False)
    prediction_id = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    status = Column(String(32), default="ACTIVE", nullable=False) # ACTIVE, PREEMPTED, EXPIRED, RELEASED
    reason = Column(String(128), nullable=True)


class TerrainForecastRecord(Base):
    """Terrain-adjusted kinematic telemetry."""
    __tablename__ = "terrain_forecasts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    entity_id = Column(String(64), nullable=False, index=True)
    source_camera = Column(String(32), nullable=False)
    target_camera = Column(String(32), nullable=False)
    base_eta_sec = Column(Float, nullable=False)
    adjusted_eta_sec = Column(Float, nullable=False)
    slope_deg = Column(Float, default=0.0)
    surface_type = Column(String(32), default="UNKNOWN")
    terrain_mode = Column(String(32), default="DISABLED") # DISABLED, SIMULATED
