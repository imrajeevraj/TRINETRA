"""
TRINETRA — Edge Camera Mesh & Distributed Intelligence Database Models (Phase XI)
PostgreSQL schema supporting edge node state, leases, outbox buffering,
evidence sync audit, and cross-spectral associations.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text
from backend.app.core.database import Base


class EdgeNodeRecord(Base):
    """Persistent edge camera sentry node state."""
    __tablename__ = "edge_nodes"

    node_id = Column(String(32), primary_key=True, index=True) # e.g. "NODE-CAM-001"
    camera_id = Column(String(32), nullable=False, unique=True, index=True)
    health = Column(String(32), default="ONLINE", nullable=False) # ONLINE, DEGRADED, OFFLINE
    network_state = Column(String(32), default="NORMAL", nullable=False) # NORMAL, DEGRADED, PARTITIONED, RECOVERING
    clock_offset_ms = Column(Float, default=0.0, nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    last_sync_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EdgeLeaseRecord(Base):
    """Authorization lease records defining edge autonomy windows."""
    __tablename__ = "edge_leases"

    lease_id = Column(String(64), primary_key=True, index=True)
    node_id = Column(String(32), nullable=False, index=True)
    issued_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    policy_version = Column(String(32), default="2.0.0", nullable=False)
    allowed_operations = Column(String(256), default="PEER_HANDOVER,PTZ_PRECUE,LOCAL_TRACK", nullable=False)


class EdgeOutboxRecord(Base):
    """Durable local outbox events buffered during network partitions."""
    __tablename__ = "edge_outbox"

    event_id = Column(String(64), primary_key=True, index=True)
    node_id = Column(String(32), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    payload_json = Column(Text, nullable=False)
    status = Column(String(32), default="PENDING", nullable=False, index=True) # PENDING, SENT, ACKNOWLEDGED, FAILED
    attempt_count = Column(Integer, default=0, nullable=False)
    last_attempt = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CrossSpectralAssociationRecord(Base):
    """Historical association decisions between Optical and Thermal sentries."""
    __tablename__ = "cross_spectral_associations"

    association_id = Column(String(64), primary_key=True, index=True)
    source_camera = Column(String(32), nullable=False, index=True)
    source_spectrum = Column(String(16), nullable=False) # OPTICAL, THERMAL
    target_camera = Column(String(32), nullable=False, index=True)
    target_spectrum = Column(String(16), nullable=False)
    optical_confidence = Column(Float, nullable=False)
    thermal_confidence = Column(Float, nullable=False)
    motion_confidence = Column(Float, nullable=False)
    cross_spectral_confidence = Column(Float, nullable=False)
    confidence_tier = Column(String(32), nullable=False) # CONFIRMED, PROBABLE, UNCONFIRMED
    mode = Column(String(32), default="SIMULATED", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class EvidenceSyncRecord(Base):
    """Evidence synchronization audit logs on reconnect."""
    __tablename__ = "evidence_sync_records"

    sync_id = Column(String(64), primary_key=True, index=True)
    node_id = Column(String(32), nullable=False, index=True)
    event_count = Column(Integer, nullable=False)
    integrity_failures = Column(Integer, default=0, nullable=False)
    duration_sec = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
