"""
TRINETRA — Normalized Incident, Global Entity & Evidence Graph Models (Phase VIII)
PostgreSQL schema supporting multi-camera incident correlation and verifiable evidence linkage.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class Incident(Base):
    """
    Unified incident abstraction consolidating multi-camera,
    multi-sensor observations into one cohesive investigation.
    """
    __tablename__ = "incidents"

    incident_id = Column(String(32), primary_key=True, index=True) # e.g. "INC-2026-000001"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    status = Column(String(32), default="OPEN", index=True, nullable=False) # OPEN, INVESTIGATING, ESCALATED, CONFIRMED, RESOLVED, FALSE_POSITIVE, ARCHIVED
    severity = Column(String(16), default="LOW", index=True, nullable=False) # INFO, LOW, MEDIUM, HIGH, CRITICAL
    risk_score = Column(Integer, default=0, nullable=False)
    summary = Column(Text, nullable=True)
    primary_zone = Column(String(64), nullable=True)
    primary_camera = Column(String(32), nullable=True)
    data_origin = Column(String(16), default="LIVE", index=True, nullable=False) # LIVE, DEMO, TEST, IMPORTED

    events = relationship("IncidentEvent", back_populates="incident", cascade="all, delete-orphan")
    evidence_links = relationship("IncidentEvidenceLink", back_populates="incident", cascade="all, delete-orphan")
    dispositions = relationship("IncidentDisposition", back_populates="incident", cascade="all, delete-orphan")


class IncidentEvent(Base):
    """Junction linking SecurityEvent records to an Incident."""
    __tablename__ = "incident_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    incident_id = Column(String(32), ForeignKey("incidents.incident_id", ondelete="CASCADE"), nullable=False, index=True)
    event_id = Column(Integer, nullable=True)
    camera_id = Column(String(32), nullable=False)
    track_id = Column(String(64), nullable=True)
    global_id = Column(String(64), nullable=True, index=True)
    event_type = Column(String(64), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    details_json = Column(Text, nullable=True)

    incident = relationship("Incident", back_populates="events")


class GlobalEntityRecord(Base):
    """Persistent representation of a cross-camera correlated entity."""
    __tablename__ = "global_entities"

    global_id = Column(String(64), primary_key=True, index=True) # e.g. "GLOBAL-PERSON-00001"
    entity_type = Column(String(32), nullable=False)             # "person", "vehicle", etc.
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    confidence = Column(Float, default=0.5, nullable=False)
    primary_plate = Column(String(32), nullable=True)
    primary_face_id = Column(String(64), nullable=True)
    status = Column(String(32), default="ACTIVE", nullable=False)

    observations = relationship("EntityObservationRecord", back_populates="entity", cascade="all, delete-orphan")


class EntityObservationRecord(Base):
    """Individual camera track observations tied to a GlobalEntity."""
    __tablename__ = "entity_observations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    global_id = Column(String(64), ForeignKey("global_entities.global_id", ondelete="CASCADE"), nullable=False, index=True)
    camera_id = Column(String(32), nullable=False)
    track_id = Column(String(64), nullable=False) # Local track ID e.g. "P-024"
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    bbox_json = Column(String(128), nullable=True)
    confidence = Column(Float, default=0.5, nullable=False)
    direction = Column(String(32), nullable=True)
    sensor_type = Column(String(32), default="GROUND_CAMERA", nullable=False)

    entity = relationship("GlobalEntityRecord", back_populates="observations")


class IncidentEvidenceLink(Base):
    """Verifiable evidence media linked to an Incident with SHA-256 integrity."""
    __tablename__ = "incident_evidence_links"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    incident_id = Column(String(32), ForeignKey("incidents.incident_id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_type = Column(String(32), nullable=False) # SNAPSHOT, CROP, VIDEO_CLIP, SAHI_TILE
    file_path = Column(String(256), nullable=False)
    integrity_hash = Column(String(64), nullable=False) # SHA-256
    camera_id = Column(String(32), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    incident = relationship("Incident", back_populates="evidence_links")


class IncidentDisposition(Base):
    """Immutable record of operator actions and review dispositions for an incident."""
    __tablename__ = "incident_dispositions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    incident_id = Column(String(32), ForeignKey("incidents.incident_id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(32), nullable=False) # ACKNOWLEDGE, CONFIRM, REJECT, ESCALATE, RESOLVE
    actor = Column(String(64), nullable=False)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    incident = relationship("Incident", back_populates="dispositions")
