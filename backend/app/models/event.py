from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Boolean
from datetime import datetime
from backend.app.core.database import Base

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    Vector = String  # fallback for sqlite if used


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    person_name = Column(String, nullable=True, index=True)
    embedding = Column(Vector(512), nullable=False)
    watchlist_status = Column(
        String, default="UNKNOWN"
    )  # UNKNOWN, AUTHORIZED, WATCHLIST
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Track(Base):
    __tablename__ = "tracks"

    id = Column(String, primary_key=True, index=True)  # e.g. "P-024" or "V-102"
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    object_type = Column(String, nullable=False)  # person, car, motorcycle, etc.
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    direction = Column(String, nullable=True)  # BORDERWARD, INWARD, OUTWARD, etc.
    approx_speed = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(String, nullable=True, index=True)
    object_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class PlateEvent(Base):
    __tablename__ = "plate_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(String, nullable=True, index=True)
    source_frame_id = Column(String, nullable=True)
    plate_number = Column(String, nullable=False)
    raw_candidates = Column(String, nullable=True)  # JSON serialized OCR results
    confidence = Column(Float, nullable=False)
    validation_state = Column(String, nullable=True)
    vehicle_bbox_x1 = Column(Integer, nullable=True)
    vehicle_bbox_y1 = Column(Integer, nullable=True)
    vehicle_bbox_x2 = Column(Integer, nullable=True)
    vehicle_bbox_y2 = Column(Integer, nullable=True)
    plate_bbox_x1 = Column(Integer, nullable=True)
    plate_bbox_y1 = Column(Integer, nullable=True)
    plate_bbox_x2 = Column(Integer, nullable=True)
    plate_bbox_y2 = Column(Integer, nullable=True)
    snapshot_path = Column(String, nullable=True)
    watchlist_status = Column(
        String, default="CLEAN"
    )  # CLEAN, WATCHLIST, NEEDS_VERIFICATION
    data_origin = Column(
        String, nullable=False, default="LIVE", index=True
    )  # LIVE, DEMO, TEST, IMPORTED
    timestamp = Column(DateTime, default=datetime.utcnow)


class FaceEvent(Base):
    __tablename__ = "face_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    track_id = Column(String, nullable=True, index=True)
    frame_id = Column(String, nullable=True)
    person_name = Column(String, nullable=True)  # Known name from watchlist database
    confidence = Column(Float, nullable=True)
    face_quality = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)
    face_bbox_x1 = Column(Integer, nullable=True)
    face_bbox_y1 = Column(Integer, nullable=True)
    face_bbox_x2 = Column(Integer, nullable=True)
    face_bbox_y2 = Column(Integer, nullable=True)
    snapshot_path = Column(String, nullable=True)
    watchlist_status = Column(
        String, default="UNKNOWN"
    )  # UNKNOWN, AUTHORIZED, WATCHLIST
    data_origin = Column(String, nullable=False, default="LIVE", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class WatchlistEntry(Base):
    __tablename__ = "watchlist_entries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    type = Column(String, nullable=False)  # PLATE, FACE
    value = Column(
        String, nullable=False, unique=True
    )  # Plate Number or Face Embedding representation/ID
    name = Column(String, nullable=True)  # Subject name for alerts
    risk_level = Column(String, default="HIGH")  # MEDIUM, HIGH, CRITICAL
    created_at = Column(DateTime, default=datetime.utcnow)


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(
        String, nullable=False
    )  # ZONE_ENTRY, ZONE_EXIT, VIRTUAL_FENCE_CROSSING
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    zone_id = Column(String, nullable=True)  # ID of the zone or fence
    track_id = Column(String, nullable=True)
    object_type = Column(String, nullable=False)  # PERSON, CAR, etc.
    timestamp = Column(DateTime, default=datetime.utcnow)
    direction = Column(String, nullable=True)  # INWARD, OUTWARD, UNKNOWN
    x = Column(Integer, nullable=True)
    y = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    risk_score = Column(Integer, nullable=True)
    severity = Column(String, nullable=True)
    risk_reasons = Column(String, nullable=True)  # JSON serialized list of reasons
    status = Column(String, nullable=False, default="NEW")
    handled_by = Column(String, nullable=True)
    handled_at = Column(DateTime, nullable=True)
    operator_notes = Column(String, nullable=True)
    snapshot_path = Column(String, nullable=True)
    video_clip_path = Column(String, nullable=True)
    data_origin = Column(
        String, nullable=False, default="LIVE", index=True
    )  # LIVE, DEMO, TEST, IMPORTED


class EventAudit(Base):
    """Immutable operator-action record for a security event."""

    __tablename__ = "event_audits"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey("security_events.id"), nullable=False, index=True
    )
    action = Column(String, nullable=False)
    actor = Column(String, nullable=False)
    notes = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class Evidence(Base):
    """Cryptographically verifiable evidence linking media to security events."""

    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey("security_events.id"), nullable=False, index=True
    )
    camera_id = Column(String, ForeignKey("cameras.id"), nullable=False)
    capture_timestamp = Column(DateTime, default=datetime.utcnow)
    data_origin = Column(
        String, nullable=False, default="LIVE", index=True
    )  # LIVE, DEMO, TEST, IMPORTED
    integrity_hash = Column(String, nullable=False)  # SHA-256
    file_path = Column(String, nullable=False)


class AuditJob(Base):
    """R-05: Persistent execution state for background forensic audits and compliance runs."""

    __tablename__ = "audit_jobs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_type = Column(
        String, nullable=False, index=True
    )  # FORENSIC_AUDIT, EVIDENCE_RECONCILIATION, BACKUP
    status = Column(
        String, nullable=False, default="PENDING", index=True
    )  # PENDING, RUNNING, COMPLETED, FAILED
    triggered_by = Column(String, nullable=False)
    summary = Column(String, nullable=True)  # JSON summary of execution metrics
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    data_origin = Column(String, nullable=False, default="LIVE", index=True)
