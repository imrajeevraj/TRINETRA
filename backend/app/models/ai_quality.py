"""
SQLAlchemy database entities for Phase XIII AI Quality Intelligence & Feedback Governance.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from backend.app.core.database import Base


class OperatorFeedbackEntity(Base):
    __tablename__ = "governed_operator_feedback"

    id = Column(String(64), primary_key=True, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    frame_id = Column(String(64), nullable=False)
    event_id = Column(String(64), nullable=True)
    track_id = Column(Integer, nullable=True)
    global_entity_id = Column(String(64), nullable=True)
    model_id = Column(String(64), nullable=False)
    model_version = Column(String(32), nullable=False)
    operator_id = Column(String(64), nullable=False)
    disposition = Column(String(32), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)
    evidence_reference = Column(String(256), nullable=True)
    source_frame_hash = Column(String(64), nullable=False, index=True)
    review_status = Column(String(32), nullable=False, default="REVIEW_REQUIRED")
    validated_class = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FeedbackReviewAuditEntity(Base):
    __tablename__ = "governed_feedback_reviews"

    id = Column(String(64), primary_key=True, index=True)
    feedback_id = Column(String(64), ForeignKey("governed_operator_feedback.id"), nullable=False)
    reviewer_id = Column(String(64), nullable=False)
    reviewer_role = Column(String(32), nullable=False)
    agrees_with_operator = Column(Boolean, nullable=False)
    assigned_disposition = Column(String(32), nullable=False)
    corrected_class = Column(String(64), nullable=True)
    notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), server_default=func.now())


class HardCaseEntity(Base):
    __tablename__ = "governed_hard_cases"

    id = Column(String(64), primary_key=True, index=True)
    camera_id = Column(String(32), nullable=False, index=True)
    frame_id = Column(String(64), nullable=False)
    model_id = Column(String(64), nullable=False)
    trigger_type = Column(String(64), nullable=False, index=True)
    hard_case_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box_json = Column(JSON, nullable=True)
    scene_metadata_json = Column(JSON, nullable=True)
    source_frame_hash = Column(String(64), nullable=False)
    review_status = Column(String(32), nullable=False, default="PENDING_REVIEW")
    discovered_at = Column(DateTime(timezone=True), server_default=func.now())


class FailureClusterEntity(Base):
    __tablename__ = "governed_failure_clusters"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    domain = Column(String(32), nullable=False, index=True)
    failure_type = Column(String(64), nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    severity = Column(String(32), nullable=False, default="MEDIUM")
    status = Column(String(32), nullable=False, default="OPEN")
    target_dataset_version = Column(String(64), nullable=True)
    curation_notes = Column(Text, nullable=True)
    last_seen = Column(DateTime(timezone=True), server_default=func.now())


class CameraQualityEntity(Base):
    __tablename__ = "governed_camera_quality"

    id = Column(String(32), primary_key=True)  # camera_id
    alert_frequency_per_hour = Column(Float, nullable=False, default=0.0)
    false_positive_rate = Column(Float, nullable=False, default=0.0)
    false_negative_count = Column(Integer, nullable=False, default=0)
    mean_confidence = Column(Float, nullable=False, default=0.80)
    tracking_instability_count = Column(Integer, nullable=False, default=0)
    ptz_cue_success_rate = Column(Float, nullable=False, default=1.00)
    overall_health = Column(String(32), nullable=False, default="HEALTHY")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PoisoningAlertEntity(Base):
    __tablename__ = "governed_poisoning_alerts"

    id = Column(String(64), primary_key=True, index=True)
    vector_type = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False)
    sample_ids_json = Column(JSON, nullable=False)
    details = Column(Text, nullable=False)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
