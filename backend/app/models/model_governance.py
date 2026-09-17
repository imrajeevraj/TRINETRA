"""
SQLAlchemy database entities for Phase XII Model Lifecycle & Governance.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from backend.app.core.database import Base


class ModelRecord(Base):
    __tablename__ = "governed_models"

    id = Column(String(64), primary_key=True, index=True)
    model_name = Column(String(128), nullable=False)
    domain = Column(String(32), nullable=False, index=True)  # GROUND, AIRBORNE, SECURITY_ITEM
    active_version = Column(String(32), nullable=False)
    architecture = Column(String(64), nullable=False)
    governance_status = Column(String(32), nullable=False, default="PRODUCTION")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ModelVersionRecord(Base):
    __tablename__ = "governed_model_versions"

    id = Column(String(64), primary_key=True, index=True)
    model_id = Column(String(64), ForeignKey("governed_models.id"), nullable=False)
    version = Column(String(32), nullable=False)
    sha256 = Column(String(64), nullable=False, index=True)
    file_path = Column(String(256), nullable=False)
    parent_version_id = Column(String(64), nullable=True)
    dataset_id = Column(String(64), nullable=True)
    training_run_id = Column(String(64), nullable=True)
    lifecycle_state = Column(String(32), nullable=False, default="ACTIVE")
    rollback_target = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DatasetRecord(Base):
    __tablename__ = "governed_datasets"

    id = Column(String(64), primary_key=True, index=True)
    dataset_version = Column(String(32), nullable=False)
    domain = Column(String(32), nullable=False)
    sample_count = Column(Integer, nullable=False, default=0)
    manifest_hash = Column(String(64), nullable=False)
    quality_status = Column(String(32), nullable=False, default="PASSED")
    benchmark_isolated = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TrainingRunRecord(Base):
    __tablename__ = "governed_training_runs"

    id = Column(String(64), primary_key=True, index=True)
    experiment_name = Column(String(128), nullable=False)
    domain = Column(String(32), nullable=False)
    architecture = Column(String(64), nullable=False)
    artifact_sha256 = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="COMPLETED")
    metrics_json = Column(JSON, nullable=True)
    hyperparameters_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ValidationRunRecord(Base):
    __tablename__ = "governed_validation_runs"

    id = Column(String(64), primary_key=True, index=True)
    candidate_model_id = Column(String(64), nullable=False)
    baseline_model_id = Column(String(64), nullable=False)
    overall_verdict = Column(String(64), nullable=False)
    stages_summary_json = Column(JSON, nullable=True)
    rejection_reasons_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CanaryDeploymentRecord(Base):
    __tablename__ = "governed_canary_deployments"

    id = Column(String(64), primary_key=True, index=True)
    candidate_model_id = Column(String(64), nullable=False)
    production_model_id = Column(String(64), nullable=False)
    target_nodes = Column(JSON, nullable=False)
    traffic_percentage = Column(Float, nullable=False, default=10.0)
    is_shadow_mode = Column(Boolean, nullable=False, default=True)
    status = Column(String(32), nullable=False, default="RUNNING")
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)


class EdgeDeploymentRecord(Base):
    __tablename__ = "governed_edge_deployments"

    id = Column(String(64), primary_key=True, index=True)
    package_id = Column(String(64), nullable=False)
    camera_id = Column(String(32), nullable=False, index=True)
    active_model_id = Column(String(64), nullable=False)
    last_known_good_model_id = Column(String(64), nullable=False)
    deployment_status = Column(String(32), nullable=False, default="ACTIVE")
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelHealthRecord(Base):
    __tablename__ = "governed_model_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(32), nullable=False)
    inference_fps = Column(Float, nullable=False)
    p50_latency_ms = Column(Float, nullable=False)
    p95_latency_ms = Column(Float, nullable=False)
    gpu_memory_mb = Column(Float, nullable=False)
    error_rate = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False, default="HEALTHY")
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())


class RollbackEventRecord(Base):
    __tablename__ = "governed_rollback_events"

    id = Column(String(64), primary_key=True, index=True)
    domain = Column(String(32), nullable=False)
    failed_model_id = Column(String(64), nullable=False)
    restored_model_id = Column(String(64), nullable=False)
    restored_sha256 = Column(String(64), nullable=False)
    operator = Column(String(64), nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="SUCCESS")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PromotionDecisionRecord(Base):
    __tablename__ = "governed_promotion_decisions"

    id = Column(String(64), primary_key=True, index=True)
    model_id = Column(String(64), nullable=False)
    decision = Column(String(32), nullable=False)  # PROMOTE, REJECT, CANARY, ROLLBACK
    operator = Column(String(64), nullable=False)
    evidence_summary = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
