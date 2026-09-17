"""
TRINETRA Phase XIV — Multimodal Sensor Intelligence Database Models
Persists sensors, observations, calibrations, synchronization records,
thermal datasets, cross-spectral associations, multimodal tracks, and sensor health.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Index,
)
from backend.app.core.database import Base


class SensorModality(str, Enum):
    OPTICAL_RGB = "OPTICAL_RGB"
    THERMAL_LWIR = "THERMAL_LWIR"
    DEPTH = "DEPTH"
    RADAR = "RADAR"
    OTHER = "OTHER"


class DataOrigin(str, Enum):
    REAL_SENSOR = "REAL_SENSOR"
    RECORDED_REAL_SENSOR = "RECORDED_REAL_SENSOR"
    SIMULATED = "SIMULATED"
    SYNTHETIC = "SYNTHETIC"


class CalibrationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    UNVALIDATED = "UNVALIDATED"
    MISSING = "MISSING"
    EXPIRED = "EXPIRED"


class SynchronizationStatus(str, Enum):
    SYNCED = "SYNCED"
    PARTIALLY_SYNCED = "PARTIALLY_SYNCED"
    UNSYNCED = "UNSYNCED"
    STALE = "STALE"


class RegistrationMode(str, Enum):
    GEOMETRIC_REGISTERED = "GEOMETRIC_REGISTERED"
    TEMPORAL_ONLY = "TEMPORAL_ONLY"
    SIMULATED = "SIMULATED"
    DISABLED = "DISABLED"


class SensorHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    SYNC_DEGRADED = "SYNC_DEGRADED"


class SensorRecord(Base):
    __tablename__ = "sensors"

    id = Column(String(64), primary_key=True, index=True)
    camera_id = Column(String(32), index=True, nullable=False)
    modality = Column(String(32), nullable=False)
    resolution_w = Column(Integer, default=1920)
    resolution_h = Column(Integer, default=1080)
    fps = Column(Integer, default=30)
    position_status = Column(String(64), default="LOGICAL / UNKNOWN")
    calibration_id = Column(String(64), nullable=True)
    sync_status = Column(String(32), default=SynchronizationStatus.SYNCED.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SensorObservationRecord(Base):
    __tablename__ = "sensor_observations"

    id = Column(String(64), primary_key=True, default=lambda: f"OBS-{uuid.uuid4().hex[:12]}")
    sensor_id = Column(String(64), index=True, nullable=False)
    camera_id = Column(String(32), index=True, nullable=False)
    modality = Column(String(32), nullable=False)
    timestamp = Column(Float, nullable=False)
    ingestion_timestamp = Column(Float, nullable=False)
    frame_id = Column(String(64), index=True, nullable=False)
    sequence_number = Column(Integer, nullable=False)
    frame_hash = Column(String(64), nullable=False)
    data_origin = Column(String(32), default=DataOrigin.SIMULATED.value)
    calibration_id = Column(String(64), nullable=True)
    capture_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_obs_sensor_time", "sensor_id", "timestamp"),
    )


class SensorCalibrationRecord(Base):
    __tablename__ = "sensor_calibrations"

    id = Column(String(64), primary_key=True, default=lambda: f"CAL-{uuid.uuid4().hex[:12]}")
    sensor_id = Column(String(64), index=True, nullable=False)
    modality = Column(String(32), nullable=False)
    status = Column(String(32), default=CalibrationStatus.UNVALIDATED.value)
    intrinsics = Column(JSON, default=dict)
    distortion = Column(JSON, default=dict)
    extrinsics = Column(JSON, default=dict)
    rmse_px = Column(Float, default=0.0)
    valid_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SensorSynchronizationRecord(Base):
    __tablename__ = "sensor_synchronizations"

    id = Column(String(64), primary_key=True, default=lambda: f"SYNC-{uuid.uuid4().hex[:12]}")
    optical_sensor_id = Column(String(64), index=True, nullable=False)
    thermal_sensor_id = Column(String(64), index=True, nullable=False)
    time_delta_ms = Column(Float, default=0.0)
    status = Column(String(32), default=SynchronizationStatus.SYNCED.value)
    clock_source = Column(String(32), default="PTP_IEEE_1588")
    evaluated_at = Column(DateTime, default=datetime.utcnow)


class ThermalDatasetRecord(Base):
    __tablename__ = "thermal_datasets"

    id = Column(String(64), primary_key=True)
    version = Column(String(32), default="v0.1-spec")
    modalities = Column(JSON, default=lambda: ["THERMAL_LWIR"])
    classes = Column(JSON, default=lambda: ["person", "vehicle"])
    label_status = Column(String(32), default="UNLABELED / PENDING_COLLECTION")
    data_origin = Column(String(32), default=DataOrigin.SIMULATED.value)
    frame_count = Column(Integer, default=0)
    benchmark_status = Column(String(32), default="QUARANTINED_BENCHMARK_PROTECTED")
    created_at = Column(DateTime, default=datetime.utcnow)


class MultimodalSpectralAssociationEntity(Base):
    __tablename__ = "multimodal_spectral_associations"

    id = Column(String(64), primary_key=True, default=lambda: f"MSA-{uuid.uuid4().hex[:12]}")
    camera_id = Column(String(32), index=True, nullable=False)
    rgb_observation_id = Column(String(64), index=True, nullable=False)
    thermal_observation_id = Column(String(64), index=True, nullable=False)
    registration_mode = Column(String(32), default=RegistrationMode.SIMULATED.value)
    optical_confidence = Column(Float, default=0.0)
    thermal_confidence = Column(Float, default=0.0)
    association_confidence = Column(Float, default=0.0)
    fusion_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class MultimodalTrackEntity(Base):
    __tablename__ = "multimodal_tracks"

    id = Column(String(64), primary_key=True, default=lambda: f"MMTRK-{uuid.uuid4().hex[:12]}")
    global_entity_id = Column(String(64), index=True, nullable=False)
    camera_id = Column(String(32), index=True, nullable=False)
    optical_track_id = Column(String(64), nullable=True)
    thermal_track_id = Column(String(64), nullable=True)
    modality_lineage = Column(JSON, default=dict)
    status = Column(String(32), default="ACTIVE")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SensorHealthEntity(Base):
    __tablename__ = "sensor_health"

    id = Column(String(64), primary_key=True, default=lambda: f"SH-{uuid.uuid4().hex[:12]}")
    sensor_id = Column(String(64), index=True, unique=True, nullable=False)
    status = Column(String(32), default=SensorHealthStatus.HEALTHY.value)
    fps_measured = Column(Float, default=30.0)
    latency_ms = Column(Float, default=12.5)
    frame_drop_rate = Column(Float, default=0.0)
    drift_detected = Column(Boolean, default=False)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)


class ThermalDatasetSampleEntity(Base):
    __tablename__ = "thermal_dataset_samples"

    id = Column(String(64), primary_key=True, default=lambda: f"SMP-{uuid.uuid4().hex[:12]}")
    dataset_id = Column(String(64), index=True, nullable=False)
    sample_id = Column(String(64), index=True, unique=True, nullable=False)
    source_sensor = Column(String(64), nullable=False)
    source_frame_id = Column(String(64), nullable=False)
    capture_timestamp = Column(Float, nullable=False)
    data_origin = Column(String(32), default=DataOrigin.SIMULATED.value)
    calibration_version = Column(String(32), nullable=True)
    annotation_version = Column(String(32), nullable=True)
    quality_status = Column(String(32), default="UNVALIDATED")
    sha256_hash = Column(String(64), index=True, nullable=False)
    split_assignment = Column(String(32), default="TRAIN")  # TRAIN, VAL, TEST, BENCHMARK_QUARANTINED
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class ThermalAnnotationEntity(Base):
    __tablename__ = "thermal_annotations"

    id = Column(String(64), primary_key=True, default=lambda: f"ANN-{uuid.uuid4().hex[:12]}")
    sample_id = Column(String(64), index=True, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    class_name = Column(String(64), nullable=False)
    occlusion_level = Column(Float, default=0.0)
    truncation_level = Column(Float, default=0.0)
    difficulty = Column(String(32), default="NORMAL")
    annotation_source = Column(String(64), default="HUMAN_VERIFIED")
    reviewer = Column(String(64), nullable=True)
    confidence = Column(Float, default=1.0)
    annotation_version = Column(String(32), default="v1.0")
    approval_status = Column(String(32), default="APPROVED")
    created_at = Column(DateTime, default=datetime.utcnow)


class ThermalEdgeDeploymentPackageEntity(Base):
    __tablename__ = "thermal_edge_packages"

    id = Column(String(64), primary_key=True, default=lambda: f"PKG-THM-{uuid.uuid4().hex[:12]}")
    package_id = Column(String(64), index=True, unique=True, nullable=False)
    model_name = Column(String(128), nullable=False)
    architecture = Column(String(32), nullable=False)  # YOLO11, YOLO26
    model_sha256 = Column(String(64), nullable=False)
    dataset_version = Column(String(32), nullable=False)
    benchmark_version = Column(String(32), nullable=False)
    calibration_dependency = Column(String(64), nullable=True)
    sensor_compatibility = Column(JSON, default=list)
    preprocessing_config = Column(JSON, default=dict)
    runtime_version = Column(String(32), default="ONNX_RUNTIME_1.17")
    rollback_target_sha256 = Column(String(64), nullable=True)
    package_status = Column(String(32), default="PACKAGED")
    created_at = Column(DateTime, default=datetime.utcnow)
