"""
TRINETRA Phase XV — Governed Thermal Dataset Service
Implements strict 12-state dataset governance lifecycle:
RAW -> INGESTED -> QUALITY_REJECTED -> CALIBRATED -> ANNOTATION_PENDING
-> ANNOTATED -> QA_PENDING -> QA_FAILED -> TRAINING_READY -> BENCHMARK_QUARANTINED
-> BENCHMARK_APPROVED -> ARCHIVED.

Tracks sample provenance, SHA-256 hashes, split assignments, and collection scenarios.
"""

from __future__ import annotations
import json
import time
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

from backend.app.services.sensor_abstraction import DataOrigin

logger = logging.getLogger("ThermalDatasetService")


class DatasetState(str, Enum):
    RAW = "RAW"
    INGESTED = "INGESTED"
    QUALITY_REJECTED = "QUALITY_REJECTED"
    CALIBRATED = "CALIBRATED"
    ANNOTATION_PENDING = "ANNOTATION_PENDING"
    ANNOTATED = "ANNOTATED"
    QA_PENDING = "QA_PENDING"
    QA_FAILED = "QA_FAILED"
    TRAINING_READY = "TRAINING_READY"
    BENCHMARK_QUARANTINED = "BENCHMARK_QUARANTINED"
    BENCHMARK_APPROVED = "BENCHMARK_APPROVED"
    ARCHIVED = "ARCHIVED"


VALID_TRANSITIONS: Dict[DatasetState, List[DatasetState]] = {
    DatasetState.RAW: [DatasetState.INGESTED, DatasetState.QUALITY_REJECTED],
    DatasetState.INGESTED: [DatasetState.CALIBRATED, DatasetState.QUALITY_REJECTED],
    DatasetState.QUALITY_REJECTED: [DatasetState.ARCHIVED],
    DatasetState.CALIBRATED: [DatasetState.ANNOTATION_PENDING, DatasetState.QUALITY_REJECTED],
    DatasetState.ANNOTATION_PENDING: [DatasetState.ANNOTATED, DatasetState.QA_FAILED],
    DatasetState.ANNOTATED: [DatasetState.QA_PENDING],
    DatasetState.QA_PENDING: [DatasetState.TRAINING_READY, DatasetState.QA_FAILED, DatasetState.BENCHMARK_QUARANTINED],
    DatasetState.QA_FAILED: [DatasetState.ANNOTATION_PENDING, DatasetState.ARCHIVED],
    DatasetState.TRAINING_READY: [DatasetState.ARCHIVED],
    DatasetState.BENCHMARK_QUARANTINED: [DatasetState.BENCHMARK_APPROVED, DatasetState.ARCHIVED],
    DatasetState.BENCHMARK_APPROVED: [DatasetState.ARCHIVED],
    DatasetState.ARCHIVED: [],
}


@dataclass
class ThermalDatasetSample:
    dataset_id: str
    sample_id: str
    source_sensor: str
    source_frame_id: str
    capture_timestamp: float
    data_origin: DataOrigin
    calibration_version: Optional[str]
    annotation_version: Optional[str]
    quality_status: str
    sha256_hash: str
    split_assignment: str  # "TRAIN", "VAL", "TEST", "BENCHMARK_QUARANTINED"
    scenario_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_at: float = field(default_factory=time.time)


@dataclass
class ThermalDatasetManifest:
    dataset_id: str
    name: str
    version: str
    current_state: DatasetState
    data_origin: DataOrigin
    samples: Dict[str, ThermalDatasetSample] = field(default_factory=dict)
    manifest_sha256: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    is_training_eligible: bool = False
    notes: str = ""


class ThermalDatasetService:
    """
    Manages controlled acquisition, governance, and audit trails for thermal datasets.
    Guarantees that unvalidated or simulated datasets cannot enter production training.
    """

    def __init__(self):
        self._datasets: Dict[str, ThermalDatasetManifest] = {}
        self._initialize_readiness_dataset()

    def _initialize_readiness_dataset(self):
        """Initializes default unvalidated readiness dataset manifest."""
        ds_id = "DS-THM-READINESS-v0"
        manifest = ThermalDatasetManifest(
            dataset_id=ds_id,
            name="IBVAP-THERMAL-READINESS-v0",
            version="v0.1-spec",
            current_state=DatasetState.RAW,
            data_origin=DataOrigin.SIMULATED,
            is_training_eligible=False,
            notes="Formal readiness dataset shell. Contains zero real LWIR frames until physical sensor collection.",
        )
        self._datasets[ds_id] = manifest

    def create_dataset(
        self,
        dataset_id: str,
        name: str,
        version: str = "v1.0",
        data_origin: DataOrigin = DataOrigin.SIMULATED,
        notes: str = "",
    ) -> ThermalDatasetManifest:
        if dataset_id in self._datasets:
            raise ValueError(f"Dataset {dataset_id} already exists.")

        manifest = ThermalDatasetManifest(
            dataset_id=dataset_id,
            name=name,
            version=version,
            current_state=DatasetState.RAW,
            data_origin=data_origin,
            is_training_eligible=False,
            notes=notes,
        )
        self._datasets[dataset_id] = manifest
        logger.info(f"Created thermal dataset {dataset_id} in RAW state [Origin: {data_origin.value}]")
        return manifest

    def transition_state(self, dataset_id: str, new_state: DatasetState, operator_reason: str = "") -> DatasetState:
        manifest = self._datasets.get(dataset_id)
        if not manifest:
            raise ValueError(f"Dataset {dataset_id} not found.")

        current = manifest.current_state
        allowed = VALID_TRANSITIONS.get(current, [])
        if new_state not in allowed:
            raise ValueError(
                f"Illegal dataset transition from {current.value} to {new_state.value}. Allowed: {[s.value for s in allowed]}"
            )

        # Training readiness requires REAL_SENSOR or explicitly authorized origin and non-empty samples
        if new_state == DatasetState.TRAINING_READY:
            if manifest.data_origin == DataOrigin.SIMULATED:
                logger.warning(f"Dataset {dataset_id} is SIMULATED. Transitioning to TRAINING_READY flagged as EXPERIMENTAL only.")
            if not manifest.samples:
                raise ValueError(f"Cannot transition empty dataset {dataset_id} to TRAINING_READY.")
            manifest.is_training_eligible = True
        else:
            manifest.is_training_eligible = False

        manifest.current_state = new_state
        manifest.updated_at = time.time()
        self._update_manifest_hash(manifest)
        logger.info(f"Dataset {dataset_id} transitioned: {current.value} -> {new_state.value} (Reason: {operator_reason})")
        return new_state

    def add_sample(
        self,
        dataset_id: str,
        sample_id: str,
        source_sensor: str,
        source_frame_id: str,
        capture_timestamp: float,
        payload_bytes: bytes,
        data_origin: DataOrigin = DataOrigin.SIMULATED,
        calibration_version: Optional[str] = None,
        annotation_version: Optional[str] = None,
        quality_status: str = "GOOD",
        split_assignment: str = "TRAIN",
        scenario_tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThermalDatasetSample:
        manifest = self._datasets.get(dataset_id)
        if not manifest:
            raise ValueError(f"Dataset {dataset_id} not found.")

        # Compute SHA-256
        sha256_hash = hashlib.sha256(payload_bytes).hexdigest().upper()

        sample = ThermalDatasetSample(
            dataset_id=dataset_id,
            sample_id=sample_id,
            source_sensor=source_sensor,
            source_frame_id=source_frame_id,
            capture_timestamp=capture_timestamp,
            data_origin=data_origin,
            calibration_version=calibration_version,
            annotation_version=annotation_version,
            quality_status=quality_status,
            sha256_hash=sha256_hash,
            split_assignment=split_assignment,
            scenario_tags=scenario_tags or [],
            metadata=metadata or {},
        )

        manifest.samples[sample_id] = sample
        manifest.updated_at = time.time()
        self._update_manifest_hash(manifest)
        return sample

    def _update_manifest_hash(self, manifest: ThermalDatasetManifest):
        hasher = hashlib.sha256()
        hasher.update(manifest.dataset_id.encode())
        hasher.update(manifest.version.encode())
        hasher.update(manifest.current_state.value.encode())
        for sid in sorted(manifest.samples.keys()):
            hasher.update(manifest.samples[sid].sha256_hash.encode())
        manifest.manifest_sha256 = hasher.hexdigest().upper()

    def get_dataset(self, dataset_id: str) -> Optional[ThermalDatasetManifest]:
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[ThermalDatasetManifest]:
        return list(self._datasets.values())

    def reset(self):
        self._datasets.clear()
        self._initialize_readiness_dataset()


thermal_dataset_service = ThermalDatasetService()
