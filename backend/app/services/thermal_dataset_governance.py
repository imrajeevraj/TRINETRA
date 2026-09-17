"""
TRINETRA Phase XIV — Thermal Dataset Governance Service
Enforces strict dataset readiness status (IBVAP-THERMAL-READINESS),
SHA-256 manifest lineage, cross-spectral label alignment, and benchmark quarantine.
"""

from __future__ import annotations
import os
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.sensor_abstraction import DataOrigin

logger = logging.getLogger("ThermalDatasetGovernance")


class LabelSource(str, Enum):
    HUMAN_VERIFIED = "HUMAN_VERIFIED"
    OPERATOR_FEEDBACK = "OPERATOR_FEEDBACK"
    IMPORT = "IMPORT"
    EXPERIMENTAL = "EXPERIMENTAL"


@dataclass
class ThermalAnnotation:
    annotation_id: str
    frame_id: str
    class_name: str
    bbox: List[float]                  # [x_center, y_center, w, h] normalized
    confidence: float
    label_source: LabelSource
    reviewer: str
    timestamp: float = field(default_factory=time.time)
    rgb_frame_id: Optional[str] = None
    thermal_frame_id: Optional[str] = None
    registration_id: Optional[str] = None
    shared_annotation_id: Optional[str] = None


@dataclass
class ThermalDatasetManifest:
    dataset_id: str
    version: str
    sensor_ids: List[str]
    modalities: List[str]
    classes: List[str]
    annotation_format: str
    calibration_version: str
    frame_hash_manifest: List[str]
    data_origin: DataOrigin
    label_status: str
    benchmark_status: str
    creation_timestamp: float = field(default_factory=time.time)
    is_training_eligible: bool = False
    notes: str = ""


class ThermalDatasetGovernanceService:
    """
    Governs thermal dataset creation, annotation curation, and benchmark isolation.
    """

    BENCHMARK_CHECKSUM_PATH = "benchmark/checksums/images.sha256"

    def __init__(self):
        self._datasets: Dict[str, ThermalDatasetManifest] = {}
        self._annotations: Dict[str, List[ThermalAnnotation]] = {}
        self._quarantined_hashes: set[str] = set()
        self._load_quarantined_benchmark_hashes()
        self._initialize_readiness_dataset()

    def _load_quarantined_benchmark_hashes(self):
        if os.path.exists(self.BENCHMARK_CHECKSUM_PATH):
            try:
                with open(self.BENCHMARK_CHECKSUM_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            self._quarantined_hashes.add(parts[0].upper())
                logger.info(f"Loaded {len(self._quarantined_hashes)} quarantined benchmark hashes for thermal isolation.")
            except Exception as e:
                logger.warning(f"Could not load benchmark hashes: {e}")

    def _initialize_readiness_dataset(self):
        """Initializes default IBVAP-THERMAL-READINESS manifest."""
        manifest = ThermalDatasetManifest(
            dataset_id="IBVAP-THERMAL-READINESS",
            version="v0.1-spec",
            sensor_ids=["SNS-CAM005-LWIR", "SNS-CAM006-LWIR", "SNS-CAM007-LWIR"],
            modalities=["THERMAL_LWIR", "OPTICAL_RGB"],
            classes=["person", "vehicle", "drone", "aircraft", "security_item"],
            annotation_format="YOLO_NORMALIZED",
            calibration_version="CAL-SPEC-v1.0",
            frame_hash_manifest=[],
            data_origin=DataOrigin.SIMULATED,
            label_status="UNLABELED / PENDING_OPERATIONAL_COLLECTION",
            benchmark_status="QUARANTINED_BENCHMARK_PROTECTED",
            is_training_eligible=False,
            notes="Infrastructure readiness specification; no genuine physical LWIR dataset collected yet.",
        )
        self._datasets[manifest.dataset_id] = manifest

    def add_sample_to_dataset(
        self,
        dataset_id: str,
        frame_id: str,
        frame_hash: str,
        annotation: Optional[ThermalAnnotation] = None,
    ) -> tuple[bool, str]:
        """
        Adds sample to dataset after verifying benchmark quarantine and label governance.
        """
        # 1. Benchmark Quarantine Check
        if frame_hash.upper() in self._quarantined_hashes:
            logger.error(f"REJECTED: Frame {frame_id} matches quarantined benchmark hash {frame_hash}!")
            return False, "BENCHMARK_LEAKAGE_REJECTED"

        manifest = self._datasets.get(dataset_id)
        if not manifest:
            return False, f"DATASET_NOT_FOUND: {dataset_id}"

        # 2. Label Source Governance (only HUMAN_VERIFIED can enter training-eligible sets)
        if manifest.is_training_eligible and annotation:
            if annotation.label_source != LabelSource.HUMAN_VERIFIED:
                return False, f"LABEL_GOVERNANCE_REJECTED: Training sets require HUMAN_VERIFIED labels (got {annotation.label_source.value})"

        manifest.frame_hash_manifest.append(frame_hash)
        if annotation:
            if frame_id not in self._annotations:
                self._annotations[frame_id] = []
            self._annotations[frame_id].append(annotation)

        return True, "SAMPLE_ACCEPTED"

    def get_dataset(self, dataset_id: str) -> Optional[ThermalDatasetManifest]:
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[ThermalDatasetManifest]:
        return list(self._datasets.values())

    def reset(self):
        self._datasets.clear()
        self._annotations.clear()
        self._initialize_readiness_dataset()


thermal_dataset_governance = ThermalDatasetGovernanceService()
