"""
TRINETRA — Dataset Governance & Label Validation Service (Phase XII)
Enforces strict dataset versioning, sample lineage tracking, label integrity checks,
bounding-box validity, and absolute quarantine of the frozen benchmark (IBVAP-GT-v1.0).
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("DatasetGovernanceService")

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_HASHES_PATH = REPO_ROOT / "benchmark" / "checksums" / "images.sha256"


class CollectedSample(BaseModel):
    sample_id: str
    camera_id: str
    frame_id: str
    timestamp: float
    source: str  # "EDGE_NODE", "OPERATOR_DISPOSITION", "OFFLINE_AUDIT"
    data_origin: str  # "FIELD_CAMERA", "SIMULATED_TEST", "SYNTHETIC"
    model_version: str
    inference_result: Dict[str, Any]
    operator_disposition: Optional[str] = None  # "CONFIRMED_THREAT", "FALSE_POSITIVE", "FALSE_NEGATIVE"
    collection_reason: str  # "SMALL_OBJECT", "OCCLUSION", "SHADOW", "EDGE_FAILURE", "DISTRACTOR"
    image_sha256: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    dataset_id: str
    dataset_version: str
    domain: str  # "GROUND", "AIRBORNE", "SECURITY_ITEM", "THERMAL"
    classes: Dict[int, str]
    annotation_format: str = "YOLO_TXT"  # class x_center y_center width height (normalized)
    sample_count: int
    manifest_hash: str
    quality_status: str  # "PASSED", "WARNING", "REJECTED"
    benchmark_isolated: bool = True
    created_at: float = Field(default_factory=time.time)
    source_lineage: List[str] = Field(default_factory=list)


class DatasetValidationResult(BaseModel):
    is_valid: bool
    total_images_checked: int
    total_labels_checked: int
    malformed_labels: List[str] = Field(default_factory=list)
    invalid_class_ids: List[str] = Field(default_factory=list)
    out_of_bounds_boxes: List[str] = Field(default_factory=list)
    zero_area_boxes: List[str] = Field(default_factory=list)
    benchmark_leakage_detected: List[str] = Field(default_factory=list)
    duplicate_samples: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)


class DatasetGovernanceService:
    """
    Governs data collection, annotation validation, and dataset immutability.
    Enforces that frozen benchmark samples never leak into training datasets.
    """

    def __init__(self):
        self.collected_samples: Dict[str, CollectedSample] = {}
        self.datasets: Dict[str, DatasetManifest] = {}
        self.frozen_benchmark_hashes: Set[str] = set()
        self._sample_counter = 0
        self._load_frozen_benchmark_hashes()

    def _load_frozen_benchmark_hashes(self):
        """Loads all SHA-256 hashes belonging to the frozen benchmark IBVAP-GT-v1.0."""
        if BENCHMARK_HASHES_PATH.exists():
            try:
                with open(BENCHMARK_HASHES_PATH, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split()
                            if parts:
                                self.frozen_benchmark_hashes.add(parts[0].upper())
                logger.info(
                    f"Loaded {len(self.frozen_benchmark_hashes)} frozen benchmark hashes for strict isolation."
                )
            except Exception as e:
                logger.error(f"Failed to load benchmark checksums: {e}")
        else:
            logger.warning(f"Benchmark checksum file not found at {BENCHMARK_HASHES_PATH}")

    def ingest_sample(
        self,
        camera_id: str,
        frame_id: str,
        source: str,
        data_origin: str,
        model_version: str,
        inference_result: Dict[str, Any],
        collection_reason: str,
        image_bytes: bytes,
        operator_disposition: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[CollectedSample]]:
        """
        Ingests a candidate frame into the data collection pipeline.
        Rejects immediately if sample matches any frozen benchmark image hash.
        """
        img_sha = hashlib.sha256(image_bytes).hexdigest().upper()

        # Strict Benchmark Isolation Check
        if img_sha in self.frozen_benchmark_hashes:
            err = f"BENCHMARK_LEAKAGE_REJECTED: Frame {frame_id} matches frozen benchmark IBVAP-GT-v1.0!"
            logger.critical(err)
            return False, err, None

        self._sample_counter += 1
        now = timestamp if timestamp is not None else time.time()
        sample_id = f"SMP-{camera_id}-{int(now * 1000)}-{self._sample_counter:04d}"

        sample = CollectedSample(
            sample_id=sample_id,
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=now,
            source=source,
            data_origin=data_origin,
            model_version=model_version,
            inference_result=inference_result,
            operator_disposition=operator_disposition,
            collection_reason=collection_reason,
            image_sha256=img_sha,
            metadata=metadata or {},
        )

        self.collected_samples[sample_id] = sample
        logger.info(f"Ingested failure sample {sample_id} ({collection_reason}) from {camera_id}")
        return True, "SAMPLE_INGESTED", sample

    def validate_dataset_annotations(
        self,
        images_and_hashes: Dict[str, str],  # image_name -> sha256
        annotations: Dict[str, List[Tuple[int, float, float, float, float]]],  # image_name -> [(cls, x, y, w, h)]
        allowed_classes: Set[int],
    ) -> DatasetValidationResult:
        """
        Validates annotations and checks for:
        1. Frozen benchmark leakage
        2. Out-of-bounds coordinates
        3. Zero-area boxes
        4. Invalid class IDs
        5. Duplicate image hashes
        """
        result = DatasetValidationResult(
            is_valid=True,
            total_images_checked=len(images_and_hashes),
            total_labels_checked=sum(len(v) for v in annotations.values()),
        )

        seen_hashes: Set[str] = set()

        for img_name, img_sha in images_and_hashes.items():
            img_sha_upper = img_sha.upper()

            # Benchmark leakage check
            if img_sha_upper in self.frozen_benchmark_hashes:
                result.benchmark_leakage_detected.append(f"{img_name}:{img_sha_upper}")
                result.is_valid = False

            # Duplicate check
            if img_sha_upper in seen_hashes:
                result.duplicate_samples.append(img_name)
            seen_hashes.add(img_sha_upper)

            # Annotation checks
            boxes = annotations.get(img_name, [])
            for box in boxes:
                cls_id, x, y, w, h = box
                if cls_id not in allowed_classes:
                    result.invalid_class_ids.append(f"{img_name}:class_{cls_id}")
                    result.is_valid = False

                # Bounding box coordinates must be normalized [0, 1]
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    result.out_of_bounds_boxes.append(f"{img_name}:coords({x},{y},{w},{h})")
                    result.is_valid = False

                # Zero area check
                if w <= 1e-6 or h <= 1e-6:
                    result.zero_area_boxes.append(f"{img_name}:zero_area({w}x{h})")
                    result.is_valid = False

        if result.benchmark_leakage_detected:
            result.rejection_reasons.append("CRITICAL: Frozen benchmark images detected in training dataset!")
        if result.invalid_class_ids:
            result.rejection_reasons.append(f"Invalid class IDs found in {len(result.invalid_class_ids)} labels")
        if result.out_of_bounds_boxes:
            result.rejection_reasons.append(f"Out of bounds boxes in {len(result.out_of_bounds_boxes)} labels")
        if result.zero_area_boxes:
            result.rejection_reasons.append(f"Zero area boxes in {len(result.zero_area_boxes)} labels")

        return result

    def register_dataset_version(
        self,
        dataset_id: str,
        dataset_version: str,
        domain: str,
        classes: Dict[int, str],
        images_and_hashes: Dict[str, str],
        annotations: Dict[str, List[Tuple[int, float, float, float, float]]],
    ) -> Tuple[bool, str, Optional[DatasetManifest]]:
        """
        Validates and registers an immutable dataset version.
        """
        if dataset_id in self.datasets:
            return False, f"DATASET_{dataset_id}_ALREADY_EXISTS", None

        # Run validation
        val_result = self.validate_dataset_annotations(
            images_and_hashes=images_and_hashes,
            annotations=annotations,
            allowed_classes=set(classes.keys()),
        )

        if not val_result.is_valid:
            reasons = "; ".join(val_result.rejection_reasons)
            return False, f"DATASET_VALIDATION_FAILED: {reasons}", None

        # Compute manifest hash
        manifest_input = json.dumps(
            {
                "id": dataset_id,
                "version": dataset_version,
                "domain": domain,
                "samples": sorted(list(images_and_hashes.values())),
            },
            sort_keys=True,
        )
        manifest_hash = hashlib.sha256(manifest_input.encode("utf-8")).hexdigest().upper()

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            domain=domain,
            classes=classes,
            sample_count=len(images_and_hashes),
            manifest_hash=manifest_hash,
            quality_status="PASSED",
            benchmark_isolated=True,
            source_lineage=[s.source for s in self.collected_samples.values()][:50],
        )

        self.datasets[dataset_id] = manifest
        logger.info(f"Registered validated dataset {dataset_id} ({manifest.sample_count} samples, SHA {manifest_hash[:16]}...)")
        return True, "DATASET_REGISTERED", manifest

    def reset(self):
        self.collected_samples.clear()
        self.datasets.clear()
        self._sample_counter = 0


dataset_governance_service = DatasetGovernanceService()
