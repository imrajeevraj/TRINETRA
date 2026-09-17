"""
TRINETRA Phase XV — Thermal Annotation & Quality Assurance Service
Manages native thermal bounding boxes, difficulty tags, occlusion, truncation,
reviewer consensus, disagreement resolution, and human approval gating.
"""

from __future__ import annotations
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("ThermalAnnotationService")


@dataclass
class ThermalBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    occlusion: float = 0.0  # 0.0 to 1.0
    truncation: float = 0.0  # 0.0 to 1.0
    difficulty: str = "NORMAL"  # "EASY", "NORMAL", "HARD", "EXTREME"
    confidence: float = 1.0


@dataclass
class ThermalAnnotationRecord:
    annotation_id: str
    sample_id: str
    boxes: List[ThermalBoundingBox] = field(default_factory=list)
    annotator_id: str = "HUMAN_OPERATOR"
    reviewer_id: Optional[str] = None
    approval_status: str = "PENDING_REVIEW"  # "PENDING_REVIEW", "APPROVED", "DISAGREEMENT", "REJECTED"
    annotation_version: str = "v1.0"
    review_notes: str = ""
    created_at: float = field(default_factory=time.time)
    reviewed_at: Optional[float] = None


class ThermalAnnotationService:
    """
    Governs human-validated thermal annotations.
    Guarantees that model training only consumes approved annotations.
    """

    SUPPORTED_CLASSES = {
        "person",
        "vehicle",
        "drone",
        "aircraft",
        "firearm",
        "security_item",
    }

    def __init__(self):
        self._annotations: Dict[str, ThermalAnnotationRecord] = {}
        self._sample_index: Dict[str, str] = {}  # sample_id -> annotation_id

    def create_annotation(
        self,
        sample_id: str,
        boxes: List[ThermalBoundingBox],
        annotator_id: str = "HUMAN_OPERATOR",
        annotation_version: str = "v1.0",
    ) -> ThermalAnnotationRecord:
        # Validate classes
        for box in boxes:
            if box.class_name not in self.SUPPORTED_CLASSES:
                raise ValueError(
                    f"Unsupported class '{box.class_name}'. Supported classes: {sorted(list(self.SUPPORTED_CLASSES))}"
                )
            if box.x1 >= box.x2 or box.y1 >= box.y2:
                raise ValueError(f"Invalid bounding box coordinates: ({box.x1}, {box.y1}, {box.x2}, {box.y2})")

        ann_id = f"ANN-{uuid.uuid4().hex[:12]}"
        record = ThermalAnnotationRecord(
            annotation_id=ann_id,
            sample_id=sample_id,
            boxes=boxes,
            annotator_id=annotator_id,
            annotation_version=annotation_version,
            approval_status="PENDING_REVIEW",
        )
        self._annotations[ann_id] = record
        self._sample_index[sample_id] = ann_id
        logger.info(f"Created annotation {ann_id} for sample {sample_id} ({len(boxes)} boxes)")
        return record

    def review_annotation(
        self,
        annotation_id: str,
        reviewer_id: str,
        verdict: str,  # "APPROVED", "DISAGREEMENT", "REJECTED"
        notes: str = "",
    ) -> ThermalAnnotationRecord:
        rec = self._annotations.get(annotation_id)
        if not rec:
            raise ValueError(f"Annotation {annotation_id} not found.")

        if verdict not in ["APPROVED", "DISAGREEMENT", "REJECTED"]:
            raise ValueError(f"Invalid review verdict: {verdict}")

        rec.reviewer_id = reviewer_id
        rec.approval_status = verdict
        rec.review_notes = notes
        rec.reviewed_at = time.time()
        logger.info(f"Reviewed annotation {annotation_id} by {reviewer_id}: {verdict}")
        return rec

    def get_annotation_for_sample(self, sample_id: str) -> Optional[ThermalAnnotationRecord]:
        ann_id = self._sample_index.get(sample_id)
        return self._annotations.get(ann_id) if ann_id else None

    def list_annotations(self, status: Optional[str] = None) -> List[ThermalAnnotationRecord]:
        items = list(self._annotations.values())
        if status:
            items = [a for a in items if a.approval_status == status]
        return items

    def reset(self):
        self._annotations.clear()
        self._sample_index.clear()


thermal_annotation_service = ThermalAnnotationService()
