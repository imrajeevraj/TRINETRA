"""
TRINETRA — Operator Feedback & Quality Governance Service (Phase XIII)
Manages structured, immutable feedback across 11 dispositions, multi-reviewer
consensus workflows, dispute resolution, and training eligibility gating.
"""

from __future__ import annotations
import enum
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("OperatorFeedbackService")


class FeedbackDisposition(str, enum.Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    MISCLASSIFICATION = "MISCLASSIFICATION"
    DUPLICATE = "DUPLICATE"
    TRACKING_ERROR = "TRACKING_ERROR"
    OCR_ERROR = "OCR_ERROR"
    FACE_ASSOCIATION_ERROR = "FACE_ASSOCIATION_ERROR"
    PTZ_CUE_ERROR = "PTZ_CUE_ERROR"
    PREDICTION_ERROR = "PREDICTION_ERROR"
    UNKNOWN = "UNKNOWN"


class FeedbackReviewStatus(str, enum.Enum):
    RAW_FEEDBACK = "RAW_FEEDBACK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    DISPUTED = "DISPUTED"
    TRAINING_ELIGIBLE = "TRAINING_ELIGIBLE"
    USED_IN_TRAINING = "USED_IN_TRAINING"


class ReviewerRole(str, enum.Enum):
    OBSERVER = "OBSERVER"
    REVIEWER = "REVIEWER"
    ML_ENGINEER = "ML_ENGINEER"
    ADMINISTRATOR = "ADMINISTRATOR"


class FeedbackReviewEntry(BaseModel):
    review_id: str
    reviewer_id: str
    reviewer_role: str
    agrees_with_operator: bool
    assigned_disposition: FeedbackDisposition
    corrected_class: Optional[str] = None
    notes: str = ""
    timestamp: float = Field(default_factory=time.time)


class OperatorFeedbackRecord(BaseModel):
    feedback_id: str
    camera_id: str
    frame_id: str
    event_id: Optional[str] = None
    track_id: Optional[int] = None
    global_entity_id: Optional[str] = None
    model_id: str
    model_version: str
    timestamp: float
    operator_id: str
    disposition: FeedbackDisposition
    confidence: float
    reason: str
    evidence_reference: Optional[str] = None
    source_frame_hash: str
    created_at: float = Field(default_factory=time.time)
    review_status: FeedbackReviewStatus = FeedbackReviewStatus.RAW_FEEDBACK
    reviews: List[FeedbackReviewEntry] = Field(default_factory=list)
    validated_class: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OperatorFeedbackService:
    """
    Governs operational feedback collection, immutability, and multi-reviewer consensus.
    Ensures that feedback never automatically mutates active production models.
    """

    def __init__(self):
        self.records: Dict[str, OperatorFeedbackRecord] = {}
        self._counter = 0

    def submit_feedback(
        self,
        camera_id: str,
        frame_id: str,
        model_id: str,
        model_version: str,
        operator_id: str,
        disposition: str,
        confidence: float,
        reason: str,
        source_frame_bytes: Optional[bytes] = None,
        source_frame_hash: Optional[str] = None,
        event_id: Optional[str] = None,
        track_id: Optional[int] = None,
        global_entity_id: Optional[str] = None,
        evidence_reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[OperatorFeedbackRecord]]:
        """
        Submits a new, immutable feedback record.
        """
        try:
            disp_enum = FeedbackDisposition(disposition.upper())
        except ValueError:
            return False, f"INVALID_DISPOSITION_{disposition}", None

        # Compute or verify source frame hash
        if source_frame_bytes:
            frame_sha = hashlib.sha256(source_frame_bytes).hexdigest().upper()
        elif source_frame_hash:
            frame_sha = source_frame_hash.upper()
        else:
            frame_sha = hashlib.sha256(f"{camera_id}:{frame_id}:{time.time()}".encode("utf-8")).hexdigest().upper()

        self._counter += 1
        now = timestamp if timestamp is not None else time.time()
        feedback_id = f"FDB-{camera_id}-{int(now * 1000)}-{self._counter:04d}"

        record = OperatorFeedbackRecord(
            feedback_id=feedback_id,
            camera_id=camera_id,
            frame_id=frame_id,
            event_id=event_id,
            track_id=track_id,
            global_entity_id=global_entity_id,
            model_id=model_id,
            model_version=model_version,
            timestamp=now,
            operator_id=operator_id,
            disposition=disp_enum,
            confidence=confidence,
            reason=reason,
            evidence_reference=evidence_reference,
            source_frame_hash=frame_sha,
            created_at=now,
            review_status=FeedbackReviewStatus.REVIEW_REQUIRED,
            metadata=metadata or {},
        )

        self.records[feedback_id] = record
        logger.info(f"Ingested operator feedback {feedback_id} [{disp_enum.value}] from {operator_id} on {camera_id}")

        publish_event("ai.feedback_received", {
            "feedback_id": feedback_id,
            "camera_id": camera_id,
            "disposition": disp_enum.value,
            "operator_id": operator_id,
            "model_version": model_version,
            "timestamp": now,
        })

        return True, "FEEDBACK_SUBMITTED", record

    def review_feedback(
        self,
        feedback_id: str,
        reviewer_id: str,
        reviewer_role: str,
        agrees_with_operator: bool,
        assigned_disposition: Optional[str] = None,
        corrected_class: Optional[str] = None,
        notes: str = "",
    ) -> Tuple[bool, str, Optional[OperatorFeedbackRecord]]:
        """
        Adds an auditable review entry and manages consensus / dispute transitions.
        """
        record = self.records.get(feedback_id)
        if not record:
            return False, f"FEEDBACK_{feedback_id}_NOT_FOUND", None

        role_str = reviewer_role.upper()
        if role_str not in [r.value for r in ReviewerRole]:
            return False, f"INVALID_REVIEWER_ROLE_{reviewer_role}", None

        if role_str == ReviewerRole.OBSERVER.value:
            return False, "OBSERVER_ROLE_CANNOT_REVIEW_FEEDBACK", None

        disp = record.disposition
        if assigned_disposition:
            try:
                disp = FeedbackDisposition(assigned_disposition.upper())
            except ValueError:
                return False, f"INVALID_ASSIGNED_DISPOSITION_{assigned_disposition}", None

        review_entry = FeedbackReviewEntry(
            review_id=f"REV-{feedback_id}-{len(record.reviews) + 1:02d}",
            reviewer_id=reviewer_id,
            reviewer_role=role_str,
            agrees_with_operator=agrees_with_operator,
            assigned_disposition=disp,
            corrected_class=corrected_class,
            notes=notes,
            timestamp=time.time(),
        )
        record.reviews.append(review_entry)

        # Multi-Reviewer Consensus Evaluation
        dispositions = [r.assigned_disposition for r in record.reviews]
        has_disagreement = len(set(dispositions)) > 1

        if has_disagreement:
            # If an administrator or lead ML engineer makes the final determination:
            if role_str in [ReviewerRole.ADMINISTRATOR.value, ReviewerRole.ML_ENGINEER.value]:
                record.review_status = FeedbackReviewStatus.VALIDATED
                record.validated_class = corrected_class
                publish_event("ai.feedback_validated", {
                    "feedback_id": feedback_id,
                    "resolved_disposition": disp.value,
                    "resolved_by": reviewer_id,
                })
            else:
                record.review_status = FeedbackReviewStatus.DISPUTED
                publish_event("ai.feedback_disputed", {
                    "feedback_id": feedback_id,
                    "dispute_reviewers": [r.reviewer_id for r in record.reviews],
                })
        else:
            if agrees_with_operator:
                record.review_status = FeedbackReviewStatus.VALIDATED
                record.validated_class = corrected_class
                publish_event("ai.feedback_validated", {
                    "feedback_id": feedback_id,
                    "resolved_disposition": disp.value,
                    "resolved_by": reviewer_id,
                })
            else:
                record.review_status = FeedbackReviewStatus.REJECTED

        logger.info(f"Feedback {feedback_id} reviewed by {reviewer_id} ({role_str}) -> Status: {record.review_status.value}")
        return True, "REVIEW_RECORDED", record

    def mark_training_eligible(self, feedback_id: str, operator_id: str) -> Tuple[bool, str]:
        """Marks validated feedback as eligible for training dataset curation."""
        rec = self.records.get(feedback_id)
        if not rec:
            return False, "FEEDBACK_NOT_FOUND"
        if rec.review_status != FeedbackReviewStatus.VALIDATED:
            return False, f"CANNOT_GATE_NON_VALIDATED_STATUS_{rec.review_status.value}"

        rec.review_status = FeedbackReviewStatus.TRAINING_ELIGIBLE
        return True, "MARKED_TRAINING_ELIGIBLE"

    def get_feedback(self, feedback_id: str) -> Optional[OperatorFeedbackRecord]:
        return self.records.get(feedback_id)

    def list_feedback(
        self,
        camera_id: Optional[str] = None,
        disposition: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 100,
    ) -> List[OperatorFeedbackRecord]:
        res = list(self.records.values())
        if camera_id:
            res = [r for r in res if r.camera_id == camera_id]
        if disposition:
            res = [r for r in res if r.disposition.value == disposition.upper()]
        if review_status:
            res = [r for r in res if r.review_status.value == review_status.upper()]
        return res[:limit]

    def reset(self):
        self.records.clear()
        self._counter = 0


operator_feedback_service = OperatorFeedbackService()
