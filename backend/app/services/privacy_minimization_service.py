"""
TRINETRA — Privacy & Data Minimization Service (Phase XIII)
Enforces retention policies, evidence expiration, frame hashing,
and auditable data pruning to prevent perpetual storage of surveillance video.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("PrivacyMinimizationService")


class RetentionPolicyConfig(BaseModel):
    feedback_retention_days: int = 90
    unvalidated_evidence_retention_days: int = 14
    hard_case_retention_days: int = 30
    quarantine_retention_days: int = 180
    anonymize_faces_after_days: int = 7


class PruneAuditRecord(BaseModel):
    prune_id: str
    items_pruned: int
    data_type: str
    storage_freed_mb: float
    executed_at: float = Field(default_factory=time.time)
    operator: str = "SYSTEM_CLEANUP"


class PrivacyMinimizationService:
    """
    Guarantees that video evidence is retained only as long as operationally justified.
    """

    def __init__(self, config: Optional[RetentionPolicyConfig] = None):
        self.config = config or RetentionPolicyConfig()
        self.prune_history: List[PruneAuditRecord] = []
        self._counter = 0

    def evaluate_expiration(
        self,
        record_timestamp: float,
        record_type: str = "HARD_CASE",
    ) -> bool:
        """Returns True if the record has exceeded its retention policy."""
        now = time.time()
        age_days = (now - record_timestamp) / 86400.0

        if record_type == "OPERATOR_FEEDBACK":
            return age_days > self.config.feedback_retention_days
        elif record_type == "HARD_CASE":
            return age_days > self.config.hard_case_retention_days
        elif record_type == "UNVALIDATED_EVIDENCE":
            return age_days > self.config.unvalidated_evidence_retention_days
        elif record_type == "QUARANTINE":
            return age_days > self.config.quarantine_retention_days
        return False

    def log_prune_action(self, data_type: str, items_pruned: int, storage_mb: float = 0.0) -> PruneAuditRecord:
        self._counter += 1
        record = PruneAuditRecord(
            prune_id=f"PRN-{int(time.time())}-{self._counter:03d}",
            items_pruned=items_pruned,
            data_type=data_type,
            storage_freed_mb=storage_mb,
            executed_at=time.time(),
        )
        self.prune_history.append(record)
        logger.info(f"Pruned {items_pruned} expired {data_type} records (Freed {storage_mb} MB)")
        return record

    def reset(self):
        self.prune_history.clear()
        self._counter = 0


privacy_minimization_service = PrivacyMinimizationService()
