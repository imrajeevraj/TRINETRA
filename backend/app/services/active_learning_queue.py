"""
TRINETRA — Active Learning Prioritization Queue (Phase XIII)
Ranks mined hard-cases and feedback records according to multi-factor expected information gain.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("ActiveLearningQueue")


class ActiveLearningItem(BaseModel):
    queue_id: str
    sample_id: str
    model_domain: str
    camera_id: str
    failure_type: str
    priority_score: float
    uncertainty_score: float
    disagreement_score: float
    model_diff_score: float
    rarity_score: float
    class_criticality: float
    camera_failure_weight: float
    review_status: str = "PENDING_REVIEW"
    enqueued_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActiveLearningQueue:
    """
    Intelligent sample selection queue that prioritizes the most informative edge cases for review.
    """

    CLASS_CRITICALITY = {
        "firearm": 1.0,
        "weapon": 1.0,
        "person": 0.90,
        "drone": 0.85,
        "aircraft": 0.85,
        "vehicle": 0.70,
    }

    def __init__(self, max_capacity: int = 5000):
        self.queue: Dict[str, ActiveLearningItem] = {}
        self.max_capacity = max_capacity
        self._counter = 0

    def calculate_priority_score(
        self,
        confidence: float,
        is_disputed: bool,
        champion_conf: Optional[float] = None,
        challenger_conf: Optional[float] = None,
        failure_type_count: int = 1,
        class_name: str = "person",
        camera_historical_fp_rate: float = 0.05,
        w_u: float = 0.25,
        w_d: float = 0.25,
        w_m: float = 0.20,
        w_r: float = 0.10,
        w_c: float = 0.10,
        w_k: float = 0.10,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates normalized priority score S_active in [0.0, 1.0].
        """
        # 1. Uncertainty: maximized at conf = 0.50
        u_score = max(0.0, 1.0 - 2.0 * abs(confidence - 0.50))

        # 2. Operator Disagreement
        d_score = 1.0 if is_disputed else 0.0

        # 3. Model Disagreement (Champion vs Challenger shadow)
        if champion_conf is not None and challenger_conf is not None:
            m_score = min(1.0, abs(champion_conf - challenger_conf))
        else:
            m_score = 0.0

        # 4. Rarity score: inverse of occurrences
        r_score = min(1.0, 1.0 / max(1, failure_type_count))

        # 5. Class Criticality
        c_score = self.CLASS_CRITICALITY.get(class_name.lower(), 0.50)

        # 6. Camera Historical Failure Rate
        k_score = min(1.0, camera_historical_fp_rate * 5.0)

        total_score = (
            (w_u * u_score) +
            (w_d * d_score) +
            (w_m * m_score) +
            (w_r * r_score) +
            (w_c * c_score) +
            (w_k * k_score)
        )
        total_score = round(min(1.0, max(0.0, total_score)), 4)

        factors = {
            "uncertainty": round(u_score, 3),
            "disagreement": round(d_score, 3),
            "model_diff": round(m_score, 3),
            "rarity": round(r_score, 3),
            "class_crit": round(c_score, 3),
            "camera_fail": round(k_score, 3),
        }
        return total_score, factors

    def enqueue_sample(
        self,
        sample_id: str,
        model_domain: str,
        camera_id: str,
        failure_type: str,
        confidence: float,
        is_disputed: bool = False,
        champion_conf: Optional[float] = None,
        challenger_conf: Optional[float] = None,
        failure_type_count: int = 1,
        class_name: str = "person",
        camera_historical_fp_rate: float = 0.05,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActiveLearningItem:
        """
        Calculates priority and enqueues sample into prioritization queue.
        """
        score, factors = self.calculate_priority_score(
            confidence=confidence,
            is_disputed=is_disputed,
            champion_conf=champion_conf,
            challenger_conf=challenger_conf,
            failure_type_count=failure_type_count,
            class_name=class_name,
            camera_historical_fp_rate=camera_historical_fp_rate,
        )

        self._counter += 1
        qid = f"ALQ-{model_domain}-{int(time.time())}-{self._counter:05d}"

        item = ActiveLearningItem(
            queue_id=qid,
            sample_id=sample_id,
            model_domain=model_domain,
            camera_id=camera_id,
            failure_type=failure_type,
            priority_score=score,
            uncertainty_score=factors["uncertainty"],
            disagreement_score=factors["disagreement"],
            model_diff_score=factors["model_diff"],
            rarity_score=factors["rarity"],
            class_criticality=factors["class_crit"],
            camera_failure_weight=factors["camera_fail"],
            metadata=metadata or {},
        )

        # Capacity management
        if len(self.queue) >= self.max_capacity:
            # Drop lowest priority item
            lowest_k = min(self.queue.keys(), key=lambda k: self.queue[k].priority_score)
            self.queue.pop(lowest_k)

        self.queue[qid] = item
        logger.info(f"Enqueued sample {sample_id} to {model_domain} ALQ with priority {score:.4f}")
        return item

    def dequeue_highest_priority(self, model_domain: Optional[str] = None) -> Optional[ActiveLearningItem]:
        """Dequeues the highest priority item for active review."""
        candidates = list(self.queue.values())
        if model_domain:
            candidates = [c for c in candidates if c.model_domain == model_domain]

        if not candidates:
            return None

        highest = max(candidates, key=lambda x: x.priority_score)
        self.queue.pop(highest.queue_id, None)
        return highest

    def list_queue(self, model_domain: Optional[str] = None, limit: int = 100) -> List[ActiveLearningItem]:
        items = list(self.queue.values())
        if model_domain:
            items = [i for i in items if i.model_domain == model_domain]
        items.sort(key=lambda x: x.priority_score, reverse=True)
        return items[:limit]

    def reset(self):
        self.queue.clear()
        self._counter = 0


active_learning_queue = ActiveLearningQueue()
