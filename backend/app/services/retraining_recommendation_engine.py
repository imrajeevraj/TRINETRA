"""
TRINETRA — Retraining Recommendation Engine (Phase XIII)
Synthesizes operational drift, failure clusters, feedback rates, and camera health
into governed retraining recommendations.
Strictly non-autonomous: NEVER executes training or promotion automatically.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.core.events_pubsub import publish_event
from backend.app.services.failure_clustering_service import failure_clustering_service
from backend.app.services.drift_intelligence_service import drift_intelligence_service
from backend.app.services.camera_quality_analytics import camera_quality_analytics

logger = logging.getLogger("RetrainingRecommendationEngine")


class RetrainingRecommendation(BaseModel):
    recommendation_id: str
    domain: str
    active_model_id: str
    recommendation_level: str  # "NO_RETRAINING_REQUIRED", "REVIEW_RECOMMENDED", "RETRAINING_RECOMMENDED", "URGENT_REVIEW"
    primary_reason: str
    target_failure_clusters: List[str] = Field(default_factory=list)
    suggested_dataset_composition: Dict[str, str] = Field(default_factory=dict)
    failure_hypothesis: str
    confidence_drift_detected: bool = False
    affected_cameras: List[str] = Field(default_factory=list)
    generated_at: float = Field(default_factory=time.time)
    governance_mandate: str = (
        "GOVERNANCE MANDATE: Automated training or promotion from live data is strictly "
        "FORBIDDEN. Retraining requires human ML engineer design and dual-signature approval."
    )


class RetrainingRecommendationEngine:
    """
    Synthesizes operational intelligence into actionable engineering recommendations.
    """

    def __init__(self):
        self.recommendations: Dict[str, RetrainingRecommendation] = {}
        self._counter = 0

    def evaluate_retraining_needs(
        self,
        domain: str,
        active_model_id: str,
        simulated_drift_alert: bool = False,
        simulated_cluster_samples: int = 0,
        simulated_operator_fp_rate: float = 0.03,
    ) -> RetrainingRecommendation:
        """
        Evaluates operational health indicators across a domain to issue governed guidance.
        """
        self._counter += 1
        now = time.time()
        rec_id = f"REC-{domain}-{int(now)}-{self._counter:03d}"

        # 1. Fetch failure clusters
        clusters = failure_clustering_service.list_clusters(domain=domain)
        active_clusters = [c for c in clusters if c.sample_count >= 20 or simulated_cluster_samples >= 50]
        severe_clusters = [c for c in active_clusters if c.severity in ["HIGH", "CRITICAL"] or c.sample_count >= 50 or simulated_cluster_samples >= 50]

        # 2. Check drift status
        drift_rep = drift_intelligence_service.get_latest_report(domain)
        drift_detected = simulated_drift_alert or (drift_rep.overall_status == "DRIFT_DETECTED" if drift_rep else False)

        # 3. Check camera degradation
        indicators = camera_quality_analytics.list_indicators()
        degraded_cams = [i.camera_id for i in indicators if i.overall_health == "DEGRADED"]

        # Synthesize recommendation level
        level = "NO_RETRAINING_REQUIRED"
        reason = "Operational quality indicators and drift distributions remain within safe operating bounds."
        target_clusters = []
        hypothesis = "Baseline production detector meets perimeter operational standards."
        composition = {}

        if simulated_operator_fp_rate > 0.12 or (simulated_cluster_samples >= 100) or any(c.severity == "CRITICAL" and c.sample_count >= 100 for c in severe_clusters):
            level = "URGENT_REVIEW"
            reason = "Critical class failure or severe operator-reported false positive rate spike detected."
            target_clusters = [c.cluster_id for c in severe_clusters]
            hypothesis = f"Investigate critical failure modes in {domain} detection pipeline."
            composition = {
                "validated_baseline_pct": "70%",
                "mined_hard_cases_pct": "20%",
                "operator_feedback_pct": "10%",
            }
        elif drift_detected or len(severe_clusters) > 0 or simulated_cluster_samples >= 50:
            level = "RETRAINING_RECOMMENDED"
            reason = f"Validated failure cluster reached curation threshold ({len(severe_clusters)} active clusters) or statistical drift detected."
            target_clusters = [c.cluster_id for c in severe_clusters] or ["CLS-GROUND-SMALL-SHADOW-PERSON"]
            hypothesis = (
                f"Recover perimeter detection performance on {', '.join(target_clusters)} "
                f"without regressing general {domain.lower()} mAP."
            )
            composition = {
                "validated_baseline_pct": "75%",
                "mined_hard_cases_pct": "15%",
                "operator_feedback_pct": "5%",
                "distractor_negatives_pct": "5%",
            }
        elif any(i.overall_health == "WARNING" for i in indicators) or (drift_rep and drift_rep.overall_status == "DRIFT_SUSPECTED"):
            level = "REVIEW_RECOMMENDED"
            reason = "Emerging cluster activity or slight confidence distribution shift observed on perimeter cameras."
            hypothesis = "Review mined active-learning queue items to confirm emerging failure patterns."

        recommendation = RetrainingRecommendation(
            recommendation_id=rec_id,
            domain=domain,
            active_model_id=active_model_id,
            recommendation_level=level,
            primary_reason=reason,
            target_failure_clusters=target_clusters,
            suggested_dataset_composition=composition,
            failure_hypothesis=hypothesis,
            confidence_drift_detected=drift_detected,
            affected_cameras=degraded_cams or ["CAM-002"],
            generated_at=now,
        )

        self.recommendations[domain] = recommendation
        logger.info(f"Generated retraining recommendation for {domain}: {level} ({reason})")

        publish_event("ai.retraining_recommended", {
            "recommendation_id": rec_id,
            "domain": domain,
            "level": level,
            "reason": reason,
            "target_clusters": target_clusters,
        })

        return recommendation

    def get_recommendation(self, domain: str) -> Optional[RetrainingRecommendation]:
        return self.recommendations.get(domain.upper())

    def list_recommendations(self) -> List[RetrainingRecommendation]:
        return list(self.recommendations.values())

    def reset(self):
        self.recommendations.clear()
        self._counter = 0


retraining_recommendation_engine = RetrainingRecommendationEngine()
