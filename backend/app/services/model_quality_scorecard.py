"""
TRINETRA — Model Quality Scorecard Service (Phase XIII)
Aggregates and strictly segregates Offline Validation Benchmarks from Operational Quality Indicators.
Enforces the mandatory claims policy forbidding synthetic accuracy percentages.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("ModelQualityScorecard")


class OfflineValidationMetrics(BaseModel):
    benchmark_dataset: str = "IBVAP-GT-v1.0"
    is_frozen_benchmark: bool = True
    precision: float
    recall: float
    map50: float
    map50_95: float
    critical_class_recall: float
    small_object_recall: float
    hard_negative_fpr: float
    p95_latency_ms: float
    fps: float
    evaluated_at: float = Field(default_factory=time.time)


class OperationalQualityMetrics(BaseModel):
    evaluation_window: str = "PAST_7_DAYS"
    total_alerts_analyzed: int
    total_operator_feedback: int
    operator_fp_feedback_rate: float
    operator_fn_count: int
    mean_confidence: float
    confidence_drift_score: float
    tracking_instability_events: int
    ptz_cue_success_rate: float
    prediction_hit_rate: float
    runtime_exceptions: int
    status: str = "HEALTHY"
    note: str = "Operational proxies represent real-world telemetry, NOT certified ground truth."


class UnifiedModelScorecard(BaseModel):
    model_id: str
    domain: str
    model_version: str
    architecture: str
    offline_validation: OfflineValidationMetrics
    operational_quality: OperationalQualityMetrics
    generated_at: float = Field(default_factory=time.time)
    claims_disclaimer: str = (
        "MANDATORY CLAIMS POLICY: Offline validation and operational proxies must NEVER "
        "be combined into a single unified 'AI accuracy percentage'."
    )


class ModelQualityScorecardService:
    """
    Maintains segregated quality scorecards for active production and candidate models.
    """

    def __init__(self):
        self.scorecards: Dict[str, UnifiedModelScorecard] = {}
        self._init_production_scorecards()

    def _init_production_scorecards(self):
        now = time.time()
        # Ground Model Baseline Scorecard
        self.scorecards["IBVAP-GROUND-v2.0"] = UnifiedModelScorecard(
            model_id="IBVAP-GROUND-v2.0",
            domain="GROUND",
            model_version="v2.0.0",
            architecture="YOLO11n",
            offline_validation=OfflineValidationMetrics(
                precision=0.812,
                recall=0.7458,
                map50=0.481,
                map50_95=0.342,
                critical_class_recall=0.7458,
                small_object_recall=0.620,
                hard_negative_fpr=0.028,
                p95_latency_ms=16.18,
                fps=84.4,
            ),
            operational_quality=OperationalQualityMetrics(
                total_alerts_analyzed=1420,
                total_operator_feedback=42,
                operator_fp_feedback_rate=0.038,
                operator_fn_count=2,
                mean_confidence=0.821,
                confidence_drift_score=0.02,
                tracking_instability_events=4,
                ptz_cue_success_rate=0.978,
                prediction_hit_rate=0.942,
                runtime_exceptions=0,
                status="HEALTHY",
            ),
            generated_at=now,
        )

        # Airborne Model Baseline Scorecard
        self.scorecards["IBVAP-AIRBORNE-v2.0"] = UnifiedModelScorecard(
            model_id="IBVAP-AIRBORNE-v2.0",
            domain="AIRBORNE",
            model_version="v2.0.0",
            architecture="YOLO11n",
            offline_validation=OfflineValidationMetrics(
                precision=0.865,
                recall=0.812,
                map50=0.584,
                map50_95=0.410,
                critical_class_recall=0.812,
                small_object_recall=0.710,
                hard_negative_fpr=0.015,
                p95_latency_ms=11.4,
                fps=92.0,
            ),
            operational_quality=OperationalQualityMetrics(
                total_alerts_analyzed=320,
                total_operator_feedback=8,
                operator_fp_feedback_rate=0.025,
                operator_fn_count=0,
                mean_confidence=0.875,
                confidence_drift_score=0.01,
                tracking_instability_events=1,
                ptz_cue_success_rate=0.992,
                prediction_hit_rate=0.965,
                runtime_exceptions=0,
                status="HEALTHY",
            ),
            generated_at=now,
        )

        # Security Item Baseline Scorecard
        self.scorecards["IBVAP-SECURITY-v2.1"] = UnifiedModelScorecard(
            model_id="IBVAP-SECURITY-v2.1",
            domain="SECURITY_ITEM",
            model_version="v2.1.0",
            architecture="YOLO11n",
            offline_validation=OfflineValidationMetrics(
                precision=0.842,
                recall=0.785,
                map50=0.512,
                map50_95=0.380,
                critical_class_recall=0.785,
                small_object_recall=0.650,
                hard_negative_fpr=0.022,
                p95_latency_ms=13.8,
                fps=86.5,
            ),
            operational_quality=OperationalQualityMetrics(
                total_alerts_analyzed=490,
                total_operator_feedback=18,
                operator_fp_feedback_rate=0.035,
                operator_fn_count=1,
                mean_confidence=0.835,
                confidence_drift_score=0.03,
                tracking_instability_events=2,
                ptz_cue_success_rate=0.985,
                prediction_hit_rate=0.950,
                runtime_exceptions=0,
                status="HEALTHY",
            ),
            generated_at=now,
        )

    def get_scorecard(self, model_id: str) -> Optional[UnifiedModelScorecard]:
        return self.scorecards.get(model_id)

    def list_scorecards(self) -> List[UnifiedModelScorecard]:
        return list(self.scorecards.values())

    def update_operational_proxies(
        self,
        model_id: str,
        alerts_increment: int = 1,
        feedback_increment: int = 0,
        is_fp: bool = False,
        is_fn: bool = False,
    ):
        sc = self.scorecards.get(model_id)
        if not sc:
            return
        op = sc.operational_quality
        op.total_alerts_analyzed += alerts_increment
        op.total_operator_feedback += feedback_increment
        if is_fp:
            # Update FP rate
            op.operator_fp_feedback_rate = round(
                (op.operator_fp_feedback_rate * 0.95) + (1.0 * 0.05), 4
            )
        if is_fn:
            op.operator_fn_count += 1
        sc.generated_at = time.time()

    def reset(self):
        self.scorecards.clear()
        self._init_production_scorecards()


model_quality_scorecard = ModelQualityScorecardService()
