"""
TRINETRA — Actual vs. Predicted Evaluation Engine (Phase IX)
Validates predictions against real-world camera observations, evaluates outcome states
(HIT, PARTIAL_HIT, MISS, EXPIRED, UNOBSERVABLE), and measures forecast calibration.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from backend.app.services.camera_transition_predictor import (
    camera_transition_predictor,
    TransitionPrediction
)
from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("ForecastEvaluationEngine")


class ForecastOutcome(str, Enum):
    HIT = "HIT"                       # Observed on primary predicted camera within feasible window
    PARTIAL_HIT = "PARTIAL_HIT"       # Observed on alternative hypothesis camera
    MISS = "MISS"                     # Observed on unpredicted camera
    EXPIRED_UNOBSERVED = "EXPIRED"    # ETA window expired without observation
    UNOBSERVABLE = "UNOBSERVABLE"     # Camera offline or unavailable


@dataclass
class PredictionEvaluationRecord:
    prediction_id: str
    entity_id: str
    predicted_camera: str
    actual_camera: Optional[str]
    forecast_confidence: float
    eta_expected_sec: float
    actual_arrival_sec: Optional[float]
    eta_error_sec: Optional[float]
    outcome: ForecastOutcome
    evaluated_at: float = field(default_factory=time.time)
    details: str = ""


class ForecastEvaluationEngine:
    def __init__(self):
        self.evaluation_records: List[PredictionEvaluationRecord] = []

    def evaluate_observation(
        self,
        entity_id: str,
        observed_camera: str,
        observation_timestamp: Optional[float] = None
    ) -> Optional[PredictionEvaluationRecord]:
        """
        Triggered when an entity is actually observed on observed_camera.
        Finds matching active predictions for entity_id and scores them.
        """
        now = observation_timestamp if observation_timestamp is not None else time.time()

        # Find candidate active predictions for this entity
        matching_preds = [
            p for p in camera_transition_predictor.active_predictions.values()
            if p.entity_id == entity_id and p.status == "PREDICTED"
        ]

        if not matching_preds:
            return None

        # Take the most recent prediction
        pred = matching_preds[-1]
        elapsed = max(0.1, now - pred.created_at)

        primary = pred.primary_hypothesis
        predicted_cam = primary.predicted_camera if primary else "NONE"
        conf = primary.confidence if primary else 0.5
        expected_eta = primary.typical_eta_sec if primary else 15.0

        eta_err = round(abs(elapsed - expected_eta), 2)

        # 1. Evaluate Outcome
        if observed_camera == predicted_cam:
            outcome = ForecastOutcome.HIT
            pred.status = "HIT"
            details = f"Target observed on predicted camera {observed_camera} in {elapsed:.1f}s (ETA Error: {eta_err:.1f}s)"
        else:
            # Check alternative hypotheses
            alt_cams = [alt.predicted_camera for alt in pred.alternative_hypotheses]
            if observed_camera in alt_cams:
                outcome = ForecastOutcome.PARTIAL_HIT
                pred.status = "PARTIAL_HIT"
                details = f"Target observed on alternative hypothesis camera {observed_camera} in {elapsed:.1f}s"
            else:
                outcome = ForecastOutcome.MISS
                pred.status = "MISS"
                details = f"Target observed on unexpected camera {observed_camera} (Predicted: {predicted_cam})"

        rec = PredictionEvaluationRecord(
            prediction_id=pred.prediction_id,
            entity_id=entity_id,
            predicted_camera=predicted_cam,
            actual_camera=observed_camera,
            forecast_confidence=conf,
            eta_expected_sec=expected_eta,
            actual_arrival_sec=round(elapsed, 2),
            eta_error_sec=eta_err,
            outcome=outcome,
            evaluated_at=now,
            details=details
        )

        self.evaluation_records.append(rec)

        publish_event("prediction.outcome.recorded", {
            "prediction_id": pred.prediction_id,
            "entity_id": entity_id,
            "outcome": outcome.value,
            "predicted_camera": predicted_cam,
            "actual_camera": observed_camera,
            "eta_error_sec": eta_err
        })

        logger.info(f"Evaluated prediction {pred.prediction_id}: {outcome.value} (Pred: {predicted_cam}, Actual: {observed_camera})")
        return rec

    def record_expiration(self, prediction_id: str) -> Optional[PredictionEvaluationRecord]:
        pred = camera_transition_predictor.get_prediction(prediction_id)
        if not pred:
            return None

        primary = pred.primary_hypothesis
        pred_cam = primary.predicted_camera if primary else "NONE"
        conf = primary.confidence if primary else 0.5
        expected_eta = primary.typical_eta_sec if primary else 15.0

        rec = PredictionEvaluationRecord(
            prediction_id=prediction_id,
            entity_id=pred.entity_id,
            predicted_camera=pred_cam,
            actual_camera=None,
            forecast_confidence=conf,
            eta_expected_sec=expected_eta,
            actual_arrival_sec=None,
            eta_error_sec=None,
            outcome=ForecastOutcome.EXPIRED_UNOBSERVED,
            details="Target unobserved during arrival window"
        )
        self.evaluation_records.append(rec)
        return rec

    def compute_accuracy_metrics(self) -> Dict[str, Any]:
        """Calculates Top-1, Top-3, ETA MAE, and Calibration Metrics."""
        if not self.evaluation_records:
            return {
                "total_evaluations": 0,
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "eta_mae_sec": 0.0,
                "eta_median_error_sec": 0.0,
                "eta_p95_error_sec": 0.0,
                "outcomes": {}
            }

        total = len(self.evaluation_records)
        hits = sum(1 for r in self.evaluation_records if r.outcome == ForecastOutcome.HIT)
        part = sum(1 for r in self.evaluation_records if r.outcome == ForecastOutcome.PARTIAL_HIT)
        miss = sum(1 for r in self.evaluation_records if r.outcome == ForecastOutcome.MISS)
        exp = sum(1 for r in self.evaluation_records if r.outcome == ForecastOutcome.EXPIRED_UNOBSERVED)

        top1 = hits / total
        top3 = (hits + part) / total

        errors = [r.eta_error_sec for r in self.evaluation_records if r.eta_error_sec is not None]
        mae = float(np.mean(errors)) if errors else 0.0
        p50_err = float(np.percentile(errors, 50)) if errors else 0.0
        p95_err = float(np.percentile(errors, 95)) if errors else 0.0

        return {
            "total_evaluations": total,
            "top1_accuracy": round(top1, 4),
            "top3_accuracy": round(top3, 4),
            "eta_mae_sec": round(mae, 2),
            "eta_median_error_sec": round(p50_err, 2),
            "eta_p95_error_sec": round(p95_err, 2),
            "outcomes": {
                "HIT": hits,
                "PARTIAL_HIT": part,
                "MISS": miss,
                "EXPIRED": exp
            }
        }

    def compute_calibration_table(self) -> List[Dict[str, Any]]:
        """Binned empirical success vs forecast confidence buckets."""
        buckets = [
            ("0.00-0.50", 0.0, 0.50),
            ("0.50-0.70", 0.50, 0.70),
            ("0.70-0.85", 0.70, 0.85),
            ("0.85-1.00", 0.85, 1.01)
        ]

        table = []
        for name, low, high in buckets:
            in_bucket = [r for r in self.evaluation_records if low <= r.forecast_confidence < high]
            count = len(in_bucket)
            hits = sum(1 for r in in_bucket if r.outcome == ForecastOutcome.HIT)
            partial = sum(1 for r in in_bucket if r.outcome == ForecastOutcome.PARTIAL_HIT)
            emp_rate = round((hits + partial) / count, 3) if count > 0 else 0.0

            table.append({
                "bucket": name,
                "predictions_count": count,
                "hits": hits,
                "partial_hits": partial,
                "empirical_success_rate": emp_rate
            })

        return table


forecast_evaluation_engine = ForecastEvaluationEngine()
