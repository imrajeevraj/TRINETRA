"""
TRINETRA — Prediction & Forecast REST APIs (Phase IX)
Provides authenticated endpoints for querying trajectory forecasts, predictive PTZ pre-cues,
actual-vs-predicted outcomes, and accuracy metrics with RBAC.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.core.security import get_current_user, require_roles
from backend.app.services.predictive_track_engine import predictive_track_engine
from backend.app.services.camera_transition_predictor import (
    camera_transition_predictor,
    TransitionPrediction
)
from backend.app.services.predictive_ptz_engine import predictive_ptz_engine
from backend.app.services.forecast_evaluation_engine import (
    forecast_evaluation_engine,
    ForecastOutcome
)

router = APIRouter(prefix="/predictions", tags=["Predictions & Forecasting"])
ptz_router = APIRouter(prefix="/ptz", tags=["Predictive PTZ Operations"])


class PredictiveCueRequest(BaseModel):
    prediction_id: str = Field(..., max_length=32)
    entity_id: str = Field(..., max_length=64)
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    eta_sec: float = Field(15.0, ge=1.0, le=300.0)
    pan: float = Field(0.0, ge=-180.0, le=180.0)
    tilt: float = Field(-5.0, ge=-90.0, le=90.0)
    zoom: float = Field(2.0, ge=1.0, le=30.0)


class PredictionCancelRequest(BaseModel):
    reason: str = Field("OPERATOR_CANCELLED", max_length=256)


@router.get("", summary="List predictions")
def list_predictions(
    entity_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    user: Any = Depends(get_current_user)
):
    preds = list(camera_transition_predictor.active_predictions.values())
    if entity_id:
        preds = [p for p in preds if p.entity_id == entity_id]
    if status_filter:
        preds = [p for p in preds if p.status == status_filter.upper()]

    preds.sort(key=lambda x: x.created_at, reverse=True)
    preds = preds[:limit]

    return [
        {
            "prediction_id": p.prediction_id,
            "entity_id": p.entity_id,
            "current_camera": p.current_camera,
            "created_at": p.created_at,
            "expires_at": p.expires_at,
            "status": p.status,
            "method": p.method,
            "primary_hypothesis": {
                "predicted_camera": p.primary_hypothesis.predicted_camera,
                "predicted_zone": p.primary_hypothesis.predicted_zone,
                "eta_min_sec": p.primary_hypothesis.eta_min_sec,
                "eta_max_sec": p.primary_hypothesis.eta_max_sec,
                "typical_eta_sec": p.primary_hypothesis.typical_eta_sec,
                "confidence": p.primary_hypothesis.confidence,
                "reason": p.primary_hypothesis.reason
            } if p.primary_hypothesis else None,
            "alternative_hypotheses_count": len(p.alternative_hypotheses)
        }
        for p in preds
    ]


@router.get("/metrics", summary="Get forecast accuracy metrics")
def get_forecast_metrics(user: Any = Depends(get_current_user)):
    metrics = forecast_evaluation_engine.compute_accuracy_metrics()
    calibration = forecast_evaluation_engine.compute_calibration_table()
    return {
        "metrics": metrics,
        "calibration_table": calibration
    }


@router.get("/{prediction_id}", summary="Get prediction details")
def get_prediction_detail(
    prediction_id: str,
    user: Any = Depends(get_current_user)
):
    pred = camera_transition_predictor.get_prediction(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    # Find evaluation record if exists
    eval_rec = next((r for r in forecast_evaluation_engine.evaluation_records if r.prediction_id == prediction_id), None)

    return {
        "prediction_id": pred.prediction_id,
        "entity_id": pred.entity_id,
        "current_camera": pred.current_camera,
        "created_at": pred.created_at,
        "expires_at": pred.expires_at,
        "status": pred.status,
        "method": pred.method,
        "primary_hypothesis": pred.primary_hypothesis.__dict__ if pred.primary_hypothesis else None,
        "alternative_hypotheses": [alt.__dict__ for alt in pred.alternative_hypotheses],
        "outcome": {
            "outcome": eval_rec.outcome.value,
            "actual_camera": eval_rec.actual_camera,
            "actual_arrival_sec": eval_rec.actual_arrival_sec,
            "eta_error_sec": eval_rec.eta_error_sec,
            "details": eval_rec.details
        } if eval_rec else None
    }


@router.get("/{prediction_id}/outcome", summary="Get prediction outcome")
def get_prediction_outcome(
    prediction_id: str,
    user: Any = Depends(get_current_user)
):
    eval_rec = next((r for r in forecast_evaluation_engine.evaluation_records if r.prediction_id == prediction_id), None)
    if not eval_rec:
        raise HTTPException(status_code=404, detail=f"Outcome for prediction {prediction_id} not recorded yet")

    return {
        "prediction_id": eval_rec.prediction_id,
        "entity_id": eval_rec.entity_id,
        "outcome": eval_rec.outcome.value,
        "predicted_camera": eval_rec.predicted_camera,
        "actual_camera": eval_rec.actual_camera,
        "expected_eta_sec": eval_rec.eta_expected_sec,
        "actual_arrival_sec": eval_rec.actual_arrival_sec,
        "eta_error_sec": eval_rec.eta_error_sec,
        "evaluated_at": eval_rec.evaluated_at,
        "details": eval_rec.details
    }


@router.post("/{prediction_id}/cancel", summary="Cancel active prediction")
def cancel_prediction(
    prediction_id: str,
    req: PredictionCancelRequest,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    pred = camera_transition_predictor.get_prediction(prediction_id)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    pred.status = "CANCELLED"
    # If primary camera had pre-cue, cancel it
    if pred.primary_hypothesis:
        predictive_ptz_engine.cancel_pre_cue(pred.primary_hypothesis.predicted_camera, reason=req.reason)

    return {
        "prediction_id": prediction_id,
        "status": "CANCELLED",
        "reason": req.reason,
        "cancelled_by": getattr(user, "username", "operator")
    }


@ptz_router.post("/{camera_id}/predictive-cue", summary="Dispatch predictive PTZ pre-cue")
def dispatch_predictive_cue(
    camera_id: str,
    req: PredictiveCueRequest,
    user: Any = Depends(require_roles("ADMIN", "OPERATOR"))
):
    accepted, reason, action = predictive_ptz_engine.evaluate_and_precue(
        prediction_id=req.prediction_id,
        entity_id=req.entity_id,
        target_camera=camera_id,
        confidence=req.confidence,
        eta_sec=req.eta_sec,
        predicted_pan=req.pan,
        predicted_tilt=req.tilt,
        predicted_zoom=req.zoom
    )

    if not accepted:
        raise HTTPException(status_code=400, detail=f"Predictive pre-cue rejected: {reason}")

    return {
        "status": "ACCEPTED",
        "action_id": action.action_id if action else None,
        "target_camera": camera_id,
        "state": action.state.value if action else None,
        "reason": reason
    }

