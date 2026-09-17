"""
TRINETRA — Model Governance & Lifecycle REST API Endpoints (Phase XII)
Exposes endpoints for model inspection, validation gates, canary control,
rollback orchestration, and operational telemetry proxies.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.services.model_registry_service import (
    model_registry_service,
    ModelMetadata,
)
from backend.app.services.model_lifecycle_state_machine import (
    model_lifecycle_engine,
    ModelLifecycleState,
)
from backend.app.services.validation_gate_service import (
    validation_gate_service,
    ValidationGateReport,
)
from backend.app.services.canary_deployment_service import (
    canary_deployment_service,
    CanaryDeploymentConfig,
)
from backend.app.services.model_rollback_service import (
    model_rollback_service,
    RollbackEvent,
)
from backend.app.services.model_monitoring_service import (
    model_monitoring_service,
    ModelHealthTelemetry,
    DriftDetectionReport,
)
from backend.app.services.dataset_governance_service import (
    dataset_governance_service,
    DatasetManifest,
)
from backend.app.services.training_orchestration_service import (
    training_orchestration_service,
    TrainingRunManifest,
)

router = APIRouter(prefix="/api/models", tags=["Model Governance"])


# --- Request/Response Schemas ---

class ValidateCandidateRequest(BaseModel):
    candidate_model_id: str
    benchmarks: Optional[Dict[str, float]] = None


class StartCanaryRequest(BaseModel):
    candidate_model_id: str
    target_nodes: List[str] = Field(default_factory=lambda: ["CAM-001", "CAM-002"])
    traffic_percentage: float = 10.0
    is_shadow_mode: bool = True
    duration_sec: float = 3600.0


class RollbackRequest(BaseModel):
    domain: str
    reason: str
    target_model_id: Optional[str] = None


class PromoteRequest(BaseModel):
    reason: str
    force_downgrade: bool = False


# --- Endpoints ---

@router.get("", response_model=List[ModelMetadata])
def list_models(
    current_user: Any = Depends(get_current_user),
):
    """Lists all registered models in the platform."""
    return list(model_registry_service.models.values())


@router.get("/production/active")
def get_active_production_models(
    current_user: Any = Depends(get_current_user),
):
    """Returns the currently active production model for each domain."""
    return {
        domain: model_registry_service.get_active_model(domain)
        for domain in ["GROUND", "AIRBORNE", "SECURITY_ITEM"]
    }


@router.get("/{model_id}", response_model=ModelMetadata)
def get_model_details(
    model_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Retrieves full lineage and metadata for a specific model."""
    model = model_registry_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return model


@router.get("/{model_id}/history")
def get_model_lifecycle_history(
    model_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Returns the immutable state transition history for a model."""
    history = model_lifecycle_engine.get_history(model_id)
    return {"model_id": model_id, "transitions": history}


@router.get("/{model_id}/health")
def get_model_health_telemetry(
    model_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Returns operational proxy telemetry and drift signals."""
    node_telemetry = [
        t for t in model_monitoring_service.list_health() if t.model_id == model_id
    ]
    model = model_registry_service.get_model(model_id)
    drift = None
    if model:
        drift = model_monitoring_service.drift_reports.get(model_id)
    return {
        "model_id": model_id,
        "nodes": node_telemetry,
        "drift_report": drift,
    }


@router.get("/{model_id}/rollback-history")
def get_rollback_history(
    model_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Returns rollback events associated with this model's domain."""
    model = model_registry_service.get_model(model_id)
    domain = model.domain if model else None
    history = model_rollback_service.get_history(domain)
    return {"model_id": model_id, "rollback_events": history}


@router.post("/validate", response_model=ValidationGateReport)
def validate_candidate_model(
    req: ValidateCandidateRequest,
    current_user: Any = Depends(get_current_user),
):
    """Runs the 13-stage validation gate on a candidate model. Requires OPERATOR or ADMIN."""
    user_roles = getattr(current_user, "roles", None)
    role_str = getattr(current_user, "role", "")
    is_authorized = (
        ("OPERATOR" in user_roles or "ADMIN" in user_roles)
        if user_roles
        else (role_str in ["OPERATOR", "ADMIN"])
    )
    if not is_authorized:
        raise HTTPException(status_code=403, detail="OPERATOR or ADMIN role required.")

    operator = getattr(current_user, "username", "OPERATOR")
    try:
        report = validation_gate_service.evaluate_candidate(
            candidate_model_id=req.candidate_model_id,
            simulated_benchmarks=req.benchmarks,
            operator=operator,
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/canary")
def start_canary_deployment(
    req: StartCanaryRequest,
    current_user: Any = Depends(get_current_user),
):
    """Launches canary deployment or shadow inference. Requires OPERATOR or ADMIN."""
    user_roles = getattr(current_user, "roles", None)
    role_str = getattr(current_user, "role", "")
    is_authorized = (
        ("OPERATOR" in user_roles or "ADMIN" in user_roles)
        if user_roles
        else (role_str in ["OPERATOR", "ADMIN"])
    )
    if not is_authorized:
        raise HTTPException(status_code=403, detail="OPERATOR or ADMIN role required.")

    operator = getattr(current_user, "username", "OPERATOR")
    ok, reason, cfg = canary_deployment_service.start_canary(
        candidate_model_id=req.candidate_model_id,
        target_nodes=req.target_nodes,
        traffic_percentage=req.traffic_percentage,
        is_shadow_mode=req.is_shadow_mode,
        duration_sec=req.duration_sec,
        operator=operator,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    return cfg


@router.post("/{model_id}/rollback")
def trigger_model_rollback(
    model_id: str,
    req: RollbackRequest,
    current_user: Any = Depends(get_current_user),
):
    """Triggers an atomic rollback to the designated last-known-good model. Requires OPERATOR or ADMIN."""
    user_roles = getattr(current_user, "roles", None)
    role_str = getattr(current_user, "role", "")
    is_authorized = (
        ("OPERATOR" in user_roles or "ADMIN" in user_roles)
        if user_roles
        else (role_str in ["OPERATOR", "ADMIN"])
    )
    if not is_authorized:
        raise HTTPException(status_code=403, detail="OPERATOR or ADMIN role required.")

    operator = getattr(current_user, "username", "OPERATOR")
    ok, reason, event = model_rollback_service.execute_rollback(
        domain=req.domain,
        reason=req.reason,
        operator=operator,
        target_model_id=req.target_model_id,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    return {"status": "SUCCESS", "event": event}


@router.post("/{model_id}/promote")
def promote_candidate_model(
    model_id: str,
    req: PromoteRequest,
    current_user: Any = Depends(get_current_user),
):
    """Promotes a candidate or canary model to active production. Requires ADMIN role."""
    user_roles = getattr(current_user, "roles", None)
    role_str = getattr(current_user, "role", "")
    is_admin = ("ADMIN" in user_roles) if user_roles else (role_str == "ADMIN")
    if not is_admin:
        raise HTTPException(status_code=403, detail="ADMIN role strictly required for production promotion.")

    operator = getattr(current_user, "username", "ADMIN")
    ok, reason = model_registry_service.promote_to_production(
        model_id=model_id,
        operator=operator,
        reason=req.reason,
        force_downgrade=req.force_downgrade,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    return {"status": "SUCCESS", "model_id": model_id, "detail": reason}


# --- Additional Datasets & Training Routes ---

datasets_router = APIRouter(prefix="/api/datasets", tags=["Dataset Governance"])


@datasets_router.get("", response_model=List[DatasetManifest])
def list_datasets(current_user: Any = Depends(get_current_user)):
    return list(dataset_governance_service.datasets.values())


training_router = APIRouter(prefix="/api/training", tags=["Training Orchestration"])


@training_router.get("/runs", response_model=List[TrainingRunManifest])
def list_training_runs(current_user: Any = Depends(get_current_user)):
    return training_orchestration_service.list_runs()
