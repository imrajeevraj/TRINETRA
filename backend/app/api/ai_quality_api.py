"""
TRINETRA — AI Quality Intelligence & Continuous Edge Learning REST API (Phase XIII)
Exposes endpoints for operator feedback ingestion, multi-reviewer consensus,
hard-case discovery, failure clustering, drift detection, active learning queues,
governed retraining recommendations, shadow canary comparisons, and poisoning defense.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from backend.app.core.security import get_current_user
from backend.app.models.user import User

from backend.app.services.operator_feedback_service import (
    operator_feedback_service,
    OperatorFeedbackRecord,
    FeedbackDisposition,
    FeedbackReviewStatus,
)
from backend.app.services.hard_case_miner import (
    hard_case_miner,
    HardCaseCandidate,
    HardCaseTrigger,
)
from backend.app.services.active_learning_queue import (
    active_learning_queue,
    ActiveLearningItem,
)
from backend.app.services.failure_clustering_service import (
    failure_clustering_service,
    FailureClusterRecord,
)
from backend.app.services.camera_quality_analytics import (
    camera_quality_analytics,
    CameraOperationalQualityIndicators,
    ModelCameraMatrixCell,
)
from backend.app.services.drift_intelligence_service import (
    drift_intelligence_service,
    WindowedDriftReport,
)
from backend.app.services.model_quality_scorecard import (
    model_quality_scorecard,
    UnifiedModelScorecard,
)
from backend.app.services.retraining_recommendation_engine import (
    retraining_recommendation_engine,
    RetrainingRecommendation,
)
from backend.app.services.training_curation_service import (
    training_curation_service,
    CuratedDatasetManifest,
)
from backend.app.services.champion_challenger_engine import (
    champion_challenger_engine,
    CanaryComparisonReport,
)
from backend.app.services.edge_feedback_sync import (
    edge_feedback_sync,
)
from backend.app.services.data_poisoning_defense import (
    data_poisoning_defense,
    PoisoningAlert,
)

feedback_router = APIRouter(prefix="/api/ai-feedback", tags=["AI Feedback Governance"])
quality_router = APIRouter(prefix="/api/ai-quality", tags=["AI Quality Intelligence"])
learning_router = APIRouter(prefix="/api/ai-learning", tags=["Continuous Edge Learning"])


# ============================================================================
# Schemas
# ============================================================================

class SubmitFeedbackRequest(BaseModel):
    camera_id: str
    frame_id: str
    model_id: str
    model_version: str
    disposition: str
    confidence: float
    reason: str
    event_id: Optional[str] = None
    track_id: Optional[int] = None
    global_entity_id: Optional[str] = None
    evidence_reference: Optional[str] = None
    source_frame_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ReviewFeedbackRequest(BaseModel):
    agrees_with_operator: bool
    assigned_disposition: Optional[str] = None
    corrected_class: Optional[str] = None
    notes: str = ""


class CurateDatasetRequest(BaseModel):
    dataset_id: str
    domain: str
    parent_dataset: str
    failure_hypothesis: str
    target_clusters: List[str]
    sample_hashes_and_labels: Dict[str, str]


class CanaryExperimentRequest(BaseModel):
    domain: str
    challenger_model_id: str
    champion_model_id: Optional[str] = None
    evaluation_period_hours: float = 24.0
    simulated_metrics: Optional[Dict[str, Any]] = None


class EvaluateDriftRequest(BaseModel):
    domain: str
    model_id: str
    operator_rejections_delta: float = 0.0
    affected_cameras: Optional[List[str]] = None


# ============================================================================
# Helper Functions
# ============================================================================

def _verify_roles(user: Any, allowed_roles: List[str]) -> str:
    user_roles = getattr(user, "roles", None)
    role_str = getattr(user, "role", "")
    username = getattr(user, "username", "UNKNOWN_OPERATOR")

    if user_roles:
        if any(r.upper() in allowed_roles for r in user_roles):
            return username
    elif role_str and role_str.upper() in allowed_roles:
        return username

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Access denied. Requires one of roles: {allowed_roles}",
    )


# ============================================================================
# 1. Operator Feedback Governance Endpoints
# ============================================================================

@feedback_router.post("", response_model=OperatorFeedbackRecord)
def submit_feedback(
    req: SubmitFeedbackRequest,
    current_user: Any = Depends(get_current_user),
):
    """
    Submits structured, immutable operator feedback for an inference event.
    Evaluates anti-poisoning defenses and triggers hard-case discovery.
    """
    operator_id = _verify_roles(current_user, ["OPERATOR", "ADMIN", "ML_ENGINEER", "SUPERVISOR"])

    # 1. Inspect for data poisoning
    sample_entry = {
        "sample_id": f"SMP-{req.camera_id}-{req.frame_id}",
        "frame_sha": req.source_frame_hash or "",
        "label": req.disposition,
        "camera_id": req.camera_id,
        "reviewer_id": operator_id,
    }
    is_clean, alerts = data_poisoning_defense.inspect_sample_batch([sample_entry])
    if not is_clean:
        leakage_alert = any(a.vector_type == "BENCHMARK_LEAKAGE" for a in alerts)
        if leakage_alert:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="BENCHMARK_LEAKAGE_REJECTED: Submitted frame matches frozen benchmark IBVAP-GT-v1.0!",
            )

    # 2. Record feedback
    ok, msg, record = operator_feedback_service.submit_feedback(
        camera_id=req.camera_id,
        frame_id=req.frame_id,
        model_id=req.model_id,
        model_version=req.model_version,
        operator_id=operator_id,
        disposition=req.disposition,
        confidence=req.confidence,
        reason=req.reason,
        source_frame_hash=req.source_frame_hash,
        event_id=req.event_id,
        track_id=req.track_id,
        global_entity_id=req.global_entity_id,
        evidence_reference=req.evidence_reference,
        metadata=req.metadata,
    )
    if not ok or not record:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    # 3. Trigger hard-case mining if disposition is an error
    disp_upper = req.disposition.upper()
    if disp_upper in ["FALSE_POSITIVE", "FALSE_NEGATIVE", "MISCLASSIFICATION"]:
        hard_case = hard_case_miner.evaluate_detection(
            camera_id=req.camera_id,
            frame_id=req.frame_id,
            model_id=req.model_id,
            model_version=req.model_version,
            confidence=req.confidence,
            class_name=req.metadata.get("class_name", "person") if req.metadata else "person",
            is_operator_fp=(disp_upper == "FALSE_POSITIVE"),
            is_operator_rejected=True,
            source_frame_hash=req.source_frame_hash,
        )
        if hard_case:
            domain = "GROUND"
            if "airborne" in req.model_id.lower():
                domain = "AIRBORNE"
            elif "security" in req.model_id.lower():
                domain = "SECURITY_ITEM"

            active_learning_queue.enqueue_sample(
                sample_id=hard_case.candidate_id,
                model_domain=domain,
                camera_id=req.camera_id,
                failure_type=disp_upper,
                confidence=req.confidence,
                is_disputed=False,
                class_name=req.metadata.get("class_name", "person") if req.metadata else "person",
            )

            # Auto-assign to failure cluster
            failure_clustering_service.assign_sample_to_cluster(
                sample_id=hard_case.candidate_id,
                domain=domain,
                failure_type=disp_upper,
                camera_id=req.camera_id,
                model_id=req.model_id,
            )

    return record


@feedback_router.get("", response_model=List[OperatorFeedbackRecord])
def list_feedback(
    camera_id: Optional[str] = Query(None),
    disposition: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user: Any = Depends(get_current_user),
):
    """Lists operator feedback records with optional filters."""
    return operator_feedback_service.list_feedback(
        camera_id=camera_id,
        disposition=disposition,
        review_status=review_status,
        limit=limit,
    )


@feedback_router.get("/{feedback_id}", response_model=OperatorFeedbackRecord)
def get_feedback(
    feedback_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Retrieves full details of a specific feedback record."""
    rec = operator_feedback_service.get_feedback(feedback_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Feedback {feedback_id} not found.")
    return rec


@feedback_router.post("/{feedback_id}/review", response_model=OperatorFeedbackRecord)
def review_feedback(
    feedback_id: str,
    req: ReviewFeedbackRequest,
    current_user: Any = Depends(get_current_user),
):
    """
    Submits a review for an operator feedback record.
    Manages multi-reviewer consensus and transitions into VALIDATED, REJECTED, or DISPUTED.
    """
    username = _verify_roles(current_user, ["OPERATOR", "ML_ENGINEER", "ADMIN"])
    role_str = getattr(current_user, "role", "REVIEWER").upper()
    if role_str not in ["OPERATOR", "ML_ENGINEER", "ADMINISTRATOR", "ADMIN", "REVIEWER"]:
        role_str = "REVIEWER"
    if role_str == "ADMIN":
        role_str = "ADMINISTRATOR"

    ok, msg, record = operator_feedback_service.review_feedback(
        feedback_id=feedback_id,
        reviewer_id=username,
        reviewer_role=role_str,
        agrees_with_operator=req.agrees_with_operator,
        assigned_disposition=req.assigned_disposition,
        corrected_class=req.corrected_class,
        notes=req.notes,
    )
    if not ok or not record:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return record


@feedback_router.post("/{feedback_id}/gate-training")
def gate_training_eligibility(
    feedback_id: str,
    current_user: Any = Depends(get_current_user),
):
    """
    Gates validated feedback into TRAINING_ELIGIBLE status for curation.
    Requires ML_ENGINEER or ADMIN role.
    """
    username = _verify_roles(current_user, ["ML_ENGINEER", "ADMIN"])
    ok, msg = operator_feedback_service.mark_training_eligible(feedback_id, operator_id=username)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"feedback_id": feedback_id, "status": "TRAINING_ELIGIBLE", "message": msg}


# ============================================================================
# 2. AI Quality & Observability Endpoints
# ============================================================================

@quality_router.get("", response_model=List[UnifiedModelScorecard])
def get_system_scorecard(
    current_user: Any = Depends(get_current_user),
):
    """Returns platform-wide Unified Model Scorecards with strict benchmark/proxy separation."""
    return model_quality_scorecard.list_scorecards()


@quality_router.get("/cameras")
def get_camera_quality_matrix(
    current_user: Any = Depends(get_current_user),
):
    """Returns the Model x Camera Matrix and Camera Operational Quality Indicators."""
    matrix = camera_quality_analytics.get_model_camera_matrix()
    indicators = camera_quality_analytics.camera_indicators
    return {
        "model_camera_matrix": matrix,
        "camera_indicators": indicators,
        "note": "Operational quality indicators are telemetry proxies, NOT ground truth accuracy.",
    }


@quality_router.get("/failures", response_model=List[FailureClusterRecord])
def list_failure_clusters(
    domain: Optional[str] = Query(None),
    current_user: Any = Depends(get_current_user),
):
    """Lists operational failure clusters across Ground, Airborne, and Security domains."""
    return failure_clustering_service.list_clusters(domain=domain)


@quality_router.get("/drift")
def get_drift_reports(
    current_user: Any = Depends(get_current_user),
):
    """Returns active windowed drift detection reports across all perception domains."""
    return {
        domain: drift_intelligence_service.get_latest_report(domain)
        for domain in ["GROUND", "AIRBORNE", "SECURITY_ITEM"]
    }


@quality_router.post("/drift/evaluate", response_model=WindowedDriftReport)
def evaluate_drift_manually(
    req: EvaluateDriftRequest,
    current_user: Any = Depends(get_current_user),
):
    """
    Triggers an on-demand windowed drift evaluation for a model domain.
    Requires OPERATOR, ML_ENGINEER, or ADMIN role.
    """
    _verify_roles(current_user, ["OPERATOR", "ML_ENGINEER", "ADMIN"])
    rep = drift_intelligence_service.evaluate_windowed_drift(
        domain=req.domain,
        model_id=req.model_id,
        operator_rejections_delta=req.operator_rejections_delta,
        affected_cameras=req.affected_cameras,
    )
    return rep


@quality_router.get("/models/{model_id}", response_model=UnifiedModelScorecard)
def get_model_scorecard(
    model_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Retrieves the unified scorecard for a specific model."""
    sc = model_quality_scorecard.get_scorecard(model_id)
    if not sc:
        raise HTTPException(status_code=404, detail=f"Scorecard for model {model_id} not found.")
    return sc


# ============================================================================
# 3. Continuous Edge Learning & Governance Endpoints
# ============================================================================

@learning_router.get("/queue", response_model=List[ActiveLearningItem])
def list_active_learning_queue(
    domain: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    current_user: Any = Depends(get_current_user),
):
    """Returns prioritized edge cases from the Active Learning Queue."""
    return active_learning_queue.list_queue(model_domain=domain, limit=limit)


@learning_router.post("/queue/dequeue", response_model=Optional[ActiveLearningItem])
def dequeue_highest_priority(
    domain: Optional[str] = Query(None),
    current_user: Any = Depends(get_current_user),
):
    """Dequeues the highest priority candidate for annotation."""
    _verify_roles(current_user, ["OPERATOR", "ML_ENGINEER", "ADMIN"])
    item = active_learning_queue.dequeue_highest_priority(model_domain=domain)
    if not item:
        raise HTTPException(status_code=404, detail="Queue is currently empty.")
    return item


@learning_router.get("/recommendations", response_model=List[RetrainingRecommendation])
def list_retraining_recommendations(
    current_user: Any = Depends(get_current_user),
):
    """Lists governed retraining recommendations with non-autonomous advisory status."""
    return retraining_recommendation_engine.list_recommendations()


@learning_router.post("/recommendations/evaluate", response_model=RetrainingRecommendation)
def trigger_recommendation_evaluation(
    domain: str = Query(..., description="Domain to evaluate: GROUND, AIRBORNE, or SECURITY_ITEM"),
    active_model_id: str = Query(..., description="Current active production model ID"),
    current_user: Any = Depends(get_current_user),
):
    """Generates an advisory retraining recommendation based on live telemetry."""
    _verify_roles(current_user, ["OPERATOR", "ML_ENGINEER", "ADMIN"])
    rec = retraining_recommendation_engine.evaluate_retraining_needs(
        domain=domain,
        active_model_id=active_model_id,
    )
    return rec


@learning_router.get("/datasets", response_model=List[CuratedDatasetManifest])
def list_curated_datasets(
    domain: Optional[str] = Query(None),
    current_user: Any = Depends(get_current_user),
):
    """Lists curated dataset manifests with benchmark isolation guarantees."""
    return training_curation_service.list_datasets(domain=domain)


@learning_router.post("/datasets/curate", response_model=CuratedDatasetManifest)
def curate_dataset(
    req: CurateDatasetRequest,
    current_user: Any = Depends(get_current_user),
):
    """
    Curates a failure-driven training dataset candidate.
    Enforces strict benchmark isolation and rejects quarantined samples with 400.
    Requires ML_ENGINEER or ADMIN role.
    """
    username = _verify_roles(current_user, ["ML_ENGINEER", "ADMIN"])
    ok, msg, manifest = training_curation_service.curate_dataset_candidate(
        dataset_id=req.dataset_id,
        domain=req.domain,
        parent_dataset=req.parent_dataset,
        failure_hypothesis=req.failure_hypothesis,
        target_clusters=req.target_clusters,
        sample_hashes_and_labels=req.sample_hashes_and_labels,
        curated_by=username,
    )
    if not ok or not manifest:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return manifest


@learning_router.get("/experiments", response_model=List[CanaryComparisonReport])
def list_canary_experiments(
    domain: Optional[str] = Query(None),
    current_user: Any = Depends(get_current_user),
):
    """Lists Canary/Shadow comparison reports between champions and challengers."""
    return champion_challenger_engine.list_reports(domain=domain)


@learning_router.post("/experiments/compare", response_model=CanaryComparisonReport)
def run_canary_comparison(
    req: CanaryExperimentRequest,
    current_user: Any = Depends(get_current_user),
):
    """
    Compiles a shadow comparison session between champion and challenger.
    Requires ML_ENGINEER or ADMIN role.
    """
    _verify_roles(current_user, ["ML_ENGINEER", "ADMIN"])
    rep = champion_challenger_engine.compile_canary_comparison(
        domain=req.domain,
        challenger_model_id=req.challenger_model_id,
        champion_model_id=req.champion_model_id,
        evaluation_period_hours=req.evaluation_period_hours,
        simulated_metrics=req.simulated_metrics,
    )
    return rep


@learning_router.get("/poisoning/alerts", response_model=List[PoisoningAlert])
def list_poisoning_alerts(
    current_user: Any = Depends(get_current_user),
):
    """Returns active alerts generated by data poisoning and anomaly defenses."""
    return data_poisoning_defense.list_alerts()


@learning_router.post("/sync/nodes")
def sync_all_edge_outboxes(
    current_user: Any = Depends(get_current_user),
):
    """Synchronizes feedback outboxes across all connected edge mesh nodes."""
    _verify_roles(current_user, ["OPERATOR", "ML_ENGINEER", "ADMIN"])
    results = edge_feedback_sync.synchronize_all_nodes()
    return {"status": "SUCCESS", "nodes_synced": results}


@learning_router.post("/sync/node/{camera_id}")
def sync_single_edge_outbox(
    camera_id: str,
    current_user: Any = Depends(get_current_user),
):
    """Synchronizes feedback outbox for a single edge camera node."""
    _verify_roles(current_user, ["OPERATOR", "ML_ENGINEER", "ADMIN"])
    ok, msg, summary = edge_feedback_sync.synchronize_node_outbox(camera_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
    return {"status": "SUCCESS", "message": msg, "summary": summary}
