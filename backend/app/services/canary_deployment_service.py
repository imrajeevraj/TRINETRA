"""
TRINETRA — Canary Deployment & Shadow Inference Service (Phase XII)
Manages safe canary traffic splits (e.g. 10% canary / 90% production),
isolated shadow inference execution, and automatic rollback interlocks.
"""

from __future__ import annotations
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.services.model_registry_service import (
    model_registry_service,
    ModelMetadata,
)
from backend.app.services.model_lifecycle_state_machine import (
    model_lifecycle_engine,
    ModelLifecycleState,
)

logger = logging.getLogger("CanaryDeploymentService")


class ShadowComparisonTelemetry(BaseModel):
    frame_id: str
    camera_id: str
    timestamp: float
    production_model_id: str
    candidate_model_id: str
    production_detections_count: int
    shadow_detections_count: int
    production_mean_confidence: float
    shadow_mean_confidence: float
    production_latency_ms: float
    shadow_latency_ms: float
    operational_decision_model: str  # Always the production model in shadow mode


class CanaryDeploymentConfig(BaseModel):
    deployment_id: str
    candidate_model_id: str
    production_model_id: str
    domain: str
    target_nodes: List[str]  # e.g. ["CAM-001", "CAM-002"]
    traffic_percentage: float = 10.0  # 10% canary, 90% production
    is_shadow_mode: bool = True  # In shadow mode, candidate runs in background without affecting operations
    status: str = "RUNNING"  # "INITIALIZING", "RUNNING", "PASSED", "ROLLING_BACK", "TERMINATED"
    started_at: float = Field(default_factory=time.time)
    duration_sec: float = 3600.0  # 1 hour standard canary window
    error_count: int = 0
    max_allowed_errors: int = 5
    max_latency_spike_ms: float = 35.0


class CanaryDeploymentService:
    """
    Coordinates canary deployments and shadow inference runs across the edge mesh.
    Ensures operational safety by confining candidate execution to controlled splits
    and triggering automated rollback upon telemetry degradation.
    """

    def __init__(self):
        self.active_canaries: Dict[str, CanaryDeploymentConfig] = {}
        self.shadow_telemetry: List[ShadowComparisonTelemetry] = []
        self._canary_counter = 0

    def start_canary(
        self,
        candidate_model_id: str,
        target_nodes: List[str],
        traffic_percentage: float = 10.0,
        is_shadow_mode: bool = True,
        duration_sec: float = 3600.0,
        operator: str = "CANARY_OPERATOR",
    ) -> Tuple[bool, str, Optional[CanaryDeploymentConfig]]:
        """
        Launches a canary or shadow deployment.
        Candidate must be in CANARY_READY or VALIDATED state.
        """
        candidate = model_registry_service.get_model(candidate_model_id)
        if not candidate:
            return False, f"CANDIDATE_MODEL_{candidate_model_id}_NOT_FOUND", None

        # Verify candidate lifecycle state
        curr_state = model_lifecycle_engine.get_state(candidate_model_id)
        if curr_state not in [ModelLifecycleState.CANARY_READY, ModelLifecycleState.VALIDATED]:
            return (
                False,
                f"CANARY_REJECTED: Model state is {curr_state.value if curr_state else 'None'}. "
                f"Must be CANARY_READY or VALIDATED.",
                None,
            )

        domain = candidate.domain
        prod_model = model_registry_service.get_active_model(domain)
        if not prod_model:
            return False, f"NO_ACTIVE_PRODUCTION_MODEL_FOR_DOMAIN_{domain}", None

        self._canary_counter += 1
        now = time.time()
        dep_id = f"CANARY-{domain}-{int(now)}-{self._canary_counter:03d}"

        canary_cfg = CanaryDeploymentConfig(
            deployment_id=dep_id,
            candidate_model_id=candidate_model_id,
            production_model_id=prod_model.model_id,
            domain=domain,
            target_nodes=target_nodes,
            traffic_percentage=traffic_percentage,
            is_shadow_mode=is_shadow_mode,
            status="RUNNING",
            started_at=now,
            duration_sec=duration_sec,
        )

        self.active_canaries[dep_id] = canary_cfg

        # Transition candidate to CANARY_RUNNING
        model_lifecycle_engine.transition(
            model_id=candidate_model_id,
            target_state=ModelLifecycleState.CANARY_RUNNING,
            operator=operator,
            reason=f"CANARY_STARTED_{dep_id}",
            metadata={"nodes": target_nodes, "traffic_pct": traffic_percentage},
        )

        logger.info(
            f"Launched Canary {dep_id}: Candidate {candidate_model_id} on {target_nodes} "
            f"({traffic_percentage}% split, Shadow: {is_shadow_mode})"
        )
        return True, "CANARY_DEPLOYMENT_ACTIVE", canary_cfg

    def route_frame_inference(
        self,
        camera_id: str,
        domain: str,
        frame_id: str,
    ) -> Dict[str, Any]:
        """
        Determines whether a frame runs under production, canary, or dual-shadow mode.
        """
        # Find active canary for this domain and camera
        active_canary = next(
            (
                c
                for c in self.active_canaries.values()
                if c.domain == domain and camera_id in c.target_nodes and c.status == "RUNNING"
            ),
            None,
        )

        if not active_canary:
            return {
                "active_model_id": model_registry_service.active_production.get(domain),
                "is_canary": False,
                "is_shadow": False,
            }

        # If shadow mode, production model ALWAYS drives operational decisions
        if active_canary.is_shadow_mode:
            return {
                "active_model_id": active_canary.production_model_id,
                "shadow_model_id": active_canary.candidate_model_id,
                "is_canary": False,
                "is_shadow": True,
                "canary_id": active_canary.deployment_id,
            }

        # Active traffic split (e.g. 10% canary)
        use_canary = (random.random() * 100.0) < active_canary.traffic_percentage
        chosen_id = (
            active_canary.candidate_model_id if use_canary else active_canary.production_model_id
        )

        return {
            "active_model_id": chosen_id,
            "is_canary": use_canary,
            "is_shadow": False,
            "canary_id": active_canary.deployment_id,
        }

    def record_shadow_telemetry(
        self,
        frame_id: str,
        camera_id: str,
        prod_model_id: str,
        cand_model_id: str,
        prod_detections: int,
        cand_detections: int,
        prod_conf: float,
        cand_conf: float,
        prod_lat_ms: float,
        cand_lat_ms: float,
    ) -> ShadowComparisonTelemetry:
        """
        Records side-by-side performance telemetry during shadow inference.
        """
        telemetry = ShadowComparisonTelemetry(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            production_model_id=prod_model_id,
            candidate_model_id=cand_model_id,
            production_detections_count=prod_detections,
            shadow_detections_count=cand_detections,
            production_mean_confidence=prod_conf,
            shadow_mean_confidence=cand_conf,
            production_latency_ms=prod_lat_ms,
            shadow_latency_ms=cand_lat_ms,
            operational_decision_model=prod_model_id,  # GUARANTEED: Production drives decisions
        )
        self.shadow_telemetry.append(telemetry)
        if len(self.shadow_telemetry) > 1000:
            self.shadow_telemetry.pop(0)

        # Automated rollback trigger check
        for canary in self.active_canaries.values():
            if canary.candidate_model_id == cand_model_id and canary.status == "RUNNING":
                if cand_lat_ms > canary.max_latency_spike_ms:
                    canary.error_count += 1
                    logger.warning(
                        f"Canary {canary.deployment_id} latency spike ({cand_lat_ms} ms > {canary.max_latency_spike_ms} ms). "
                        f"Error count: {canary.error_count}"
                    )
                if canary.error_count >= canary.max_allowed_errors:
                    self.trigger_canary_rollback(
                        canary.deployment_id,
                        reason="EXCESSIVE_LATENCY_SPIKES_AND_ERRORS",
                        operator="AUTO_ROLLBACK_INTERLOCK",
                    )

        return telemetry

    def complete_canary_success(
        self,
        canary_id: str,
        operator: str = "SUPERVISOR",
    ) -> Tuple[bool, str]:
        """
        Marks canary deployment as PASSED and advances candidate to CANARY_PASSED.
        """
        canary = self.active_canaries.get(canary_id)
        if not canary:
            return False, f"CANARY_{canary_id}_NOT_FOUND"

        canary.status = "PASSED"
        model_lifecycle_engine.transition(
            model_id=canary.candidate_model_id,
            target_state=ModelLifecycleState.CANARY_PASSED,
            operator=operator,
            reason=f"CANARY_EVALUATION_PASSED_{canary_id}",
        )
        logger.info(f"Canary {canary_id} successfully PASSED. Model ready for edge rollout.")
        return True, "CANARY_SUCCESSFULLY_COMPLETED"

    def trigger_canary_rollback(
        self,
        canary_id: str,
        reason: str,
        operator: str = "SYSTEM_WATCHDOG",
    ) -> Tuple[bool, str]:
        """
        Immediately aborts canary deployment, restoring 100% traffic to production.
        """
        canary = self.active_canaries.get(canary_id)
        if not canary:
            return False, f"CANARY_{canary_id}_NOT_FOUND"

        canary.status = "ROLLING_BACK"
        model_lifecycle_engine.transition(
            model_id=canary.candidate_model_id,
            target_state=ModelLifecycleState.ROLLBACK_PENDING,
            operator=operator,
            reason=f"CANARY_ABORTED: {reason}",
        )
        model_lifecycle_engine.transition(
            model_id=canary.candidate_model_id,
            target_state=ModelLifecycleState.ROLLED_BACK,
            operator=operator,
            reason="CANARY_ROLLBACK_EXECUTED",
        )
        canary.status = "TERMINATED"
        logger.warning(f"CANARY {canary_id} ROLLED BACK: {reason}")
        return True, "CANARY_ROLLED_BACK"

    def reset(self):
        self.active_canaries.clear()
        self.shadow_telemetry.clear()
        self._canary_counter = 0


canary_deployment_service = CanaryDeploymentService()
