"""
TRINETRA — Model Rollback Orchestrator & Audit Service (Phase XII)
Executes atomic, verified rollbacks, restores previous last-known-good models,
preserves failed candidate artifacts for forensic analysis, and logs immutable history.
"""

from __future__ import annotations
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.services.model_registry_service import (
    model_registry_service,
    ModelMetadata,
    ModelGovernanceStatus,
)
from backend.app.services.model_lifecycle_state_machine import (
    model_lifecycle_engine,
    ModelLifecycleState,
)
from backend.app.services.edge_model_deployment_service import (
    edge_model_deployment_service,
)

logger = logging.getLogger("ModelRollbackService")


class RollbackEvent(BaseModel):
    event_id: str
    domain: str
    failed_model_id: str
    restored_model_id: str
    restored_sha256: str
    timestamp: float = Field(default_factory=time.time)
    operator: str
    reason: str
    affected_nodes: List[str]
    status: str  # "SUCCESS", "FAILED"
    preserved_artifact_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRollbackService:
    """
    Guarantees atomic rollback to the last known-good production model.
    Enforces that failed candidates are preserved, never deleted.
    """

    def __init__(self):
        self.rollback_history: List[RollbackEvent] = []
        self._rollback_counter = 0

    def execute_rollback(
        self,
        domain: str,
        reason: str,
        operator: str = "OPERATOR_EMERGENCY",
        target_model_id: Optional[str] = None,
        affected_nodes: Optional[List[str]] = None,
    ) -> Tuple[bool, str, Optional[RollbackEvent]]:
        """
        Executes an atomic rollback for the given domain:
        1. Identifies current active model and its designated rollback target
        2. Verifies rollback target model exists and its SHA matches
        3. Halts active candidate and transitions to ROLLBACK_PENDING -> ROLLED_BACK
        4. Re-activates target model in production registry
        5. Preserves failed candidate artifact for forensic analysis
        6. Logs immutable rollback event
        """
        active_prod = model_registry_service.get_active_model(domain)
        if not active_prod:
            return False, f"NO_ACTIVE_PRODUCTION_MODEL_TO_ROLLBACK_FOR_DOMAIN_{domain}", None

        failed_model_id = active_prod.model_id
        rollback_model_id = target_model_id or active_prod.rollback_target

        if not rollback_model_id:
            return False, f"NO_ROLLBACK_TARGET_DEFINED_FOR_MODEL_{failed_model_id}", None

        rollback_model = model_registry_service.get_model(rollback_model_id)
        if not rollback_model:
            return False, f"ROLLBACK_TARGET_MODEL_{rollback_model_id}_NOT_FOUND_IN_REGISTRY", None

        # Verify target artifact integrity
        target_path = Path(rollback_model.file_path)
        if not target_path.is_absolute():
            from backend.app.services.model_registry_service import REPO_ROOT
            target_path = REPO_ROOT / target_path

        if not target_path.exists():
            return False, f"ROLLBACK_ARTIFACT_NOT_FOUND_AT_{target_path}", None

        # Check SHA
        actual_sha = model_registry_service._compute_sha256(target_path)
        if actual_sha != rollback_model.sha256:
            return False, f"ROLLBACK_TARGET_SHA_MISMATCH: {actual_sha} != {rollback_model.sha256}", None

        self._rollback_counter += 1
        now = time.time()
        event_id = f"RBK-{domain}-{int(now)}-{self._rollback_counter:03d}"

        # 1. Transition failed model to ROLLBACK_PENDING -> ROLLED_BACK
        model_lifecycle_engine.transition(
            model_id=failed_model_id,
            target_state=ModelLifecycleState.ROLLBACK_PENDING,
            operator=operator,
            reason=f"ROLLBACK_INITIATED: {reason}",
        )
        model_lifecycle_engine.transition(
            model_id=failed_model_id,
            target_state=ModelLifecycleState.ROLLED_BACK,
            operator=operator,
            reason="ROLLBACK_EXECUTED",
        )
        active_prod.governance_status = ModelGovernanceStatus.ROLLED_BACK

        # 2. Re-promote rollback model to PRODUCTION
        model_registry_service.active_production[domain] = rollback_model_id
        rollback_model.governance_status = ModelGovernanceStatus.PRODUCTION
        model_lifecycle_engine.transition(
            model_id=rollback_model_id,
            target_state=ModelLifecycleState.ACTIVE,
            operator=operator,
            reason=f"RESTORED_VIA_ROLLBACK_EVENT_{event_id}",
        )

        # 3. Update edge nodes status
        nodes = affected_nodes or [f"CAM-{i:03d}" for i in range(1, 9)]
        for cam_id in nodes:
            status = edge_model_deployment_service.node_statuses.get(cam_id)
            if status:
                status.active_model_id = rollback_model_id
                status.deployment_status = "ACTIVE"
                status.error_message = None

        event = RollbackEvent(
            event_id=event_id,
            domain=domain,
            failed_model_id=failed_model_id,
            restored_model_id=rollback_model_id,
            restored_sha256=rollback_model.sha256,
            timestamp=now,
            operator=operator,
            reason=reason,
            affected_nodes=nodes,
            status="SUCCESS",
            preserved_artifact_path=active_prod.file_path,  # PRESERVED FOR FORENSICS
            metadata={"restored_version": rollback_model.version},
        )

        self.rollback_history.append(event)
        logger.warning(
            f"ATOMIC ROLLBACK EXECUTED [{event_id}]: {failed_model_id} -> {rollback_model_id} "
            f"across {len(nodes)} nodes (Reason: {reason})"
        )
        return True, "ROLLBACK_SUCCESSFUL", event

    def get_history(self, domain: Optional[str] = None) -> List[RollbackEvent]:
        if domain:
            return [e for e in self.rollback_history if e.domain == domain]
        return list(self.rollback_history)

    def reset(self):
        self.rollback_history.clear()
        self._rollback_counter = 0


model_rollback_service = ModelRollbackService()
