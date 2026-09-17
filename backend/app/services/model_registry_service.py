"""
TRINETRA — Model Registry & Versioning Service (Phase XII)
Manages immutable model identities, lineage trees, parent-child relationships,
cryptographic SHA-256 validation, and downgrade prevention.
"""

from __future__ import annotations
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.services.model_lifecycle_state_machine import (
    model_lifecycle_engine,
    ModelLifecycleState,
)

logger = logging.getLogger("ModelRegistryService")

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPO_ROOT / "configs" / "production_model_manifest.json"


class ModelGovernanceStatus(str):
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    CANARY = "CANARY"
    PRODUCTION = "PRODUCTION"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    ARCHIVED = "ARCHIVED"


class ModelMetadata(BaseModel):
    model_id: str
    model_name: str
    domain: str  # "GROUND", "AIRBORNE", "SECURITY_ITEM", "THERMAL"
    version: str
    parent_model_id: Optional[str] = None
    architecture: str
    task: str
    classes: Dict[str, str]
    file_path: str
    sha256: str
    dataset_id: str
    training_run_id: str
    governance_status: str = ModelGovernanceStatus.EXPERIMENTAL
    lifecycle_state: ModelLifecycleState = ModelLifecycleState.COLLECTED
    rollback_target: Optional[str] = None
    input_dimensions: List[int] = Field(default_factory=lambda: [640, 640, 3])
    created_at: float = Field(default_factory=time.time)
    promoted_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRegistryService:
    """
    Central repository for immutable model records, version trees, and checksums.
    Enforces that model artifacts are never overwritten in-place.
    """

    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self.active_production: Dict[str, str] = {}  # domain -> model_id
        self._load_frozen_production_models()

    def _compute_sha256(self, filepath: Path) -> str:
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest().upper()

    def _load_frozen_production_models(self):
        """Loads and verifies the frozen baseline production models from manifest."""
        if not MANIFEST_PATH.exists():
            logger.warning(f"Production manifest not found at {MANIFEST_PATH}")
            return

        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            manifest_models = data.get("models", {})
            domain_map = {
                "ground": "GROUND",
                "airborne": "AIRBORNE",
                "security_item": "SECURITY_ITEM",
            }

            for key, m in manifest_models.items():
                domain = domain_map.get(key, key.upper())
                model_id = m["model_id"]
                file_rel = m["file_path"]
                file_abs = REPO_ROOT / file_rel

                # Verify file and hash
                if file_abs.exists():
                    actual_sha = self._compute_sha256(file_abs)
                    if actual_sha != m["sha256"]:
                        logger.error(
                            f"FATAL: Production model {model_id} SHA mismatch: {actual_sha} != {m['sha256']}"
                        )
                        continue

                record = ModelMetadata(
                    model_id=model_id,
                    model_name=m["model_name"],
                    domain=domain,
                    version=m["version"],
                    parent_model_id=m.get("rollback_version"),
                    architecture=m["architecture"],
                    task=m["task"],
                    classes=m["classes"],
                    file_path=str(file_rel),
                    sha256=m["sha256"],
                    dataset_id=m["dataset_version"],
                    training_run_id=f"RUN-{model_id}",
                    governance_status=ModelGovernanceStatus.PRODUCTION,
                    lifecycle_state=ModelLifecycleState.ACTIVE,
                    rollback_target=m.get("rollback_version"),
                    input_dimensions=m.get("input_dimensions", [640, 640, 3]),
                    promoted_at=time.time(),
                )

                self.models[model_id] = record
                self.active_production[domain] = model_id
                model_lifecycle_engine.register_model(
                    model_id=model_id,
                    initial_state=ModelLifecycleState.ACTIVE,
                    operator="SYSTEM_INIT",
                    reason="PRODUCTION_FREEZE_BASELINE",
                )
                logger.info(f"Loaded frozen production model {model_id} for domain {domain}")

        except Exception as e:
            logger.error(f"Error loading frozen production models: {e}")

    def register_candidate(
        self,
        model_id: str,
        model_name: str,
        domain: str,
        version: str,
        architecture: str,
        task: str,
        classes: Dict[str, str],
        artifact_path: str,
        dataset_id: str,
        training_run_id: str,
        parent_model_id: Optional[str] = None,
        rollback_target: Optional[str] = None,
        input_dimensions: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, Optional[ModelMetadata]]:
        """
        Registers a new candidate model.
        Prevents in-place overwrites and enforces SHA-256 computation.
        """
        if model_id in self.models:
            return False, f"MODEL_ID_{model_id}_ALREADY_EXISTS_IMMUTABLE", None

        # Verify artifact exists
        art_path = Path(artifact_path)
        if not art_path.is_absolute():
            art_path = REPO_ROOT / art_path

        if not art_path.exists():
            return False, f"ARTIFACT_NOT_FOUND_AT_{art_path}", None

        # Verify parent model exists if specified
        if parent_model_id and parent_model_id not in self.models:
            return False, f"PARENT_MODEL_{parent_model_id}_NOT_FOUND", None

        # Compute artifact hash
        sha256_hash = self._compute_sha256(art_path)

        # Default rollback target to currently active production model for domain if not provided
        effective_rollback = rollback_target or self.active_production.get(domain)

        record = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            domain=domain,
            version=version,
            parent_model_id=parent_model_id,
            architecture=architecture,
            task=task,
            classes=classes,
            file_path=str(artifact_path),
            sha256=sha256_hash,
            dataset_id=dataset_id,
            training_run_id=training_run_id,
            governance_status=ModelGovernanceStatus.CANDIDATE,
            lifecycle_state=ModelLifecycleState.TRAINED,
            rollback_target=effective_rollback,
            input_dimensions=input_dimensions or [640, 640, 3],
            metadata=metadata or {},
        )

        self.models[model_id] = record
        model_lifecycle_engine.register_model(
            model_id=model_id,
            initial_state=ModelLifecycleState.TRAINED,
            operator="TRAINING_ORCHESTRATOR",
            reason="CANDIDATE_REGISTRATION",
        )

        logger.info(f"Registered candidate model {model_id} ({architecture}) with SHA {sha256_hash[:16]}...")
        return True, "CANDIDATE_REGISTERED", record

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self.models.get(model_id)

    def get_active_model(self, domain: str) -> Optional[ModelMetadata]:
        model_id = self.active_production.get(domain)
        if model_id:
            return self.models.get(model_id)
        return None

    def promote_to_production(
        self,
        model_id: str,
        operator: str,
        reason: str,
        force_downgrade: bool = False,
    ) -> Tuple[bool, str]:
        """
        Promotes a candidate or canary model to active production.
        Enforces downgrade protection and valid lifecycle transition.
        """
        candidate = self.models.get(model_id)
        if not candidate:
            return False, f"MODEL_{model_id}_NOT_FOUND"

        domain = candidate.domain
        current_active_id = self.active_production.get(domain)
        current_active = self.models.get(current_active_id) if current_active_id else None

        # Downgrade Protection Check
        if current_active and not force_downgrade:
            # Semantic version check (e.g. v2.1.0 vs v2.0.0)
            if candidate.version < current_active.version:
                return (
                    False,
                    f"DOWNGRADE_ATTACK_REJECTED: Candidate version {candidate.version} is older "
                    f"than active production version {current_active.version}. Requires authorized rollback.",
                )

        # Transition lifecycle state
        ok, r_trans, _ = model_lifecycle_engine.transition(
            model_id=model_id,
            target_state=ModelLifecycleState.ACTIVE,
            operator=operator,
            reason=reason,
        )
        if not ok:
            return False, f"PROMOTION_BLOCKED: {r_trans}"

        # If previous model existed, transition it to ARCHIVED or ROLLED_BACK
        if current_active:
            current_active.governance_status = ModelGovernanceStatus.ARCHIVED
            model_lifecycle_engine.transition(
                model_id=current_active.model_id,
                target_state=ModelLifecycleState.ARCHIVED,
                operator=operator,
                reason=f"SUPERSEDED_BY_{model_id}",
            )

        candidate.governance_status = ModelGovernanceStatus.PRODUCTION
        candidate.promoted_at = time.time()
        self.active_production[domain] = model_id

        logger.info(f"PROMOTED model {model_id} to PRODUCTION for domain {domain} by {operator}")
        return True, "PROMOTION_SUCCESSFUL"

    def reset(self):
        self.models.clear()
        self.active_production.clear()
        self._load_frozen_production_models()


model_registry_service = ModelRegistryService()
