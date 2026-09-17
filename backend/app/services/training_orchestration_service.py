"""
TRINETRA — Training Orchestration & Experiment Registry Service (Phase XII)
Emits reproducible training manifests, tracks experiment runs in training_runs/<run_id>,
and enforces the rule that training runs can never directly activate production.
"""

from __future__ import annotations
import os
import json
import yaml
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("TrainingOrchestrationService")

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_RUNS_DIR = REPO_ROOT / "training_runs"


class TrainingHyperparameters(BaseModel):
    architecture: str  # "YOLO11n", "YOLO26n"
    base_checkpoint: str
    dataset_version: str
    image_size: int = 640
    epochs: int = 50
    batch_size: int = 16
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    augmentation: Dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    hardware: str = "NVIDIA CUDA (RTX / Jetson Orin)"
    framework_version: str = "Ultralytics YOLO 8.x / PyTorch 2.x"
    dependencies: Dict[str, str] = Field(default_factory=dict)


class TrainingRunManifest(BaseModel):
    run_id: str
    experiment_name: str
    domain: str  # "GROUND", "AIRBORNE", "SECURITY_ITEM", "THERMAL"
    hyperparameters: TrainingHyperparameters
    status: str = "COMPLETED"  # "INITIALIZING", "RUNNING", "COMPLETED", "FAILED"
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    artifact_path: str
    artifact_sha256: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    can_activate_production: bool = False  # Strictly False by design


class TrainingOrchestrationService:
    """
    Orchestrates training runs and records full provenance manifests.
    Guarantees that a training run produces only candidate artifacts,
    never directly active production models.
    """

    def __init__(self, runs_root: Optional[Path] = None):
        self.runs_root = runs_root or TRAINING_RUNS_DIR
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.runs: Dict[str, TrainingRunManifest] = {}
        self._run_counter = 0

    def create_training_run(
        self,
        experiment_name: str,
        domain: str,
        hyperparameters: TrainingHyperparameters,
        simulated_metrics: Optional[Dict[str, float]] = None,
        artifact_bytes: Optional[bytes] = None,
    ) -> TrainingRunManifest:
        """
        Creates an immutable training run with full hyperparameter and environment tracking.
        Writes manifest.yaml, metrics.json, sha256.txt, and saves model artifact.
        """
        self._run_counter += 1
        now = time.time()
        run_id = f"RUN-{domain}-{int(now)}-{self._run_counter:03d}"
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        # Artifact handling
        artifact_path = run_dir / "artifacts" / "best.pt"
        if artifact_bytes is not None:
            with open(artifact_path, "wb") as f:
                f.write(artifact_bytes)
        else:
            # Create a mock candidate weight checkpoint
            mock_data = f"IBVAP_WEIGHTS_{domain}_{run_id}_{now}".encode("utf-8")
            with open(artifact_path, "wb") as f:
                f.write(mock_data)

        # Compute hash
        hasher = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        art_sha = hasher.hexdigest().upper()

        # Write sha256.txt
        with open(run_dir / "sha256.txt", "w", encoding="utf-8") as f:
            f.write(f"{art_sha}  best.pt\n")

        # Write metrics.json
        metrics = simulated_metrics or {
            "train_loss": 0.024,
            "val_loss": 0.029,
            "mAP50": 0.72,
            "mAP50_95": 0.48,
        }
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Build manifest
        manifest = TrainingRunManifest(
            run_id=run_id,
            experiment_name=experiment_name,
            domain=domain,
            hyperparameters=hyperparameters,
            status="COMPLETED",
            started_at=now,
            completed_at=time.time(),
            artifact_path=str(artifact_path),
            artifact_sha256=art_sha,
            metrics=metrics,
            can_activate_production=False,  # IMMUTABLE SAFETY RULE
        )

        # Write manifest.yaml
        with open(run_dir / "manifest.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(manifest.model_dump(), f, sort_keys=False)

        self.runs[run_id] = manifest
        logger.info(f"Recorded training run {run_id} ({experiment_name}) with artifact SHA {art_sha[:16]}...")
        return manifest

    def get_run(self, run_id: str) -> Optional[TrainingRunManifest]:
        return self.runs.get(run_id)

    def list_runs(self, domain: Optional[str] = None) -> List[TrainingRunManifest]:
        if domain:
            return [r for r in self.runs.values() if r.domain == domain]
        return list(self.runs.values())

    def reset(self):
        self.runs.clear()
        self._run_counter = 0


training_orchestration_service = TrainingOrchestrationService()
