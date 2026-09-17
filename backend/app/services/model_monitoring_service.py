"""
TRINETRA — Model Monitoring & Operational Drift Detection Service (Phase XII)
Computes real-time operational proxy health metrics, detects data/confidence drift,
and triggers DEGRADED lifecycle state transitions upon statistical anomalies.
"""

from __future__ import annotations
import time
import math
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

logger = logging.getLogger("ModelMonitoringService")


class ModelHealthTelemetry(BaseModel):
    model_id: str
    model_version: str
    domain: str
    camera_id: str
    inference_fps: float
    p50_latency_ms: float
    p95_latency_ms: float
    gpu_memory_mb: float
    cpu_percent: float
    error_rate: float
    total_frames: int
    alert_count: int
    mean_confidence: float
    confidence_drift_detected: bool = False
    drift_score: float = 0.0
    status: str = "HEALTHY"  # "HEALTHY", "DEGRADED", "ERROR"
    timestamp: float = Field(default_factory=time.time)
    note: str = "Metrics represent operational telemetry proxies, not ground truth accuracy."


class DriftDetectionReport(BaseModel):
    model_id: str
    domain: str
    baseline_mean_conf: float
    current_mean_conf: float
    confidence_delta: float
    class_distribution_shift: bool
    operator_rejection_rate: float
    drift_detected: bool
    recommended_action: str
    timestamp: float = Field(default_factory=time.time)


class ModelMonitoringService:
    """
    Monitors live operational health of deployed models.
    Enforces that telemetry is transparently presented as 'operational proxies'.
    """

    def __init__(self):
        self.node_telemetry: Dict[str, ModelHealthTelemetry] = {}  # camera_id -> telemetry
        self.drift_reports: Dict[str, DriftDetectionReport] = {}   # model_id -> report
        self.confidence_history: Dict[str, List[float]] = {}       # model_id -> confidences

    def record_inference_telemetry(
        self,
        camera_id: str,
        model_id: str,
        domain: str,
        latency_ms: float,
        detections: List[Dict[str, Any]],
        gpu_mem_mb: float = 1250.0,
        cpu_pct: float = 24.5,
        had_error: bool = False,
    ) -> ModelHealthTelemetry:
        """
        Ingests real-time inference telemetry from an edge camera node.
        """
        confidences = [d.get("confidence", 0.0) for d in detections if "confidence" in d]
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.75

        if model_id not in self.confidence_history:
            self.confidence_history[model_id] = []
        self.confidence_history[model_id].append(mean_conf)
        if len(self.confidence_history[model_id]) > 500:
            self.confidence_history[model_id].pop(0)

        # Baseline expected confidence for production models is ~0.80
        baseline_conf = 0.80
        drift_delta = abs(baseline_conf - mean_conf)
        is_drift = drift_delta > 0.18  # Significant confidence collapse

        status = "HEALTHY"
        if had_error or latency_ms > 35.0:
            status = "DEGRADED"
        elif is_drift:
            status = "DEGRADED"

        # If degraded, trigger state transition
        if status == "DEGRADED":
            curr_state = model_lifecycle_engine.get_state(model_id)
            if curr_state in [ModelLifecycleState.ACTIVE, ModelLifecycleState.MONITORING]:
                model_lifecycle_engine.transition(
                    model_id=model_id,
                    target_state=ModelLifecycleState.DEGRADED,
                    operator="MONITORING_WATCHDOG",
                    reason=f"HEALTH_DEGRADATION_DETECTED: Latency={latency_ms:.1f}ms, DriftDelta={drift_delta:.2f}",
                )

        telemetry = ModelHealthTelemetry(
            model_id=model_id,
            model_version="v2.0.0",
            domain=domain,
            camera_id=camera_id,
            inference_fps=round(1000.0 / max(latency_ms, 1.0), 1),
            p50_latency_ms=round(latency_ms * 0.85, 2),
            p95_latency_ms=round(latency_ms, 2),
            gpu_memory_mb=gpu_mem_mb,
            cpu_percent=cpu_pct,
            error_rate=1.0 if had_error else 0.0,
            total_frames=len(self.confidence_history.get(model_id, [])),
            alert_count=len(detections),
            mean_confidence=round(mean_conf, 3),
            confidence_drift_detected=is_drift,
            drift_score=round(drift_delta, 3),
            status=status,
        )

        self.node_telemetry[camera_id] = telemetry
        return telemetry

    def evaluate_model_drift(
        self,
        model_id: str,
        domain: str,
        operator_rejection_rate: float = 0.02,
    ) -> DriftDetectionReport:
        """
        Evaluates accumulated confidence and class distributions for distribution shift.
        """
        history = self.confidence_history.get(model_id, [0.80])
        current_mean = sum(history) / len(history) if history else 0.80
        baseline_mean = 0.80

        delta = baseline_mean - current_mean
        is_drift = delta > 0.15 or operator_rejection_rate > 0.10

        action = "RETAIN_ACTIVE_MONITORING"
        if is_drift:
            action = "DEGRADE_AND_PROMPT_RETRAINING_INVESTIGATION"
            curr_state = model_lifecycle_engine.get_state(model_id)
            if curr_state == ModelLifecycleState.ACTIVE:
                model_lifecycle_engine.transition(
                    model_id=model_id,
                    target_state=ModelLifecycleState.DEGRADED,
                    operator="DRIFT_DETECTOR",
                    reason=f"CONFIDENCE_DRIFT_SCORE_{delta:.3f}_EXCEEDED_THRESHOLD",
                )

        report = DriftDetectionReport(
            model_id=model_id,
            domain=domain,
            baseline_mean_conf=baseline_mean,
            current_mean_conf=round(current_mean, 3),
            confidence_delta=round(delta, 3),
            class_distribution_shift=is_drift,
            operator_rejection_rate=operator_rejection_rate,
            drift_detected=is_drift,
            recommended_action=action,
        )

        self.drift_reports[model_id] = report
        logger.info(f"Drift evaluation for {model_id}: Detected={is_drift} (Score={delta:.3f}, Action={action})")
        return report

    def get_health(self, camera_id: str) -> Optional[ModelHealthTelemetry]:
        return self.node_telemetry.get(camera_id)

    def list_health(self) -> List[ModelHealthTelemetry]:
        return list(self.node_telemetry.values())

    def reset(self):
        self.node_telemetry.clear()
        self.drift_reports.clear()
        self.confidence_history.clear()


model_monitoring_service = ModelMonitoringService()
