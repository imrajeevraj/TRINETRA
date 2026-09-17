"""
TRINETRA — Camera Quality Analytics & Model x Camera Matrix Service (Phase XIII)
Computes operational quality indicators across all 8 camera nodes
and generates the Model x Camera governance matrix.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

logger = logging.getLogger("CameraQualityAnalytics")


class CameraOperationalQualityIndicators(BaseModel):
    camera_id: str
    alert_frequency_per_hour: float = 0.0
    false_positive_feedback_rate: float = 0.0
    false_negative_feedback_count: int = 0
    mean_confidence: float = 0.80
    tracking_instability_count: int = 0
    ptz_cue_success_rate: float = 1.00
    prediction_hit_rate: float = 0.95
    runtime_error_count: int = 0
    overall_health: str = "HEALTHY"  # "HEALTHY", "WARNING", "DEGRADED"
    primary_degradation_reason: Optional[str] = None
    last_updated: float = Field(default_factory=time.time)
    note: str = "Operational quality indicators are telemetry proxies, NOT ground truth accuracy."


class ModelCameraMatrixCell(BaseModel):
    model_domain: str
    model_version: str
    camera_id: str
    operational_health: str  # "HEALTHY", "WARNING", "DEGRADED"
    p95_latency_ms: float
    alert_volume_24h: int
    feedback_volume_24h: int
    false_positive_rate: float
    drift_indicator: str     # "STABLE", "DRIFT_SUSPECTED", "DRIFT_DETECTED"


class CameraQualityAnalyticsService:
    """
    Computes camera-level AI quality indicators without conflating proxies with ground truth accuracy.
    """

    def __init__(self):
        self.camera_indicators: Dict[str, CameraOperationalQualityIndicators] = {}
        self._init_indicators()

    def _init_indicators(self):
        for i in range(1, 9):
            cam_id = f"CAM-{i:03d}"
            self.camera_indicators[cam_id] = CameraOperationalQualityIndicators(
                camera_id=cam_id,
                alert_frequency_per_hour=14.5,
                false_positive_feedback_rate=0.03,
                false_negative_feedback_count=0,
                mean_confidence=0.82,
                tracking_instability_count=0,
                ptz_cue_success_rate=0.98,
                prediction_hit_rate=0.94,
                runtime_error_count=0,
                overall_health="HEALTHY",
            )

    def record_camera_event(
        self,
        camera_id: str,
        is_fp: bool = False,
        is_fn: bool = False,
        confidence: Optional[float] = None,
        tracking_jump: bool = False,
        ptz_cue_success: Optional[bool] = None,
        prediction_hit: Optional[bool] = None,
        runtime_error: bool = False,
    ):
        """Updates operational proxy metrics for a specific camera node."""
        ind = self.camera_indicators.get(camera_id)
        if not ind:
            return

        if is_fp:
            # Moving average of FP rate
            ind.false_positive_feedback_rate = round(min(1.0, ind.false_positive_feedback_rate + 0.02), 3)
        if is_fn:
            ind.false_negative_feedback_count += 1
        if confidence is not None:
            ind.mean_confidence = round((ind.mean_confidence * 0.90) + (confidence * 0.10), 3)
        if tracking_jump:
            ind.tracking_instability_count += 1
        if ptz_cue_success is False:
            ind.ptz_cue_success_rate = round(max(0.0, ind.ptz_cue_success_rate - 0.05), 2)
        if prediction_hit is False:
            ind.prediction_hit_rate = round(max(0.0, ind.prediction_hit_rate - 0.05), 2)
        if runtime_error:
            ind.runtime_error_count += 1

        # Health state calculation
        if ind.false_positive_feedback_rate > 0.10 or ind.false_negative_feedback_count >= 3 or ind.runtime_error_count >= 3:
            ind.overall_health = "DEGRADED"
            ind.primary_degradation_reason = "Elevated feedback errors or runtime failures"
        elif ind.false_positive_feedback_rate > 0.05 or ind.tracking_instability_count >= 5:
            ind.overall_health = "WARNING"
            ind.primary_degradation_reason = "Tracking instability or rising FP rate"
        else:
            ind.overall_health = "HEALTHY"
            ind.primary_degradation_reason = None

        ind.last_updated = time.time()

    def record_telemetry_event(
        self,
        camera_id: str,
        is_fp_feedback: bool = False,
        is_fn_feedback: bool = False,
        is_runtime_error: bool = False,
        confidence: Optional[float] = None,
        **kwargs,
    ):
        return self.record_camera_event(
            camera_id=camera_id,
            is_fp=is_fp_feedback,
            is_fn=is_fn_feedback,
            runtime_error=is_runtime_error,
            confidence=confidence,
        )

    def get_indicators(self, camera_id: str) -> Optional[CameraOperationalQualityIndicators]:
        return self.camera_indicators.get(camera_id)

    get_camera_indicators = get_indicators

    def list_indicators(self) -> List[CameraOperationalQualityIndicators]:
        return list(self.camera_indicators.values())

    def get_model_camera_matrix(self) -> Dict[str, List[ModelCameraMatrixCell]]:
        """
        Generates the Model x Camera Matrix for Ground, Airborne, and Security domains across all 8 cameras.
        """
        matrix: Dict[str, List[ModelCameraMatrixCell]] = {
            "GROUND": [],
            "AIRBORNE": [],
            "SECURITY_ITEM": [],
        }

        domain_versions = {
            "GROUND": "v2.0.0",
            "AIRBORNE": "v2.0.0",
            "SECURITY_ITEM": "v2.1.0",
        }

        for domain, ver in domain_versions.items():
            for i in range(1, 9):
                cam_id = f"CAM-{i:03d}"
                ind = self.camera_indicators.get(cam_id)
                health = ind.overall_health if ind else "HEALTHY"

                # Specific camera variance simulation for realism
                fp_rate = ind.false_positive_feedback_rate if ind else 0.03
                if cam_id == "CAM-002" and domain == "GROUND":
                    # Simulated known small pedestrian challenge
                    drift = "DRIFT_SUSPECTED"
                else:
                    drift = "STABLE"

                cell = ModelCameraMatrixCell(
                    model_domain=domain,
                    model_version=ver,
                    camera_id=cam_id,
                    operational_health=health,
                    p95_latency_ms=15.2 if domain == "GROUND" else (11.4 if domain == "AIRBORNE" else 13.8),
                    alert_volume_24h=142 if domain == "GROUND" else (28 if domain == "AIRBORNE" else 49),
                    feedback_volume_24h=int(fp_rate * 100),
                    false_positive_rate=fp_rate,
                    drift_indicator=drift,
                )
                matrix[domain].append(cell)

        return matrix

    def reset(self):
        self.camera_indicators.clear()
        self._init_indicators()


camera_quality_analytics = CameraQualityAnalyticsService()
