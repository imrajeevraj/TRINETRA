"""
TRINETRA Phase XIV — Thermal Data Quality Gate Service
Evaluates incoming thermal frames across 11 automated quality rules.
Outputs verdicts: ACCEPT, REVIEW, REJECT.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.thermal_ingestion_service import IngestedThermalFrame, ThermalPayloadType

logger = logging.getLogger("ThermalQualityGate")


class QualityGateVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass
class QualityGateResult:
    frame_id: str
    sensor_id: str
    verdict: QualityGateVerdict
    failed_rules: List[str]
    warning_rules: List[str]
    quality_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)


class ThermalQualityGate:
    """
    Executes 11 quality integrity checks on thermal LWIR frames.
    """

    def __init__(self):
        self._last_frame_hashes: Dict[str, str] = {}
        self._last_frame_timestamps: Dict[str, float] = {}
        self._last_frame_sequences: Dict[str, int] = {}
        self._gate_history: List[QualityGateResult] = []

    def evaluate_frame(
        self,
        frame: IngestedThermalFrame,
        expected_resolution: Tuple[int, int] = (640, 512),
        expected_calibration_id: Optional[str] = None,
    ) -> QualityGateResult:
        failed_rules: List[str] = []
        warning_rules: List[str] = []

        # 1. Corrupted frame
        if not frame.raw_payload_bytes or len(frame.raw_payload_bytes) < 32:
            failed_rules.append("RULE_CORRUPT_FRAME")

        # 2. Missing metadata
        if not frame.sensor_id or not frame.camera_id or not frame.frame_id:
            failed_rules.append("RULE_MISSING_METADATA")

        # 3. Invalid dimensions
        if frame.resolution != expected_resolution:
            failed_rules.append("RULE_INVALID_DIMENSIONS")

        # 4. Saturated regions
        if frame.is_saturated:
            warning_rules.append("RULE_SATURATED_REGIONS")

        # 5. Dead pixels
        if frame.dead_pixel_count > 50:
            warning_rules.append("RULE_DEAD_PIXELS")

        # 6. Excessive noise
        if frame.radiometric_meta and frame.radiometric_meta.thermal_sensitivity_netd_mk > 150.0:
            warning_rules.append("RULE_EXCESSIVE_NOISE")

        # 7. Missing timestamps
        if frame.timestamp <= 0.0:
            failed_rules.append("RULE_MISSING_TIMESTAMPS")

        # 8. Duplicate frames (frozen camera encoder)
        last_hash = self._last_frame_hashes.get(frame.sensor_id)
        if last_hash and last_hash == frame.frame_hash:
            warning_rules.append("RULE_DUPLICATE_FRAMES")
        self._last_frame_hashes[frame.sensor_id] = frame.frame_hash

        # 9. Calibration mismatch
        if expected_calibration_id and frame.calibration_id != expected_calibration_id:
            failed_rules.append("RULE_CALIBRATION_MISMATCH")

        # 10. Temperature metadata inconsistency (radiometric mode)
        if frame.payload_type == ThermalPayloadType.RADIOMETRIC_THERMAL_DATA:
            rm = frame.radiometric_meta
            if rm.min_scene_temp_c is not None and rm.max_scene_temp_c is not None:
                if rm.min_scene_temp_c > rm.max_scene_temp_c or rm.min_scene_temp_c < -60.0 or rm.max_scene_temp_c > 350.0:
                    failed_rules.append("RULE_TEMPERATURE_INCONSISTENCY")

        # 11. Temporal discontinuity
        last_time = self._last_frame_timestamps.get(frame.sensor_id)
        if last_time:
            dt = frame.timestamp - last_time
            if dt < 0.0 or dt > 1.0:  # backward or >1000ms gap
                warning_rules.append("RULE_TEMPORAL_DISCONTINUITY")
        self._last_frame_timestamps[frame.sensor_id] = frame.timestamp

        # Compute quality score & verdict
        score = 1.0 - (len(failed_rules) * 0.40) - (len(warning_rules) * 0.10)
        score = round(max(0.0, min(1.0, score)), 3)

        if failed_rules:
            verdict = QualityGateVerdict.REJECT
        elif warning_rules:
            verdict = QualityGateVerdict.REVIEW
        else:
            verdict = QualityGateVerdict.ACCEPT

        result = QualityGateResult(
            frame_id=frame.frame_id,
            sensor_id=frame.sensor_id,
            verdict=verdict,
            failed_rules=failed_rules,
            warning_rules=warning_rules,
            quality_score=score,
            details={
                "failed_count": len(failed_rules),
                "warning_count": len(warning_rules),
                "resolution": frame.resolution,
                "is_saturated": frame.is_saturated,
            },
        )

        self._gate_history.append(result)
        if len(self._gate_history) > 500:
            self._gate_history.pop(0)

        if verdict == QualityGateVerdict.REJECT:
            logger.warning(f"Thermal frame {frame.frame_id} REJECTED by quality gate: {failed_rules}")

        return result

    def reset(self):
        self._last_frame_hashes.clear()
        self._last_frame_timestamps.clear()
        self._last_frame_sequences.clear()
        self._gate_history.clear()


thermal_quality_gate = ThermalQualityGate()
