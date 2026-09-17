"""
TRINETRA Phase XIV — Multimodal Inference Manager
Coordinates three independent inference modes:
MODE A: OPTICAL ONLY (Production baseline)
MODE B: THERMAL ONLY (NOT VALIDATED)
MODE C: OPTICAL + THERMAL FUSION
Enforces decoupled operation so failure of one sensor modality never halts the platform.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.sensor_abstraction import SensorObservation, DataOrigin

logger = logging.getLogger("MultimodalInferenceManager")


class MultimodalInferenceMode(str, Enum):
    MODE_A_OPTICAL_ONLY = "MODE_A_OPTICAL_ONLY"
    MODE_B_THERMAL_ONLY = "MODE_B_THERMAL_ONLY"
    MODE_C_FUSION = "MODE_C_FUSION"


class SystemOperationalHealth(str, Enum):
    FULLY_OPERATIONAL = "FULLY_OPERATIONAL"
    THERMAL_DEGRADED = "THERMAL_DEGRADED"
    OPTICAL_DEGRADED = "OPTICAL_DEGRADED"
    SENSOR_OUTAGE = "SENSOR_OUTAGE"


@dataclass
class ModalityDetection:
    detection_id: str
    class_name: str
    confidence: float
    bbox: List[float]                  # [x1, y1, x2, y2]
    modality: str                      # OPTICAL / THERMAL / FUSED
    is_validated: bool = False         # True for optical production; False for native thermal


@dataclass
class MultimodalInferenceResult:
    inference_id: str
    camera_id: str
    requested_mode: MultimodalInferenceMode
    effective_mode: MultimodalInferenceMode
    system_health: SystemOperationalHealth
    optical_detections: List[ModalityDetection]
    thermal_detections: List[ModalityDetection]
    fused_detections: List[ModalityDetection]
    optical_latency_ms: float
    thermal_latency_ms: float
    fusion_latency_ms: float
    total_latency_ms: float
    status_message: str
    timestamp: float = field(default_factory=time.time)


class MultimodalInferenceManager:
    """
    Executes decoupled multimodal inference across optical, thermal, and fusion modes.
    """

    def __init__(self):
        self._history: List[MultimodalInferenceResult] = []

    def execute_inference(
        self,
        camera_id: str,
        mode: MultimodalInferenceMode = MultimodalInferenceMode.MODE_C_FUSION,
        optical_obs: Optional[SensorObservation] = None,
        thermal_obs: Optional[SensorObservation] = None,
        mock_detections: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> MultimodalInferenceResult:
        """
        Runs independent single-modality or fused inferences with fail-safe degradation.
        """
        t0 = time.perf_counter()
        opt_latency = 0.0
        thm_latency = 0.0
        fus_latency = 0.0

        opt_dets: List[ModalityDetection] = []
        thm_dets: List[ModalityDetection] = []
        fused_dets: List[ModalityDetection] = []

        # Evaluate availability
        has_optical = optical_obs is not None
        has_thermal = thermal_obs is not None

        if not has_optical and not has_thermal:
            effective_mode = mode
            health = SystemOperationalHealth.SENSOR_OUTAGE
            msg = "CRITICAL: Both optical and thermal sensor observations are missing."
        elif not has_thermal and mode in [MultimodalInferenceMode.MODE_B_THERMAL_ONLY, MultimodalInferenceMode.MODE_C_FUSION]:
            effective_mode = MultimodalInferenceMode.MODE_A_OPTICAL_ONLY
            health = SystemOperationalHealth.THERMAL_DEGRADED
            msg = "THERMAL_DEGRADED: Thermal stream offline; optical surveillance continuing uninterrupted."
        elif not has_optical and mode in [MultimodalInferenceMode.MODE_A_OPTICAL_ONLY, MultimodalInferenceMode.MODE_C_FUSION]:
            effective_mode = MultimodalInferenceMode.MODE_B_THERMAL_ONLY
            health = SystemOperationalHealth.OPTICAL_DEGRADED
            msg = "OPTICAL_DEGRADED: Optical stream offline; operating in thermal-only fallback."
        else:
            effective_mode = mode
            health = SystemOperationalHealth.FULLY_OPERATIONAL
            msg = "Multimodal sensor feeds synchronized and operational."

        # Optical inference (Mode A or Mode C)
        if effective_mode in [MultimodalInferenceMode.MODE_A_OPTICAL_ONLY, MultimodalInferenceMode.MODE_C_FUSION] and has_optical:
            t_opt = time.perf_counter()
            if mock_detections and "optical" in mock_detections:
                for idx, d in enumerate(mock_detections["optical"]):
                    opt_dets.append(ModalityDetection(
                        detection_id=f"DET-OPT-{idx:03d}",
                        class_name=d.get("class", "person"),
                        confidence=d.get("confidence", 0.91),
                        bbox=d.get("bbox", [0.2, 0.3, 0.4, 0.7]),
                        modality="OPTICAL",
                        is_validated=True,  # Optical baseline is validated
                    ))
            opt_latency = (time.perf_counter() - t_opt) * 1000.0

        # Thermal inference (Mode B or Mode C)
        if effective_mode in [MultimodalInferenceMode.MODE_B_THERMAL_ONLY, MultimodalInferenceMode.MODE_C_FUSION] and has_thermal:
            t_thm = time.perf_counter()
            if mock_detections and "thermal" in mock_detections:
                for idx, d in enumerate(mock_detections["thermal"]):
                    thm_dets.append(ModalityDetection(
                        detection_id=f"DET-THM-{idx:03d}",
                        class_name=d.get("class", "person"),
                        confidence=d.get("confidence", 0.85),
                        bbox=d.get("bbox", [0.21, 0.29, 0.41, 0.69]),
                        modality="THERMAL",
                        is_validated=False,  # Native thermal AI is NOT VALIDATED
                    ))
            thm_latency = (time.perf_counter() - t_thm) * 1000.0

        # Fusion evaluation (Mode C)
        if effective_mode == MultimodalInferenceMode.MODE_C_FUSION:
            t_fus = time.perf_counter()
            # Detection matching & fusion
            for o in opt_dets:
                matched_thm = None
                for t in thm_dets:
                    # BBox center distance
                    dist = ((o.bbox[0] - t.bbox[0])**2 + (o.bbox[1] - t.bbox[1])**2)**0.5
                    if dist < 0.15:
                        matched_thm = t
                        break

                if matched_thm:
                    fused_conf = round(0.55 * o.confidence + 0.45 * matched_thm.confidence, 4)
                    fused_dets.append(ModalityDetection(
                        detection_id=f"DET-FUS-{len(fused_dets):03d}",
                        class_name=o.class_name,
                        confidence=fused_conf,
                        bbox=o.bbox,
                        modality="FUSED",
                        is_validated=False,
                    ))
                else:
                    # Optical only in fusion mode
                    fused_dets.append(ModalityDetection(
                        detection_id=f"DET-FUS-OPT-{len(fused_dets):03d}",
                        class_name=o.class_name,
                        confidence=round(o.confidence * 0.90, 4),
                        bbox=o.bbox,
                        modality="FUSED",
                        is_validated=False,
                    ))
            fus_latency = (time.perf_counter() - t_fus) * 1000.0

        total_latency = (time.perf_counter() - t0) * 1000.0
        inf_id = f"INF-MM-{int(time.time()*1000)}"

        result = MultimodalInferenceResult(
            inference_id=inf_id,
            camera_id=camera_id,
            requested_mode=mode,
            effective_mode=effective_mode,
            system_health=health,
            optical_detections=opt_dets,
            thermal_detections=thm_dets,
            fused_detections=fused_dets,
            optical_latency_ms=round(opt_latency, 3),
            thermal_latency_ms=round(thm_latency, 3),
            fusion_latency_ms=round(fus_latency, 3),
            total_latency_ms=round(total_latency, 3),
            status_message=msg,
        )

        self._history.append(result)
        if len(self._history) > 200:
            self._history.pop(0)

        return result

    def reset(self):
        self._history.clear()


multimodal_inference_manager = MultimodalInferenceManager()
