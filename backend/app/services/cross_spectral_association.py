"""
TRINETRA — Cross-Spectral Association Engine (Phase XI)
Orchestrates Optical <-> Thermal sensor handovers, spatiotemporal kinematic continuity,
bounding-box geometry matching, and separated confidence metrics.
"""

from __future__ import annotations
import time
import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.ptz_handover_mesh import HandoverConfidenceTier
from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("CrossSpectralAssociation")


class SensorSpectrum(str, Enum):
    OPTICAL = "OPTICAL"
    THERMAL = "THERMAL"
    DUAL = "DUAL"


@dataclass
class CrossSpectralObservation:
    camera_id: str
    spectrum: SensorSpectrum
    track_id: str
    bbox: List[float]                  # [x1, y1, x2, y2]
    detection_confidence: float
    timestamp: float = field(default_factory=time.time)
    velocity_mps: float = 1.3          # Estimated speed in m/s
    heading_deg: float = 180.0         # 0-360 degrees


@dataclass
class CrossSpectralAssociationResult:
    association_id: str
    source_camera: str
    source_spectrum: SensorSpectrum
    target_camera: str
    target_spectrum: SensorSpectrum
    optical_confidence: float
    thermal_confidence: float
    motion_confidence: float
    cross_spectral_confidence: float
    confidence_tier: HandoverConfidenceTier
    mode: str = "SIMULATED"            # SIMULATED / EXPERIMENTAL
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class CrossSpectralAssociationEngine:
    """
    Evaluates correlation between Optical (RGB) and Thermal (LWIR) target tracks.
    Eliminates reliance on RGB color appearance, utilizing spatiotemporal arrival,
    heading vector alignment, and bounding box geometric ratio consistency.
    """
    def __init__(self):
        self.associations: Dict[str, CrossSpectralAssociationResult] = {}
        self._counter = 0

    def compute_geometry_similarity(self, bbox_a: List[float], bbox_b: List[float]) -> float:
        """Compares aspect ratio (height / width) of targets across spectra."""
        if len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.5
        w_a = max(1.0, bbox_a[2] - bbox_a[0])
        h_a = max(1.0, bbox_a[3] - bbox_a[1])
        ar_a = h_a / w_a

        w_b = max(1.0, bbox_b[2] - bbox_b[0])
        h_b = max(1.0, bbox_b[3] - bbox_b[1])
        ar_b = h_b / w_b

        ratio_diff = abs(ar_a - ar_b) / max(ar_a, ar_b)
        score = max(0.0, 1.0 - ratio_diff)
        return score

    def compute_motion_continuity(
        self,
        obs_a: CrossSpectralObservation,
        obs_b: CrossSpectralObservation,
        typical_travel_sec: float = 10.0
    ) -> float:
        """Evaluates heading alignment and velocity consistency."""
        heading_diff = abs(obs_a.heading_deg - obs_b.heading_deg)
        if heading_diff > 180.0:
            heading_diff = 360.0 - heading_diff
        heading_score = max(0.0, 1.0 - (heading_diff / 90.0))

        vel_diff = abs(obs_a.velocity_mps - obs_b.velocity_mps) / max(0.1, obs_a.velocity_mps)
        vel_score = max(0.0, 1.0 - vel_diff)

        return 0.6 * heading_score + 0.4 * vel_score

    def evaluate_association(
        self,
        source_obs: CrossSpectralObservation,
        target_obs: CrossSpectralObservation,
        current_time: Optional[float] = None
    ) -> CrossSpectralAssociationResult:
        self._counter += 1
        now = current_time if current_time is not None else time.time()
        assoc_id = f"XSPEC-ASSOC-{int(now * 1000)}-{self._counter:04d}"

        elapsed_sec = abs(target_obs.timestamp - source_obs.timestamp)
        feasible, reason = camera_topology_service.is_transition_feasible(
            source_obs.camera_id,
            target_obs.camera_id,
            elapsed_sec
        )

        geom_score = self.compute_geometry_similarity(source_obs.bbox, target_obs.bbox)
        motion_score = self.compute_motion_continuity(source_obs, target_obs)

        # Distinguish optical vs thermal confidence
        if source_obs.spectrum == SensorSpectrum.OPTICAL:
            opt_conf = source_obs.detection_confidence
            therm_conf = target_obs.detection_confidence
        else:
            opt_conf = target_obs.detection_confidence
            therm_conf = source_obs.detection_confidence

        if not feasible:
            total_conf = 0.20 * (opt_conf + therm_conf) / 2.0
            tier = HandoverConfidenceTier.UNCONFIRMED
            reason_str = f"INFEASIBLE_TRANSITION_{reason}"
        else:
            # Weighted formula: Spatiotemporal (40%) + Motion (35%) + Geometry (25%)
            composite = 0.40 * (1.0) + 0.35 * motion_score + 0.25 * geom_score
            total_conf = min(0.95, composite * ((opt_conf + therm_conf) / 2.0))

            if total_conf >= 0.70:
                tier = HandoverConfidenceTier.CONFIRMED
                reason_str = "HIGH_CONFIDENCE_CROSS_SPECTRAL_MATCH"
            elif total_conf >= 0.50:
                tier = HandoverConfidenceTier.PROBABLE
                reason_str = "PROBABLE_MATCH_GEOMETRY_COMPATIBLE"
            else:
                tier = HandoverConfidenceTier.UNCONFIRMED
                reason_str = "LOW_CONFIDENCE_TELEMETRY_DIVERGENCE"

        res = CrossSpectralAssociationResult(
            association_id=assoc_id,
            source_camera=source_obs.camera_id,
            source_spectrum=source_obs.spectrum,
            target_camera=target_obs.camera_id,
            target_spectrum=target_obs.spectrum,
            optical_confidence=round(opt_conf, 2),
            thermal_confidence=round(therm_conf, 2),
            motion_confidence=round(motion_score, 2),
            cross_spectral_confidence=round(total_conf, 2),
            confidence_tier=tier,
            mode="SIMULATED",
            reason=reason_str,
            timestamp=now
        )

        self.associations[assoc_id] = res
        publish_event("cross_spectral.handover_confirmed", {
            "association_id": assoc_id,
            "source": source_obs.camera_id,
            "target": target_obs.camera_id,
            "tier": tier.value,
            "confidence": total_conf
        })

        return res

    def reset(self):
        self.associations.clear()
        self._counter = 0


cross_spectral_association_engine = CrossSpectralAssociationEngine()
