"""
TRINETRA — Camera Transition & Zone Predictor (Phase IX)
Predicts candidate next cameras, arrival ETA windows, and zone entries
using spatial topology and kinematic trajectory heading.
Method Version: TOPOLOGY_HEADING_v1
"""

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.predictive_track_engine import predictive_track_engine

logger = logging.getLogger("CameraTransitionPredictor")


@dataclass
class CameraPredictionHypothesis:
    predicted_camera: str
    predicted_zone: str
    eta_min_sec: float
    eta_max_sec: float
    typical_eta_sec: float
    confidence: float
    direction_match: bool
    reason: str


@dataclass
class TransitionPrediction:
    prediction_id: str
    entity_id: str
    current_camera: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=time.time)
    primary_hypothesis: Optional[CameraPredictionHypothesis] = None
    alternative_hypotheses: List[CameraPredictionHypothesis] = field(default_factory=list)
    status: str = "PREDICTED"  # PREDICTED, EXPIRED, HIT, PARTIAL_HIT, MISS
    method: str = "TOPOLOGY_HEADING_v1"


class CameraTransitionPredictor:
    def __init__(self, expiry_padding_sec: float = 15.0):
        self.expiry_padding = expiry_padding_sec
        self.active_predictions: Dict[str, TransitionPrediction] = {}
        self._pred_counter = 0

    def _next_prediction_id(self) -> str:
        self._pred_counter += 1
        year = time.strftime("%Y", time.gmtime())
        return f"PRED-{year}-{self._pred_counter:06d}"

    def predict_transition(
        self,
        entity_id: str,
        current_camera: str,
        track_id: Optional[str] = None,
        speed_factor: float = 1.0,
        timestamp: Optional[float] = None
    ) -> TransitionPrediction:
        now = timestamp if timestamp is not None else time.time()
        pred_id = self._next_prediction_id()

        # 1. Retrieve kinematic heading if track is available
        heading_deg = 0.0
        cardinal_dir = "UNKNOWN"
        speed = 1.0
        if track_id and track_id in predictive_track_engine.tracks:
            heading_deg, cardinal_dir = predictive_track_engine.estimate_heading(track_id)
            _, _, speed = predictive_track_engine.estimate_velocity(track_id)

        # 2. Retrieve topological adjacent candidate edges
        candidates = camera_topology_service.predict_next_cameras(current_camera)

        hypotheses: List[CameraPredictionHypothesis] = []

        if not candidates:
            # Sentry terminal boundary (no outgoing edges)
            hyp = CameraPredictionHypothesis(
                predicted_camera="TERMINAL_BOUNDARY",
                predicted_zone="ZONE-PERIMETER-EXTERIOR",
                eta_min_sec=30.0,
                eta_max_sec=60.0,
                typical_eta_sec=45.0,
                confidence=0.30,
                direction_match=False,
                reason="Terminal perimeter boundary; no adjacent connected cameras"
            )
            hypotheses.append(hyp)
        else:
            for c in candidates:
                edge_dir = c.get("direction", "").upper()
                # Check direction consistency
                dir_match = False
                if cardinal_dir != "UNKNOWN" and cardinal_dir != "STATIONARY":
                    if edge_dir in cardinal_dir or cardinal_dir in edge_dir:
                        dir_match = True

                # Confidence scoring: base 0.50 + direction match 0.30
                conf = 0.50
                if dir_match:
                    conf += 0.32
                elif cardinal_dir == "STATIONARY":
                    conf -= 0.15

                # Adjust ETA if target speed is high
                speed_scale = max(0.5, min(2.0, speed if speed > 0.1 else 1.0))
                min_eta = round(c["min_eta_sec"] / speed_scale, 1)
                max_eta = round(c["max_eta_sec"] / speed_scale, 1)
                typ_eta = round(c["typical_eta_sec"] / speed_scale, 1)

                target_cam = c["to_camera"]
                zone_info = camera_topology_service.nodes.get(target_cam, {}).get("zone_id", "ZONE-UNKNOWN")

                reason = f"Topological transition from {current_camera} -> {target_cam}"
                if dir_match:
                    reason += f" (Heading {cardinal_dir} matches corridor direction {edge_dir})"

                hypotheses.append(CameraPredictionHypothesis(
                    predicted_camera=target_cam,
                    predicted_zone=zone_info,
                    eta_min_sec=min_eta,
                    eta_max_sec=max_eta,
                    typical_eta_sec=typ_eta,
                    confidence=round(conf, 2),
                    direction_match=dir_match,
                    reason=reason
                ))

        # Rank hypotheses by confidence descending
        hypotheses.sort(key=lambda x: x.confidence, reverse=True)

        primary = hypotheses[0] if hypotheses else None
        alternatives = hypotheses[1:] if len(hypotheses) > 1 else []

        max_window = primary.eta_max_sec if primary else 30.0
        expires_at = now + max_window + self.expiry_padding

        pred = TransitionPrediction(
            prediction_id=pred_id,
            entity_id=entity_id,
            current_camera=current_camera,
            created_at=now,
            expires_at=expires_at,
            primary_hypothesis=primary,
            alternative_hypotheses=alternatives,
            status="PREDICTED",
            method="TOPOLOGY_HEADING_v1"
        )

        self.active_predictions[pred_id] = pred
        logger.info(f"Generated transition prediction {pred_id}: {current_camera} -> {primary.predicted_camera if primary else 'NONE'} (Conf: {primary.confidence if primary else 0})")
        return pred

    def get_prediction(self, prediction_id: str) -> Optional[TransitionPrediction]:
        return self.active_predictions.get(prediction_id)

    def expire_stale_predictions(self, current_time: Optional[float] = None) -> List[str]:
        now = current_time if current_time is not None else time.time()
        expired_ids = []
        for pid, pred in list(self.active_predictions.items()):
            if pred.status == "PREDICTED" and now > pred.expires_at:
                pred.status = "EXPIRED"
                expired_ids.append(pid)
                logger.info(f"Prediction {pid} expired without target observation (ETA window elapsed)")
        return expired_ids


camera_transition_predictor = CameraTransitionPredictor()
