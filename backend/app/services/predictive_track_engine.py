"""
TRINETRA — Predictive Track & Trajectory Engine (Phase IX)
Deterministic 2D Constant Velocity (CV) Kalman Filter motion estimator
calculating position, velocity vectors, heading angle, and multi-horizon forecasts.
Method Version: KALMAN_CV_v1
"""

from __future__ import annotations
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("PredictiveTrackEngine")


@dataclass
class TrajectoryPoint:
    timestamp: float
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    heading_deg: float
    cardinal_dir: str
    camera_id: str
    zone_id: Optional[str] = None
    observation_id: Optional[str] = None
    confidence: float = 0.8


@dataclass
class KalmanTrackState:
    track_id: str
    camera_id: str
    last_update: float
    # State: [x, y, vx, vy]
    x: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float64))
    # Covariance: 4x4
    P: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64) * 100.0)
    # Measurement matrix: 2x4
    H: np.ndarray = field(default_factory=lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64))
    # Measurement noise: 2x2
    R: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float64) * 4.0)
    # Observation history
    history: List[TrajectoryPoint] = field(default_factory=list)
    confidence: float = 0.8
    track_age: int = 1


class PredictiveTrackEngine:
    METHOD_NAME = "KALMAN_CV"
    METHOD_VERSION = "v1.0"

    def __init__(
        self,
        default_horizons: Optional[List[float]] = None,
        max_history_points: int = 100,
        process_noise_std: float = 0.5
    ):
        self.horizons = default_horizons or [5.0, 10.0, 15.0, 30.0, 60.0]
        self.max_history = max_history_points
        self.q_std = process_noise_std
        self.tracks: Dict[str, KalmanTrackState] = {}

    def _get_q_matrix(self, dt: float) -> np.ndarray:
        q = self.q_std ** 2
        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0
        dt4 = (dt ** 4) / 4.0
        return np.array([
            [dt4 * q, 0, dt3 * q, 0],
            [0, dt4 * q, 0, dt3 * q],
            [dt3 * q, 0, dt2 * q, 0],
            [0, dt3 * q, 0, dt2 * q]
        ], dtype=np.float64)

    def _get_f_matrix(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

    def angle_to_cardinal(self, angle_deg: float) -> str:
        """
        Converts heading angle in degrees [0, 360) in image coordinates
        (+x = EAST, +y = SOUTH/downwards, -x = WEST, -y = NORTH/upwards)
        to 8-point cardinal compass.
        """
        dirs = ["EAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "WEST", "NORTHWEST", "NORTH", "NORTHEAST"]
        idx = int((angle_deg + 22.5) // 45) % 8
        return dirs[idx]

    def update_track(
        self,
        track_id: str,
        x: float,
        y: float,
        camera_id: str,
        timestamp: Optional[float] = None,
        zone_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        confidence: float = 0.8
    ) -> TrajectoryPoint:
        now = timestamp if timestamp is not None else time.time()

        if track_id not in self.tracks:
            # Initialize new Kalman filter state
            track = KalmanTrackState(
                track_id=track_id,
                camera_id=camera_id,
                last_update=now,
                x=np.array([x, y, 0.0, 0.0], dtype=np.float64),
                confidence=confidence,
                track_age=1
            )
            self.tracks[track_id] = track
            tp = TrajectoryPoint(
                timestamp=now,
                x=x,
                y=y,
                vx=0.0,
                vy=0.0,
                speed=0.0,
                heading_deg=0.0,
                cardinal_dir="UNKNOWN",
                camera_id=camera_id,
                zone_id=zone_id,
                observation_id=observation_id,
                confidence=confidence
            )
            track.history.append(tp)
            return tp

        track = self.tracks[track_id]
        dt = max(0.001, now - track.last_update)
        track.last_update = now
        track.camera_id = camera_id
        track.track_age += 1
        track.confidence = round((track.confidence * 0.7) + (confidence * 0.3), 2)

        # 1. Predict Step
        F = self._get_f_matrix(dt)
        Q = self._get_q_matrix(dt)
        x_pred = F @ track.x
        P_pred = F @ track.P @ F.T + Q

        # 2. Update Step (Kalman Gain)
        z = np.array([x, y], dtype=np.float64)
        y_residual = z - (track.H @ x_pred)
        S = track.H @ P_pred @ track.H.T + track.R
        K = P_pred @ track.H.T @ np.linalg.inv(S)

        track.x = x_pred + (K @ y_residual)
        track.P = (np.eye(4, dtype=np.float64) - K @ track.H) @ P_pred

        # 3. Kinematic Estimates
        est_x, est_y, est_vx, est_vy = track.x
        speed = float(np.sqrt(est_vx ** 2 + est_vy ** 2))

        if speed > 0.5:
            # Heading in degrees [0, 360)
            rad = math.atan2(est_vy, est_vx)
            deg = math.degrees(rad) % 360.0
            cardinal = self.angle_to_cardinal(deg)
        else:
            deg = 0.0
            cardinal = "STATIONARY"

        tp = TrajectoryPoint(
            timestamp=now,
            x=float(est_x),
            y=float(est_y),
            vx=float(est_vx),
            vy=float(est_vy),
            speed=round(speed, 2),
            heading_deg=round(deg, 1),
            cardinal_dir=cardinal,
            camera_id=camera_id,
            zone_id=zone_id,
            observation_id=observation_id,
            confidence=track.confidence
        )

        track.history.append(tp)
        if len(track.history) > self.max_history:
            track.history.pop(0)

        return tp

    def estimate_velocity(self, track_id: str) -> Tuple[float, float, float]:
        """Returns (vx, vy, speed) in coordinate units per second."""
        track = self.tracks.get(track_id)
        if not track:
            return 0.0, 0.0, 0.0
        vx = float(track.x[2])
        vy = float(track.x[3])
        speed = float(np.sqrt(vx ** 2 + vy ** 2))
        return round(vx, 2), round(vy, 2), round(speed, 2)

    def estimate_heading(self, track_id: str) -> Tuple[float, str]:
        """Returns (heading_degrees, cardinal_direction)."""
        track = self.tracks.get(track_id)
        if not track or len(track.history) == 0:
            return 0.0, "UNKNOWN"
        last_pt = track.history[-1]
        return last_pt.heading_deg, last_pt.cardinal_dir

    def predict_position(self, track_id: str, horizon_sec: float) -> Optional[Tuple[float, float, float]]:
        """
        Forecasts future (x, y) at t + horizon_sec with confidence decay.
        Returns: (pred_x, pred_y, decayed_confidence)
        """
        track = self.tracks.get(track_id)
        if not track:
            return None

        # Constant velocity forward extrapolation
        pred_x = float(track.x[0] + track.x[2] * horizon_sec)
        pred_y = float(track.x[1] + track.x[3] * horizon_sec)

        # Confidence decays exponentially over forecast horizon
        # 5s: 0.90x, 15s: 0.72x, 30s: 0.52x, 60s: 0.27x
        decay_factor = math.exp(-0.021 * horizon_sec)
        decayed_conf = max(0.10, min(0.99, round(track.confidence * decay_factor, 2)))

        return round(pred_x, 1), round(pred_y, 1), decayed_conf

    def forecast(
        self,
        track_id: str,
        horizons: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Produces multi-horizon trajectory predictions.
        Explicitly labeled PREDICTED / SIMULATED.
        """
        h_list = horizons or self.horizons
        track = self.tracks.get(track_id)
        if not track:
            return []

        now = track.last_update
        predictions = []

        for h in h_list:
            res = self.predict_position(track_id, h)
            if res:
                px, py, conf = res
                predictions.append({
                    "horizon_sec": h,
                    "target_timestamp": now + h,
                    "predicted_x": px,
                    "predicted_y": py,
                    "confidence": conf,
                    "confidence_tier": "HIGH" if conf >= 0.75 else ("MEDIUM" if conf >= 0.50 else "LOW"),
                    "status": "PREDICTED",
                    "method": f"{self.METHOD_NAME}_{self.METHOD_VERSION}"
                })

        return predictions

    def get_trajectory_history(self, track_id: str) -> List[Dict[str, Any]]:
        track = self.tracks.get(track_id)
        if not track:
            return []
        return [
            {
                "timestamp": pt.timestamp,
                "x": pt.x,
                "y": pt.y,
                "vx": pt.vx,
                "vy": pt.vy,
                "speed": pt.speed,
                "heading_deg": pt.heading_deg,
                "cardinal_dir": pt.cardinal_dir,
                "camera_id": pt.camera_id,
                "zone_id": pt.zone_id,
                "confidence": pt.confidence,
                "status": "OBSERVED"
            }
            for pt in track.history
        ]


predictive_track_engine = PredictiveTrackEngine()
