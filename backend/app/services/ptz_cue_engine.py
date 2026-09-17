"""
TRINETRA — Hardware-Agnostic PTZ Slew-to-Cue Engine (Phase VII)
Coordinates intelligent target prioritization, hysteresis anti-oscillation,
camera slew-to-cue commands, and vibration settling states.
"""

from __future__ import annotations
import time
import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("PTZCueEngine")


class PTZState(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    SETTLING = "SETTLING"
    STABLE = "STABLE"
    ERROR = "ERROR"


class CuePriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PTZPosition:
    pan: float = 0.0      # -180.0 to +180.0 degrees
    tilt: float = 0.0     # -90.0 to +90.0 degrees
    zoom: float = 1.0     # 1.0x to 30.0x optical zoom


@dataclass
class CueTarget:
    track_id: str
    camera_id: str
    class_name: str
    confidence: float
    bbox: List[float]       # [x1, y1, x2, y2]
    risk_score: int
    fence_violation: bool
    priority: CuePriority
    cue_score: float
    timestamp: float = field(default_factory=time.time)


class PTZController:
    """Abstract hardware-agnostic PTZ interface."""
    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError

    def slew_to(self, pan: float, tilt: float, zoom: float, speed: float = 1.0) -> bool:
        raise NotImplementedError

    def home(self) -> bool:
        raise NotImplementedError

    def stop(self) -> bool:
        raise NotImplementedError

    def update(self) -> PTZState:
        raise NotImplementedError


class MockPTZController(PTZController):
    """
    Simulated PTZ camera with realistic physics, slew duration,
    vibration settling time, and state transitions.
    """
    def __init__(self, camera_id: str, max_slew_rate: float = 60.0, settle_time: float = 0.5):
        self.camera_id = camera_id
        self.current_pos = PTZPosition()
        self.target_pos = PTZPosition()
        self.home_pos = PTZPosition(pan=0.0, tilt=0.0, zoom=1.0)
        self.max_slew_rate = max_slew_rate   # degrees per second
        self.settle_time = settle_time       # seconds to settle after moving

        self.state = PTZState.IDLE
        self.hardware_mode = "SIMULATED"
        self._move_start_time = 0.0
        self._move_duration = 0.0
        self._settle_start_time = 0.0
        self.last_error: Optional[str] = None

    def slew_to(self, pan: float, tilt: float, zoom: float, speed: float = 1.0) -> bool:
        pan = max(-180.0, min(180.0, float(pan)))
        tilt = max(-90.0, min(90.0, float(tilt)))
        zoom = max(1.0, min(30.0, float(zoom)))

        d_pan = abs(pan - self.current_pos.pan)
        d_tilt = abs(tilt - self.current_pos.tilt)
        max_dist = max(d_pan, d_tilt, abs(zoom - self.current_pos.zoom) * 5.0)

        rate = max(10.0, self.max_slew_rate * max(0.2, min(2.0, speed)))
        self._move_duration = max_dist / rate if max_dist > 0 else 0.05
        self._move_start_time = time.time()
        self.target_pos = PTZPosition(pan=pan, tilt=tilt, zoom=zoom)
        self.state = PTZState.MOVING
        logger.info(f"[{self.camera_id}] PTZ slew initiated -> Pan={pan:.1f}, Tilt={tilt:.1f}, Zoom={zoom:.1f}x (Est. {self._move_duration:.2f}s)")
        return True

    def home(self) -> bool:
        return self.slew_to(self.home_pos.pan, self.home_pos.tilt, self.home_pos.zoom)

    def stop(self) -> bool:
        self.target_pos = PTZPosition(self.current_pos.pan, self.current_pos.tilt, self.current_pos.zoom)
        self.state = PTZState.IDLE
        return True

    def update(self) -> PTZState:
        now = time.time()
        if self.state == PTZState.MOVING:
            elapsed = now - self._move_start_time
            if elapsed >= self._move_duration:
                # Arrived at target, now settling optics
                self.current_pos = PTZPosition(self.target_pos.pan, self.target_pos.tilt, self.target_pos.zoom)
                self.state = PTZState.SETTLING
                self._settle_start_time = now
            else:
                frac = min(1.0, elapsed / max(0.01, self._move_duration))
                self.current_pos.pan += (self.target_pos.pan - self.current_pos.pan) * frac
                self.current_pos.tilt += (self.target_pos.tilt - self.current_pos.tilt) * frac
                self.current_pos.zoom += (self.target_pos.zoom - self.current_pos.zoom) * frac

        elif self.state == PTZState.SETTLING:
            if now - self._settle_start_time >= self.settle_time:
                self.state = PTZState.STABLE
                logger.info(f"[{self.camera_id}] PTZ stabilized at Pan={self.current_pos.pan:.1f}, Tilt={self.current_pos.tilt:.1f}, Zoom={self.current_pos.zoom:.1f}x")

        return self.state

    def get_status(self) -> Dict[str, Any]:
        self.update()
        return {
            "camera_id": self.camera_id,
            "state": self.state.value,
            "pan": round(self.current_pos.pan, 2),
            "tilt": round(self.current_pos.tilt, 2),
            "zoom": round(self.current_pos.zoom, 2),
            "hardware_mode": self.hardware_mode,
            "is_stable": (self.state == PTZState.STABLE),
            "last_error": self.last_error
        }


class DisabledPTZController(PTZController):
    """Fallback controller for fixed cameras without PTZ capabilities."""
    def __init__(self, camera_id: str):
        self.camera_id = camera_id

    def slew_to(self, pan: float, tilt: float, zoom: float, speed: float = 1.0) -> bool:
        return False

    def home(self) -> bool:
        return False

    def stop(self) -> bool:
        return True

    def update(self) -> PTZState:
        return PTZState.IDLE

    def get_status(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "state": PTZState.IDLE.value,
            "pan": 0.0,
            "tilt": 0.0,
            "zoom": 1.0,
            "hardware_mode": "HARDWARE_UNAVAILABLE",
            "is_stable": True,
            "last_error": None
        }


class PTZCueEngine:
    """
    Intelligent PTZ Cue Controller.
    Implements target ranking, hysteresis anti-oscillation, cooldown rate limiting,
    and automatic return-to-home behavior.
    """
    def __init__(
        self,
        cooldown_sec: float = 4.0,
        hysteresis_margin: float = 20.0,
        lock_dwell_sec: float = 3.0,
        target_timeout_sec: float = 6.0,
        fov_h_deg: float = 60.0,
        fov_v_deg: float = 35.0
    ):
        self.cooldown_sec = cooldown_sec
        self.hysteresis_margin = hysteresis_margin
        self.lock_dwell_sec = lock_dwell_sec
        self.target_timeout_sec = target_timeout_sec
        self.fov_h = fov_h_deg
        self.fov_v = fov_v_deg

        self.controllers: Dict[str, PTZController] = {}
        self.active_targets: Dict[str, CueTarget] = {}
        self.last_cue_time: Dict[str, float] = {}
        self.last_target_seen_time: Dict[str, float] = {}

    def reset(self) -> None:
        self.controllers.clear()
        self.active_targets.clear()
        self.last_cue_time.clear()
        self.last_target_seen_time.clear()

    def get_or_create_controller(self, camera_id: str, ptz_enabled: bool = True) -> PTZController:
        if camera_id not in self.controllers:
            if ptz_enabled:
                self.controllers[camera_id] = MockPTZController(camera_id)
            else:
                self.controllers[camera_id] = DisabledPTZController(camera_id)
        return self.controllers[camera_id]

    def calculate_cue_score(
        self,
        track: Dict[str, Any],
        is_current_target: bool = False
    ) -> float:
        """
        Ranks target tracks using:
        1. Virtual fence crossing (+40)
        2. Threat class (+15 for person/firearm/drone)
        3. Risk score (0.3x)
        4. Track persistence (+5 per hit up to +20)
        5. Detector confidence (10x)
        6. Continuity bonus (+15 if currently locked)
        """
        fence_crossed = track.get("fence_violation", False) or track.get("in_restricted_zone", False)
        risk_score = float(track.get("risk_score", 0))
        cname = str(track.get("class_name", "")).lower()
        conf = float(track.get("confidence", 0.5))
        hits = min(4, int(track.get("hits", 1)))

        class_weight = 15.0 if cname in ["person", "firearm", "drone"] else 5.0
        fence_weight = 40.0 if fence_crossed else 0.0
        continuity = 15.0 if is_current_target else 0.0

        score = fence_weight + (risk_score * 0.3) + class_weight + (hits * 5.0) + (conf * 10.0) + continuity
        return score

    def rank_targets(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]]
    ) -> List[CueTarget]:
        current_locked = self.active_targets.get(camera_id)
        ranked = []

        for trk in tracks:
            tid = str(trk.get("track_id", trk.get("id", "")))
            is_curr = (current_locked is not None and current_locked.track_id == tid)
            score = self.calculate_cue_score(trk, is_current_target=is_curr)

            # Assign Priority Tier
            if score >= 65.0:
                prio = CuePriority.CRITICAL
            elif score >= 45.0:
                prio = CuePriority.HIGH
            elif score >= 25.0:
                prio = CuePriority.MEDIUM
            else:
                prio = CuePriority.LOW

            ranked.append(CueTarget(
                track_id=tid,
                camera_id=camera_id,
                class_name=trk.get("class_name", "unknown"),
                confidence=float(trk.get("confidence", 0.0)),
                bbox=trk.get("bbox", [0, 0, 0, 0]),
                risk_score=int(trk.get("risk_score", 0)),
                fence_violation=trk.get("fence_violation", False),
                priority=prio,
                cue_score=score
            ))

        ranked.sort(key=lambda x: x.cue_score, reverse=True)
        return ranked

    def bbox_to_ptz(
        self,
        bbox: List[float],
        img_w: int = 1920,
        img_h: int = 1080
    ) -> Tuple[float, float, float]:
        """Converts pixel bounding box into Pan, Tilt angles and Zoom magnification."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        bw = max(10.0, bbox[2] - bbox[0])
        bh = max(10.0, bbox[3] - bbox[1])

        # Angular offset from frame center
        dx_norm = (cx - (img_w / 2.0)) / (img_w / 2.0)   # -1.0 to +1.0
        dy_norm = (cy - (img_h / 2.0)) / (img_h / 2.0)   # -1.0 to +1.0

        pan_deg = dx_norm * (self.fov_h / 2.0)
        tilt_deg = -dy_norm * (self.fov_v / 2.0)

        # Target bounding box filling ~35% of frame height
        target_fill_h = img_h * 0.35
        zoom_level = min(15.0, max(1.0, target_fill_h / bh))

        return round(pan_deg, 2), round(tilt_deg, 2), round(zoom_level, 2)

    def evaluate_and_cue(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]],
        img_w: int = 1920,
        img_h: int = 1080
    ) -> Optional[CueTarget]:
        """
        Evaluates candidate tracks, enforces hysteresis & cooldown, and issues cue command.
        """
        ctrl = self.get_or_create_controller(camera_id)
        now = time.time()

        if not tracks:
            # Check target timeout -> return home if inactive
            if camera_id in self.active_targets:
                if now - self.last_target_seen_time.get(camera_id, now) > self.target_timeout_sec:
                    logger.info(f"[{camera_id}] Target lost for > {self.target_timeout_sec}s. Returning PTZ to HOME.")
                    ctrl.home()
                    self.active_targets.pop(camera_id, None)
            return None

        ranked = self.rank_targets(camera_id, tracks)
        if not ranked:
            return None

        best_candidate = ranked[0]
        current_target = self.active_targets.get(camera_id)

        # Only HIGH and CRITICAL priorities can automatically trigger PTZ cueing
        if best_candidate.priority not in [CuePriority.HIGH, CuePriority.CRITICAL]:
            return current_target

        # Check Target Lock & Hysteresis
        if current_target is not None:
            time_locked = now - current_target.timestamp
            is_same = (current_target.track_id == best_candidate.track_id)
            if not is_same:
                # Must beat current score by hysteresis margin AND have passed minimum lock dwell
                if (best_candidate.cue_score < current_target.cue_score + self.hysteresis_margin) or (time_locked < self.lock_dwell_sec):
                    self.last_target_seen_time[camera_id] = now
                    return current_target

        # Check Cooldown
        last_cue = self.last_cue_time.get(camera_id, 0.0)
        if (now - last_cue < self.cooldown_sec) and (current_target is not None and best_candidate.priority == current_target.priority):
            return current_target

        # Issue Slew-to-Cue Command
        pan, tilt, zoom = self.bbox_to_ptz(best_candidate.bbox, img_w, img_h)
        ctrl.slew_to(pan=pan, tilt=tilt, zoom=zoom)

        self.active_targets[camera_id] = best_candidate
        self.last_cue_time[camera_id] = now
        self.last_target_seen_time[camera_id] = now
        return best_candidate


ptz_cue_engine = PTZCueEngine()
