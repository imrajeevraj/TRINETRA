"""
TRINETRA — Temporal Threat Confirmation & Multi-Stage Confidence Fusion Engine (Phase VII)
Coordinates track state machine, multi-frame evidence accumulation,
explainable confidence fusion, and per-camera deployment profiles.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("TemporalThreatConfirmation")


class TrackThreatState(str, Enum):
    NEW = "NEW"                     # Single frame observation
    OBSERVED = "OBSERVED"           # 2-3 frames observed
    SUSPICIOUS = "SUSPICIOUS"       # Approach to fence or high risk detected
    CONFIRMED = "CONFIRMED"         # Multi-frame + ROI confirmed threat
    ALERTED = "ALERTED"             # Dispatched to operator/system
    RESOLVED = "RESOLVED"           # Track exited or confirmed benign


class AlertTier(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HumanReviewStatus(str, Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED_BY_OPERATOR = "CONFIRMED_BY_OPERATOR"
    REJECTED_BY_OPERATOR = "REJECTED_BY_OPERATOR"
    AUTO_CONFIRMED = "AUTO_CONFIRMED"
    EXPIRED = "EXPIRED"


class DeploymentProfile(str, Enum):
    PROFILE_A_PERIMETER = "PROFILE_A"       # Raw detector, no global tool filter
    PROFILE_B_LOGISTICS = "PROFILE_B"       # Tool-heavy gate, threat verifier enabled
    PROFILE_C_INVESTIGATION = "PROFILE_C"   # Manual investigation, full verifier


@dataclass
class TemporalTrackRecord:
    track_id: str
    camera_id: str
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    observation_count: int = 1
    state: TrackThreatState = TrackThreatState.NEW
    confidence_history: List[float] = field(default_factory=list)
    risk_history: List[int] = field(default_factory=list)
    fence_violations: int = 0
    roi_confirmations: int = 0
    alert_tier: AlertTier = AlertTier.INFO
    review_status: HumanReviewStatus = HumanReviewStatus.PENDING_REVIEW
    operator_notes: Optional[str] = None
    last_bbox: List[float] = field(default_factory=list)
    fused_confidence: float = 0.0
    alert_reason: str = ""


class TemporalThreatEngine:
    def __init__(
        self,
        min_confirmed_frames: int = 4,
        track_timeout_sec: float = 8.0,
        high_risk_threshold: int = 70,
        critical_risk_threshold: int = 85
    ):
        self.min_confirmed = min_confirmed_frames
        self.track_timeout = track_timeout_sec
        self.high_risk_thresh = high_risk_threshold
        self.critical_risk_thresh = critical_risk_threshold

        # In-memory track history: (camera_id, track_id) -> TemporalTrackRecord
        self.tracks: Dict[Tuple[str, str], TemporalTrackRecord] = {}

    def update_track(
        self,
        camera_id: str,
        track_id: str,
        class_name: str,
        confidence: float,
        bbox: List[float],
        risk_score: int = 0,
        fence_violation: bool = False,
        roi_confirmed: bool = False,
        profile: DeploymentProfile = DeploymentProfile.PROFILE_A_PERIMETER
    ) -> TemporalTrackRecord:
        """
        Updates temporal state machine, accumulates multi-frame evidence,
        and executes transparent confidence fusion.
        """
        now = time.time()
        key = (camera_id, track_id)

        if key not in self.tracks:
            rec = TemporalTrackRecord(
                track_id=track_id,
                camera_id=camera_id,
                class_name=class_name,
                first_seen=now,
                last_seen=now,
                observation_count=1,
                state=TrackThreatState.NEW,
                confidence_history=[confidence],
                risk_history=[risk_score],
                fence_violations=1 if fence_violation else 0,
                roi_confirmations=1 if roi_confirmed else 0,
                last_bbox=bbox
            )
            self.tracks[key] = rec
        else:
            rec = self.tracks[key]
            rec.last_seen = now
            rec.observation_count += 1
            rec.confidence_history.append(confidence)
            rec.risk_history.append(risk_score)
            rec.last_bbox = bbox
            if fence_violation:
                rec.fence_violations += 1
            if roi_confirmed:
                rec.roi_confirmations += 1

        # 1. State Machine Transitions (Single transition per frame)
        obs = rec.observation_count
        if rec.state == TrackThreatState.NEW and obs >= 2:
            rec.state = TrackThreatState.OBSERVED
        elif rec.state == TrackThreatState.OBSERVED:
            if fence_violation or risk_score >= 40:
                rec.state = TrackThreatState.SUSPICIOUS
        elif rec.state == TrackThreatState.SUSPICIOUS:
            if (obs >= self.min_confirmed and rec.fence_violations >= 1) or (risk_score >= self.critical_risk_thresh):
                rec.state = TrackThreatState.CONFIRMED
        elif rec.state == TrackThreatState.CONFIRMED and obs >= (self.min_confirmed + 1):
            rec.state = TrackThreatState.ALERTED

        # 2. Multi-Factor Transparent Confidence Fusion
        fused, reason = self.fuse_confidence(rec, profile)
        rec.fused_confidence = fused
        rec.alert_reason = reason

        # 3. Alert Tier Mapping
        if risk_score >= self.critical_risk_thresh or (rec.fence_violations >= 2 and fused >= 0.85):
            rec.alert_tier = AlertTier.CRITICAL
        elif risk_score >= self.high_risk_thresh or rec.fence_violations >= 1:
            rec.alert_tier = AlertTier.HIGH
        elif risk_score >= 40 or rec.state == TrackThreatState.SUSPICIOUS:
            rec.alert_tier = AlertTier.MEDIUM
        elif obs >= 3:
            rec.alert_tier = AlertTier.LOW
        else:
            rec.alert_tier = AlertTier.INFO

        return rec

    def fuse_confidence(
        self,
        record: TemporalTrackRecord,
        profile: DeploymentProfile
    ) -> Tuple[float, str]:
        """
        Calculates explainable fused confidence from:
        - Base detector average confidence (0.4x)
        - Temporal persistence streak (0.25x)
        - Virtual fence crossing confirmation (0.2x)
        - High-resolution ROI confirmation (0.15x)
        """
        mean_conf = float(np.mean(record.confidence_history[-5:]))
        streak_bonus = min(0.25, (record.observation_count / 10.0) * 0.25)
        fence_bonus = 0.20 if record.fence_violations > 0 else 0.0
        roi_bonus = 0.15 if record.roi_confirmations > 0 else 0.0

        fused = min(0.99, (mean_conf * 0.40) + streak_bonus + fence_bonus + roi_bonus)

        reasons = [f"Base conf: {mean_conf:.2f}"]
        if record.observation_count >= 3:
            reasons.append(f"persistent over {record.observation_count} frames")
        if record.fence_violations > 0:
            reasons.append(f"crossed virtual fence ({record.fence_violations}x)")
        if record.roi_confirmations > 0:
            reasons.append("verified via high-res ROI")

        reason_str = f"Track {record.track_id} ({record.class_name}) " + "; ".join(reasons) + f" [Profile: {profile.value}]"
        return round(fused, 2), reason_str

    def record_human_disposition(
        self,
        camera_id: str,
        track_id: str,
        disposition: str,
        notes: Optional[str] = None
    ) -> Optional[TemporalTrackRecord]:
        """Auditable human review status update."""
        key = (camera_id, track_id)
        if key not in self.tracks:
            return None
        rec = self.tracks[key]

        if disposition.upper() == "CONFIRMED":
            rec.review_status = HumanReviewStatus.CONFIRMED_BY_OPERATOR
        elif disposition.upper() == "REJECTED":
            rec.review_status = HumanReviewStatus.REJECTED_BY_OPERATOR
            rec.state = TrackThreatState.RESOLVED
        rec.operator_notes = notes
        logger.info(f"[{camera_id}] Track {track_id} disposition updated to {rec.review_status.value} by operator (Notes: {notes})")
        return rec

    def prune_stale_tracks(self):
        """Prunes tracks older than timeout."""
        now = time.time()
        to_del = [k for k, v in self.tracks.items() if (now - v.last_seen) > self.track_timeout]
        for k in to_del:
            self.tracks.pop(k, None)


temporal_threat_engine = TemporalThreatEngine()
