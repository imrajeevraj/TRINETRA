"""
TRINETRA Phase XIV — Multimodal Tracking & Global Entity Lineage Service
Synthesizes global multimodal track entities from optical and thermal track streams.
Preserves immutable camera-qualified track IDs and full modality lineage.
"""

from __future__ import annotations
import math
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("MultimodalTracking")


class CrossSpectralMatchTier(str, Enum):
    SAME_ENTITY_LIKELY = "SAME_ENTITY_LIKELY"
    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    UNRELATED = "UNRELATED"
    UNKNOWN = "UNKNOWN"


@dataclass
class SingleModalityTrack:
    track_id: str                      # E.g. "CAM-001:RGB:P-024"
    camera_id: str
    modality: str                      # "RGB" or "LWIR"
    class_name: str
    bbox: List[float]                  # [x1, y1, x2, y2]
    velocity_mps: float
    heading_deg: float
    confidence: float
    last_update: float = field(default_factory=time.time)


@dataclass
class GlobalMultimodalEntity:
    global_entity_id: str              # E.g. "GLOBAL-PERSON-00001"
    camera_id: str
    class_name: str
    optical_track_id: Optional[str]
    thermal_track_id: Optional[str]
    match_tier: CrossSpectralMatchTier
    kinematic_match_score: float
    current_bbox: List[float]
    velocity_mps: float
    heading_deg: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    status: str = "ACTIVE"
    lineage_history: List[Dict[str, Any]] = field(default_factory=list)


class MultimodalTrackingService:
    """
    Associates optical and thermal tracks without replacing original camera-qualified identities.
    """

    def __init__(self):
        self._global_entities: Dict[str, GlobalMultimodalEntity] = {}
        self._global_seq = 0

    def evaluate_track_association(
        self,
        optical_track: SingleModalityTrack,
        thermal_track: SingleModalityTrack,
    ) -> Tuple[CrossSpectralMatchTier, float]:
        """
        Evaluates spatial, velocity, and heading compatibility between tracks.
        """
        # 1. Bounding box distance (center-to-center)
        c_opt = [(optical_track.bbox[0] + optical_track.bbox[2])/2.0, (optical_track.bbox[1] + optical_track.bbox[3])/2.0]
        c_thm = [(thermal_track.bbox[0] + thermal_track.bbox[2])/2.0, (thermal_track.bbox[1] + thermal_track.bbox[3])/2.0]
        dist = math.hypot(c_opt[0] - c_thm[0], c_opt[1] - c_thm[1])
        spatial_score = max(0.0, 1.0 - (dist / 0.30))

        # 2. Velocity difference
        vel_diff = abs(optical_track.velocity_mps - thermal_track.velocity_mps)
        vel_score = max(0.0, 1.0 - (vel_diff / 2.0))

        # 3. Heading angular difference
        h_diff = abs(optical_track.heading_deg - thermal_track.heading_deg) % 360.0
        if h_diff > 180.0:
            h_diff = 360.0 - h_diff
        heading_score = max(0.0, 1.0 - (h_diff / 45.0))

        # Combined kinematic score
        total_score = round(0.45 * spatial_score + 0.30 * vel_score + 0.25 * heading_score, 4)

        if total_score >= 0.80:
            tier = CrossSpectralMatchTier.SAME_ENTITY_LIKELY
        elif total_score >= 0.60:
            tier = CrossSpectralMatchTier.POSSIBLE_MATCH
        elif total_score >= 0.30:
            tier = CrossSpectralMatchTier.UNKNOWN
        else:
            tier = CrossSpectralMatchTier.UNRELATED

        return tier, total_score

    def register_or_update_entity(
        self,
        camera_id: str,
        class_name: str,
        optical_track: Optional[SingleModalityTrack] = None,
        thermal_track: Optional[SingleModalityTrack] = None,
    ) -> GlobalMultimodalEntity:
        """
        Associates or creates a global entity while preserving camera-qualified track IDs.
        """
        now = time.time()

        # Check existing match
        for entity in self._global_entities.values():
            if entity.camera_id != camera_id or entity.status != "ACTIVE":
                continue

            # Check matching track IDs
            if optical_track and entity.optical_track_id == optical_track.track_id:
                if thermal_track:
                    tier, score = self.evaluate_track_association(optical_track, thermal_track)
                    entity.thermal_track_id = thermal_track.track_id
                    entity.match_tier = tier
                    entity.kinematic_match_score = score
                entity.last_seen = now
                return entity

            if thermal_track and entity.thermal_track_id == thermal_track.track_id:
                if optical_track:
                    tier, score = self.evaluate_track_association(optical_track, thermal_track)
                    entity.optical_track_id = optical_track.track_id
                    entity.match_tier = tier
                    entity.kinematic_match_score = score
                entity.last_seen = now
                return entity

        # Check kinematic fusion if both tracks provided
        if optical_track and thermal_track:
            tier, score = self.evaluate_track_association(optical_track, thermal_track)
        else:
            tier = CrossSpectralMatchTier.UNKNOWN
            score = 0.50

        self._global_seq += 1
        gid = f"GLOBAL-{class_name.upper()}-{self._global_seq:05d}"

        bbox = optical_track.bbox if optical_track else (thermal_track.bbox if thermal_track else [0, 0, 0, 0])
        vel = optical_track.velocity_mps if optical_track else (thermal_track.velocity_mps if thermal_track else 1.3)
        heading = optical_track.heading_deg if optical_track else (thermal_track.heading_deg if thermal_track else 0.0)

        entity = GlobalMultimodalEntity(
            global_entity_id=gid,
            camera_id=camera_id,
            class_name=class_name,
            optical_track_id=optical_track.track_id if optical_track else None,
            thermal_track_id=thermal_track.track_id if thermal_track else None,
            match_tier=tier,
            kinematic_match_score=score,
            current_bbox=bbox,
            velocity_mps=vel,
            heading_deg=heading,
            lineage_history=[{
                "timestamp": now,
                "opt_id": optical_track.track_id if optical_track else None,
                "thm_id": thermal_track.track_id if thermal_track else None,
                "score": score,
                "tier": tier.value,
            }],
        )

        self._global_entities[gid] = entity
        return entity

    def get_entity(self, global_entity_id: str) -> Optional[GlobalMultimodalEntity]:
        return self._global_entities.get(global_entity_id)

    def list_active_entities(self, camera_id: Optional[str] = None) -> List[GlobalMultimodalEntity]:
        entities = [e for e in self._global_entities.values() if e.status == "ACTIVE"]
        if camera_id:
            entities = [e for e in entities if e.camera_id == camera_id]
        return entities

    def reset(self):
        self._global_entities.clear()
        self._global_seq = 0


multimodal_tracking_service = MultimodalTrackingService()
