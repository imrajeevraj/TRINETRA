"""
TRINETRA — Cross-Camera Entity Association Engine (Phase VIII)
Probabilistically correlates local camera-qualified tracks into Global Entities
(e.g., GLOBAL-PERSON-00001, GLOBAL-VEHICLE-00001) while strictly preserving
underlying camera track lineage (CAM-001:P-024).
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.services.camera_topology import camera_topology_service

logger = logging.getLogger("EntityAssociationEngine")


class AssociationDecision(str, Enum):
    SAME_ENTITY_LIKELY = "SAME_ENTITY_LIKELY"   # Score >= 0.75
    POSSIBLE_MATCH = "POSSIBLE_MATCH"           # 0.50 <= Score < 0.75
    UNRELATED = "UNRELATED"                     # Score < 0.50
    UNKNOWN = "UNKNOWN"


@dataclass
class LocalTrackObservation:
    camera_id: str
    track_id: str                      # Local track ID e.g. "P-024"
    class_name: str                    # "person", "vehicle", etc.
    timestamp: float = field(default_factory=time.time)
    bbox: List[float] = field(default_factory=list) # [x1, y1, x2, y2]
    direction: Optional[str] = None
    plate_number: Optional[str] = None
    face_id: Optional[str] = None
    confidence: float = 0.5


@dataclass
class GlobalEntity:
    global_id: str                     # e.g. "GLOBAL-PERSON-00001"
    entity_type: str                   # "person", "vehicle", etc.
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    observations: List[LocalTrackObservation] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "ACTIVE"             # "ACTIVE", "RESOLVED", "MERGED"


class EntityAssociationEngine:
    def __init__(
        self,
        same_entity_thresh: float = 0.70,
        possible_match_thresh: float = 0.50,
        max_association_gap_sec: float = 180.0
    ):
        self.same_entity_thresh = same_entity_thresh
        self.possible_match_thresh = possible_match_thresh
        self.max_gap_sec = max_association_gap_sec

        self.global_entities: Dict[str, GlobalEntity] = {}
        self._counter: Dict[str, int] = {"person": 0, "vehicle": 0, "aircraft": 0, "unknown": 0}

    def _next_global_id(self, entity_type: str) -> str:
        etype = entity_type.lower() if entity_type.lower() in self._counter else "unknown"
        self._counter[etype] += 1
        return f"GLOBAL-{etype.upper()}-{self._counter[etype]:05d}"

    def compute_aspect_ratio(self, bbox: List[float]) -> float:
        if not bbox or len(bbox) < 4:
            return 1.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        return w / h

    def evaluate_association(
        self,
        obs1: LocalTrackObservation,
        obs2: LocalTrackObservation
    ) -> Tuple[float, AssociationDecision, List[str], List[str]]:
        """
        Evaluates probabilistic match between two observations across cameras.
        Returns: (score, decision, supporting_evidence, conflicting_evidence)
        """
        supporting = []
        conflicting = []

        # 1. Class mismatch is hard disqualifier
        if obs1.class_name.lower() != obs2.class_name.lower():
            conflicting.append(f"Class mismatch: {obs1.class_name} vs {obs2.class_name}")
            return 0.0, AssociationDecision.UNRELATED, supporting, conflicting

        # 2. Simultaneous distinct tracks on same camera cannot be same entity
        t_delta = abs(obs2.timestamp - obs1.timestamp)
        if obs1.camera_id == obs2.camera_id and obs1.track_id != obs2.track_id and t_delta < 5.0:
            conflicting.append(f"Simultaneous co-occurring tracks on {obs1.camera_id} ({obs1.track_id} & {obs2.track_id})")
            return 0.05, AssociationDecision.UNRELATED, supporting, conflicting

        # 3. Time delta check
        t_delta = abs(obs2.timestamp - obs1.timestamp)
        if t_delta > self.max_gap_sec:
            conflicting.append(f"Observation gap too large: {t_delta:.1f}s > {self.max_gap_sec}s")
            return 0.1, AssociationDecision.UNRELATED, supporting, conflicting

        # Order chronologically
        earlier = obs1 if obs1.timestamp <= obs2.timestamp else obs2
        later = obs2 if obs1.timestamp <= obs2.timestamp else obs1
        elapsed = later.timestamp - earlier.timestamp

        score = 0.20  # Base prior for same class in perimeter

        # 3. Topology Feasibility
        is_feasible, topo_reason = camera_topology_service.is_transition_feasible(
            from_cam=earlier.camera_id,
            to_cam=later.camera_id,
            elapsed_sec=elapsed
        )
        if is_feasible:
            score += 0.35
            supporting.append(f"Topology feasible ({earlier.camera_id} -> {later.camera_id} in {elapsed:.1f}s: {topo_reason})")
        else:
            score -= 0.30
            conflicting.append(f"Topology violation ({topo_reason})")

        # 4. Aspect Ratio & Geometric Consistency
        ar1 = self.compute_aspect_ratio(obs1.bbox)
        ar2 = self.compute_aspect_ratio(obs2.bbox)
        ar_diff = abs(ar1 - ar2)
        if ar_diff < 0.30:
            score += 0.15
            supporting.append(f"Consistent geometric aspect ratio ({ar1:.2f} vs {ar2:.2f})")
        else:
            conflicting.append(f"Aspect ratio discrepancy ({ar1:.2f} vs {ar2:.2f})")

        # 5. Vehicle ANPR Match (Decisive booster)
        if obs1.plate_number and obs2.plate_number:
            if obs1.plate_number.upper() == obs2.plate_number.upper():
                score += 0.40
                supporting.append(f"Confirmed ANPR plate match: {obs1.plate_number}")
            else:
                score -= 0.40
                conflicting.append(f"Different license plates: {obs1.plate_number} vs {obs2.plate_number}")

        # 6. Face Embedding Match
        if obs1.face_id and obs2.face_id:
            if obs1.face_id == obs2.face_id:
                score += 0.35
                supporting.append(f"Consistent face observation: {obs1.face_id}")

        score = max(0.0, min(0.99, score))

        if score >= self.same_entity_thresh:
            decision = AssociationDecision.SAME_ENTITY_LIKELY
        elif score >= self.possible_match_thresh:
            decision = AssociationDecision.POSSIBLE_MATCH
        else:
            decision = AssociationDecision.UNRELATED

        # 7. Low confidence observation cap (Section 13 & 14)
        if obs1.confidence < 0.50 or obs2.confidence < 0.50:
            if decision == AssociationDecision.SAME_ENTITY_LIKELY:
                decision = AssociationDecision.POSSIBLE_MATCH
                conflicting.append("Capped to POSSIBLE_MATCH due to low observation confidence (< 0.50)")

        return round(score, 2), decision, supporting, conflicting

    def ingest_observation(
        self,
        camera_id: str,
        track_id: str,
        class_name: str,
        bbox: List[float],
        timestamp: Optional[float] = None,
        direction: Optional[str] = None,
        plate_number: Optional[str] = None,
        face_id: Optional[str] = None,
        confidence: float = 0.5
    ) -> Tuple[GlobalEntity, AssociationDecision, str]:
        """
        Ingests a local camera track observation, compares against existing active
        Global Entities, and associates or instantiates a new Global Entity.
        """
        now = timestamp if timestamp is not None else time.time()
        obs = LocalTrackObservation(
            camera_id=camera_id,
            track_id=track_id,
            class_name=class_name,
            timestamp=now,
            bbox=bbox,
            direction=direction,
            plate_number=plate_number,
            face_id=face_id,
            confidence=confidence
        )

        best_entity: Optional[GlobalEntity] = None
        best_score = 0.0
        best_decision = AssociationDecision.UNRELATED
        best_reason = ""

        # Search active global entities of matching class
        for entity in self.global_entities.values():
            if entity.entity_type.lower() != class_name.lower():
                continue
            if now - entity.last_seen > self.max_gap_sec:
                continue

            # Compare against the entity's latest observation
            last_obs = entity.observations[-1]
            score, dec, supp, conf = self.evaluate_association(last_obs, obs)

            if score > best_score:
                best_score = score
                best_decision = dec
                best_entity = entity
                best_reason = "; ".join(supp) if supp else "; ".join(conf)

        # Associate or Create
        if best_entity is not None and best_decision in [AssociationDecision.SAME_ENTITY_LIKELY, AssociationDecision.POSSIBLE_MATCH]:
            best_entity.observations.append(obs)
            best_entity.last_seen = now
            best_entity.confidence = round((best_entity.confidence + confidence) / 2.0, 2)
            logger.info(f"Associated {camera_id}:{track_id} to {best_entity.global_id} ({best_decision.value}, score: {best_score})")
            return best_entity, best_decision, best_reason
        else:
            # Instantiate new global entity
            new_gid = self._next_global_id(class_name)
            new_entity = GlobalEntity(
                global_id=new_gid,
                entity_type=class_name,
                first_seen=now,
                last_seen=now,
                observations=[obs],
                confidence=confidence
            )
            self.global_entities[new_gid] = new_entity
            logger.info(f"Created new Global Entity: {new_gid} for {camera_id}:{track_id}")
            return new_entity, AssociationDecision.UNRELATED, "New entity initialized"

    def get_entity_timeline(self, global_id: str) -> Optional[List[Dict[str, Any]]]:
        entity = self.global_entities.get(global_id)
        if not entity:
            return None
        timeline = []
        for obs in sorted(entity.observations, key=lambda x: x.timestamp):
            timeline.append({
                "camera_id": obs.camera_id,
                "track_id": f"{obs.camera_id}:{obs.track_id}",
                "timestamp": obs.timestamp,
                "direction": obs.direction or "UNKNOWN",
                "plate_number": obs.plate_number,
                "confidence": obs.confidence,
                "bbox": obs.bbox
            })
        return timeline


entity_association_engine = EntityAssociationEngine()
