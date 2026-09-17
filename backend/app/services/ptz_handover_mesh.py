"""
TRINETRA — Autonomous Multi-Camera PTZ Handover Mesh & Chain Orchestrator (Phase X)
Manages multi-hop coordinated handoffs (CAM-001 -> CAM-002 -> CAM-003 -> CAM-004),
state machine transitions, anti-oscillation loop protection, and separated confidence metrics.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.entity_association_engine import (
    entity_association_engine,
    AssociationDecision
)
from backend.app.services.predictive_ptz_engine import predictive_ptz_engine, PreCueState
from backend.app.services.ptz_arbitration_engine import (
    ptz_arbitration_engine,
    CameraResourceState
)
from backend.app.services.camera_transition_predictor import camera_transition_predictor
from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("PTZHandoverMesh")


class HandoverState(str, Enum):
    PREDICTED = "PREDICTED"
    PROPOSED = "PROPOSED"
    AUTHORIZED = "AUTHORIZED"
    PRE_CUE_REQUESTED = "PRE_CUE_REQUESTED"
    PTZ_POSITIONING = "PTZ_POSITIONING"
    PTZ_READY = "PTZ_READY"
    WAITING_FOR_TARGET = "WAITING_FOR_TARGET"
    TARGET_ACQUIRED = "TARGET_ACQUIRED"
    HANDOVER_CONFIRMED = "HANDOVER_CONFIRMED"
    NEXT_HANDOVER = "NEXT_HANDOVER"

    # Terminal / Exception states
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    PTZ_FAILED = "PTZ_FAILED"
    TARGET_NOT_ACQUIRED = "TARGET_NOT_ACQUIRED"
    SOURCE_LOST = "SOURCE_LOST"
    TOPOLOGY_INVALID = "TOPOLOGY_INVALID"
    HANDOVER_LOOP_DETECTED = "HANDOVER_LOOP_DETECTED"


class HandoverConfidenceTier(str, Enum):
    CONFIRMED = "CONFIRMED"      # High association + arrival within feasible window
    PROBABLE = "PROBABLE"        # Target compatible but minor appearance ambiguity
    UNCONFIRMED = "UNCONFIRMED"  # Target arrival unverified or conflicting telemetry


@dataclass
class HandoverHop:
    hop_index: int               # 1, 2, 3, 4
    handover_id: str
    source_camera: str
    target_camera: str
    prediction_id: str
    prediction_confidence: float
    association_confidence: Optional[float] = None
    handover_confidence: HandoverConfidenceTier = HandoverConfidenceTier.UNCONFIRMED
    state: HandoverState = HandoverState.PREDICTED
    created_at: float = field(default_factory=time.time)
    expected_eta_sec: float = 15.0
    actual_arrival_sec: Optional[float] = None
    lead_time_sec: Optional[float] = None
    source_track_id: Optional[str] = None
    target_track_id: Optional[str] = None
    ptz_action_id: Optional[str] = None
    reservation_id: Optional[str] = None
    reason: str = "PREDICTIVE_HANDOVER"


@dataclass
class HandoverChain:
    chain_id: str
    entity_id: str
    incident_id: Optional[str] = None
    hops: List[HandoverHop] = field(default_factory=list)
    max_depth: int = 4
    chain_status: str = "ACTIVE" # ACTIVE, COMPLETED, PAUSED, CANCELLED, LOOP_TERMINATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class PTZHandoverMesh:
    def __init__(self, max_depth: int = 4, arbitration_engine: Optional[PTZArbitrationEngine] = None):
        self.max_depth = max_depth
        self.arbitration_engine = arbitration_engine or ptz_arbitration_engine
        self.active_chains: Dict[str, HandoverChain] = {} # entity_id -> HandoverChain
        self.active_handovers: Dict[str, HandoverHop] = {} # handover_id -> HandoverHop
        self._chain_counter = 0
        self._handover_counter = 0

    def reset(self) -> None:
        self.active_chains.clear()
        self.active_handovers.clear()
        self._chain_counter = 0
        self._handover_counter = 0

    def _next_chain_id(self) -> str:
        self._chain_counter += 1
        year = time.strftime("%Y", time.gmtime())
        return f"CHAIN-{year}-{self._chain_counter:06d}"

    def _next_handover_id(self) -> str:
        self._handover_counter += 1
        year = time.strftime("%Y", time.gmtime())
        return f"HO-{year}-{self._handover_counter:06d}"

    def get_or_create_chain(self, entity_id: str, incident_id: Optional[str] = None) -> HandoverChain:
        if entity_id in self.active_chains:
            return self.active_chains[entity_id]

        chain = HandoverChain(
            chain_id=self._next_chain_id(),
            entity_id=entity_id,
            incident_id=incident_id,
            max_depth=self.max_depth
        )
        self.active_chains[entity_id] = chain
        publish_event("chain.created", {
            "chain_id": chain.chain_id,
            "entity_id": entity_id,
            "incident_id": incident_id
        })
        logger.info(f"Initialized Handover Chain {chain.chain_id} for entity {entity_id}")
        return chain

    def detect_handover_loop(self, chain: HandoverChain, candidate_camera: str) -> bool:
        """
        Anti-oscillation protection:
        Detects if target is oscillating back-and-forth (e.g. CAM-001 -> CAM-002 -> CAM-001 -> CAM-002)
        """
        cam_sequence = [h.source_camera for h in chain.hops]
        if chain.hops:
            cam_sequence.append(chain.hops[-1].target_camera)
        cam_sequence.append(candidate_camera)

        if len(cam_sequence) >= 4:
            # Check last 4: A -> B -> A -> B
            if cam_sequence[-4] == cam_sequence[-2] and cam_sequence[-3] == cam_sequence[-1]:
                logger.warning(f"Handover loop detected in chain {chain.chain_id}: {cam_sequence[-4:]}")
                return True
        return False

    def initiate_handover(
        self,
        entity_id: str,
        source_camera: str,
        target_camera: str,
        prediction_id: str,
        prediction_confidence: float,
        expected_eta_sec: float,
        priority_score: float = 75.0,
        source_track_id: Optional[str] = None,
        incident_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str, Optional[HandoverHop]]:
        """
        Initiates a single handover hop within the entity's HandoverChain.
        Evaluates topology, max depth, loop detection, and camera arbitration.
        """
        now = timestamp if timestamp is not None else time.time()
        chain = self.get_or_create_chain(entity_id, incident_id)

        # 1. Depth Check
        if len(chain.hops) >= chain.max_depth:
            logger.info(f"Chain {chain.chain_id} reached max depth {chain.max_depth}. Handover completed.")
            chain.chain_status = "COMPLETED"
            return False, f"Maximum handover depth ({chain.max_depth}) reached", None

        # 2. Topology Check
        if not camera_topology_service.is_transition_feasible(source_camera, target_camera, elapsed_sec=expected_eta_sec)[0]:
            return False, f"Topology invalid between {source_camera} and {target_camera}", None

        # 3. Loop Detection
        if self.detect_handover_loop(chain, target_camera):
            chain.chain_status = "LOOP_TERMINATED"
            return False, "HANDOVER_LOOP_DETECTED", None

        # 4. Camera Arbitration & Reservation
        granted, arb_reason, res = self.arbitration_engine.request_camera_reservation(
            camera_id=target_camera,
            entity_id=entity_id,
            priority_score=priority_score,
            prediction_id=prediction_id,
            duration_sec=expected_eta_sec + 15.0,
            timestamp=now
        )

        ho_id = self._next_handover_id()
        hop_idx = len(chain.hops) + 1

        hop = HandoverHop(
            hop_index=hop_idx,
            handover_id=ho_id,
            source_camera=source_camera,
            target_camera=target_camera,
            prediction_id=prediction_id,
            prediction_confidence=prediction_confidence,
            state=HandoverState.PROPOSED,
            created_at=now,
            expected_eta_sec=expected_eta_sec,
            source_track_id=source_track_id,
            reservation_id=res.reservation_id if res else None,
            reason=f"Corridor handover {source_camera} -> {target_camera}"
        )

        if not granted:
            hop.state = HandoverState.REJECTED
            hop.reason = f"Arbitration contention: {arb_reason}"
            logger.warning(f"Handover {ho_id} rejected on {target_camera}: {arb_reason}")
            return False, arb_reason, hop

        hop.state = HandoverState.AUTHORIZED

        # 5. Issue Predictive PTZ Pre-Cue
        cued, cue_reason, ptz_act = predictive_ptz_engine.evaluate_and_precue(
            prediction_id=prediction_id,
            entity_id=entity_id,
            target_camera=target_camera,
            confidence=prediction_confidence,
            eta_sec=expected_eta_sec,
            timestamp=now
        )

        if cued and ptz_act:
            hop.state = HandoverState.PTZ_POSITIONING
            hop.ptz_action_id = ptz_act.action_id
        else:
            hop.state = HandoverState.WAITING_FOR_TARGET
            hop.reason += f" (PTZ skipped: {cue_reason})"

        chain.hops.append(hop)
        self.active_handovers[ho_id] = hop

        publish_event("handover.created", {
            "chain_id": chain.chain_id,
            "handover_id": ho_id,
            "hop": hop_idx,
            "source_camera": source_camera,
            "target_camera": target_camera,
            "state": hop.state.value,
            "eta_sec": expected_eta_sec
        })

        logger.info(f"Handover {ho_id} [Hop {hop_idx}] {source_camera} -> {target_camera} AUTHORIZED and in progress.")
        return True, "HANDOVER_AUTHORIZED", hop

    def notify_camera_ready(self, camera_id: str, timestamp: Optional[float] = None) -> None:
        """Called when PTZ settling completes and camera is poised at anticipation vector."""
        for hop in self.active_handovers.values():
            if hop.target_camera == camera_id and hop.state in [HandoverState.PTZ_POSITIONING, HandoverState.AUTHORIZED]:
                hop.state = HandoverState.PTZ_READY
                publish_event("handover.ptz_ready", {
                    "handover_id": hop.handover_id,
                    "target_camera": camera_id
                })
                logger.info(f"Handover {hop.handover_id} on {camera_id} reached PTZ_READY state.")

    def confirm_acquisition(
        self,
        target_camera: str,
        entity_id: str,
        target_track_id: str,
        association_score: float = 0.85,
        timestamp: Optional[float] = None
    ) -> Optional[HandoverHop]:
        """
        Confirms target acquisition on destination camera and verifies handover confirmation.
        Locks destination camera, updates lead time, and triggers next hop preparation.
        """
        now = timestamp if timestamp is not None else time.time()

        # Find matching active hop
        hop = next((
            h for h in self.active_handovers.values()
            if h.target_camera == target_camera and h.state in [
                HandoverState.PTZ_READY, HandoverState.WAITING_FOR_TARGET, HandoverState.PTZ_POSITIONING
            ]
        ), None)

        if not hop:
            return None

        elapsed = max(0.1, now - hop.created_at)
        hop.actual_arrival_sec = round(elapsed, 2)
        hop.target_track_id = target_track_id
        hop.association_confidence = round(association_score, 2)

        # 1. Evaluate Handover Confidence (Section 9)
        if association_score >= 0.70 and elapsed <= (hop.expected_eta_sec + 15.0):
            hop.handover_confidence = HandoverConfidenceTier.CONFIRMED
            hop.state = HandoverState.HANDOVER_CONFIRMED
        elif association_score >= 0.50:
            hop.handover_confidence = HandoverConfidenceTier.PROBABLE
            hop.state = HandoverState.HANDOVER_CONFIRMED
        else:
            hop.handover_confidence = HandoverConfidenceTier.UNCONFIRMED
            hop.state = HandoverState.TARGET_ACQUIRED

        # 2. Lock destination camera and compute lead time
        self.arbitration_engine.lock_camera(target_camera, entity_id)
        lead = predictive_ptz_engine.record_target_observation(target_camera, entity_id, timestamp=now)
        hop.lead_time_sec = lead

        # 3. Release source camera back to available
        self.arbitration_engine.release_camera(hop.source_camera, reason="HANDOVER_COMPLETED")

        publish_event("handover.confirmed", {
            "handover_id": hop.handover_id,
            "source_camera": hop.source_camera,
            "target_camera": target_camera,
            "handover_confidence": hop.handover_confidence.value,
            "lead_time_sec": lead,
            "arrival_sec": elapsed
        })

        logger.info(f"Handover {hop.handover_id} CONFIRMED on {target_camera} ({hop.handover_confidence.value}, Lead: {lead}s)")
        return hop

    def cancel_handover(self, handover_id: str, reason: str = "OPERATOR_OVERRIDE") -> bool:
        hop = self.active_handovers.get(handover_id)
        if not hop:
            return False

        hop.state = HandoverState.CANCELLED
        hop.reason = reason
        self.arbitration_engine.release_camera(hop.target_camera, reason=reason)
        predictive_ptz_engine.cancel_pre_cue(hop.target_camera, reason=reason)

        publish_event("handover.cancelled", {
            "handover_id": handover_id,
            "reason": reason
        })
        logger.info(f"Handover {handover_id} CANCELLED: {reason}")
        return True

    def pause_chain(self, chain_id: str) -> bool:
        chain = next((c for c in self.active_chains.values() if c.chain_id == chain_id), None)
        if not chain:
            return False
        chain.chain_status = "PAUSED"
        logger.info(f"Handover Chain {chain_id} PAUSED by operator.")
        return True

    def resume_chain(self, chain_id: str) -> bool:
        chain = next((c for c in self.active_chains.values() if c.chain_id == chain_id), None)
        if not chain:
            return False
        chain.chain_status = "ACTIVE"
        logger.info(f"Handover Chain {chain_id} RESUMED by operator.")
        return True


ptz_handover_mesh = PTZHandoverMesh()
