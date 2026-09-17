"""
TRINETRA — Edge Camera Node & Distributed Mesh Manager (Phase XI)
Represents local edge computing sentries, short-lived authorization leases,
heartbeats, local outbox buffering, peer handover negotiation, and central fallback.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set

from backend.app.services.camera_topology import camera_topology_service
from backend.app.services.edge_handover_protocol import (
    edge_handover_protocol_engine,
    PeerMessageType,
    PeerHandoverMessage
)
from backend.app.services.ptz_arbitration_engine import (
    ptz_arbitration_engine,
    CameraResourceState
)
from backend.app.services.ptz_handover_mesh import (
    ptz_handover_mesh,
    HandoverState,
    HandoverConfidenceTier
)
from backend.app.core.events_pubsub import publish_event

logger = logging.getLogger("EdgeCameraNode")


class NetworkPartitionMode(str, Enum):
    NORMAL = "NORMAL"
    DEGRADED = "DEGRADED"
    PARTITIONED = "PARTITIONED"
    RECOVERING = "RECOVERING"


class NodeHealth(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


@dataclass
class EdgeAuthorizationLease:
    lease_id: str
    node_id: str
    issued_at: float
    expires_at: float
    allowed_operations: List[str] = field(default_factory=lambda: ["PEER_HANDOVER", "PTZ_PRECUE", "LOCAL_TRACK"])
    policy_version: str = "2.0.0"

    def is_valid(self, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        return now < self.expires_at


@dataclass
class EdgeOutboxItem:
    event_id: str
    node_id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    attempt_count: int = 0
    last_attempt: Optional[float] = None
    status: str = "PENDING"  # PENDING, SENT, ACKNOWLEDGED, FAILED, EXPIRED


class EdgeCameraNode:
    """
    Independent edge compute agent attached to a camera sentry.
    Executes local perception, track continuity, peer negotiation,
    and outbox buffering during control plane disconnection.
    """
    def __init__(self, camera_id: str, node_id: Optional[str] = None):
        self.camera_id = camera_id
        self.node_id = node_id or f"NODE-{camera_id}"
        self.capabilities = camera_topology_service.get_camera_capabilities(camera_id)
        self.neighbor_cameras = camera_topology_service.get_authorized_peers(camera_id)
        self.health = NodeHealth.ONLINE
        self.network_state = NetworkPartitionMode.NORMAL
        self.clock_offset_ms: float = 0.0
        self.authorization_lease: Optional[EdgeAuthorizationLease] = None
        self.outbox: List[EdgeOutboxItem] = []
        self.local_tracks: Dict[str, Any] = {}
        self.local_handovers: Dict[str, Dict[str, Any]] = {}
        self.is_enabled: bool = True
        self.global_ptz_lock_active: bool = False
        self.last_control_plane_sync: float = time.time()
        self.last_peer_sync: float = time.time()
        self._event_counter = 0

        # Grant default 5-minute lease upon instantiation
        self.renew_lease(ttl_sec=300.0)

    def renew_lease(self, ttl_sec: float = 300.0, timestamp: Optional[float] = None) -> EdgeAuthorizationLease:
        now = timestamp if timestamp is not None else time.time()
        lease = EdgeAuthorizationLease(
            lease_id=f"LEASE-{self.node_id}-{int(now)}",
            node_id=self.node_id,
            issued_at=now,
            expires_at=now + ttl_sec
        )
        self.authorization_lease = lease
        return lease

    def has_active_autonomy(self, current_time: Optional[float] = None) -> bool:
        """Determines if the edge node has legal autonomy to coordinate handovers."""
        if not self.is_enabled:
            return False
        if self.global_ptz_lock_active:
            return False
        if self.health == NodeHealth.OFFLINE:
            return False
        now = current_time if current_time is not None else time.time()
        if self.authorization_lease and not self.authorization_lease.is_valid(now):
            return False
        return True

    def buffer_event(self, event_type: str, payload: Dict[str, Any], timestamp: Optional[float] = None) -> EdgeOutboxItem:
        self._event_counter += 1
        now = timestamp if timestamp is not None else time.time()
        item = EdgeOutboxItem(
            event_id=f"EVT-{self.node_id}-{int(now * 1000)}-{self._event_counter:04d}",
            node_id=self.node_id,
            event_type=event_type,
            payload=payload,
            created_at=now
        )
        self.outbox.append(item)
        return item

    def send_heartbeat(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        now = current_time if current_time is not None else time.time()
        return {
            "node_id": self.node_id,
            "camera_id": self.camera_id,
            "health": self.health.value,
            "network_state": self.network_state.value,
            "clock_offset_ms": self.clock_offset_ms,
            "lease_valid": self.has_active_autonomy(now),
            "lease_expires_in_sec": max(0.0, self.authorization_lease.expires_at - now) if self.authorization_lease else 0.0,
            "outbox_depth": len([o for o in self.outbox if o.status == "PENDING"]),
            "timestamp": now,
            "protocol_version": "1.0.0"
        }

    def propose_peer_handover(
        self,
        entity_id: str,
        target_camera: str,
        prediction_id: str,
        confidence: float,
        expected_eta_sec: float,
        priority_score: float = 75.0,
        chain_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str, Optional[PeerHandoverMessage]]:
        """
        Initiates a peer-to-peer handover proposal directly to a neighboring camera node.
        """
        now = timestamp if timestamp is not None else time.time()

        if not self.has_active_autonomy(now):
            reason = "EDGE_AUTONOMY_DISABLED_OR_LEASE_EXPIRED"
            logger.warning(f"Node {self.node_id} cannot propose handover: {reason}")
            return False, reason, None

        if target_camera not in self.neighbor_cameras:
            reason = f"TARGET_CAMERA_{target_camera}_NOT_IN_NEIGHBORS"
            logger.warning(f"Topological violation: {self.camera_id} -> {target_camera} is not an authorized peer.")
            return False, reason, None

        # Check clock offset
        if abs(self.clock_offset_ms) > 500.0:
            logger.warning(f"Clock offset {self.clock_offset_ms}ms exceeds 500ms tolerance; degrading confidence.")
            confidence = max(0.1, confidence - 0.20)

        c_id = chain_id or f"CHAIN-{int(now)}-{entity_id}"
        h_id = f"PEER-HO-{self.camera_id}-{target_camera}-{int(now * 1000)}"

        msg = edge_handover_protocol_engine.create_message(
            message_type=PeerMessageType.HANDOVER_PROPOSE,
            handover_id=h_id,
            chain_id=c_id,
            source_node=self.camera_id,
            destination_node=target_camera,
            entity_id=entity_id,
            prediction_id=prediction_id,
            confidence=confidence,
            ttl_sec=max(15.0, expected_eta_sec + 10.0),
            payload={
                "expected_eta_sec": expected_eta_sec,
                "priority_score": priority_score,
                "timestamp": now
            },
            timestamp=now
        )

        self.local_handovers[h_id] = {
            "handover_id": h_id,
            "chain_id": c_id,
            "target_camera": target_camera,
            "entity_id": entity_id,
            "state": HandoverState.PROPOSED.value,
            "created_at": now
        }

        self.buffer_event("edge.handover_proposed", {
            "handover_id": h_id,
            "source": self.camera_id,
            "target": target_camera,
            "entity_id": entity_id
        }, timestamp=now)

        return True, "PROPOSAL_CREATED", msg

    def handle_incoming_peer_message(
        self,
        msg: PeerHandoverMessage,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str, Optional[PeerHandoverMessage]]:
        """
        Receives and processes an incoming message from a neighboring edge node.
        """
        now = current_time if current_time is not None else time.time()

        # Step 1: Validate message security and topology
        valid, reason = edge_handover_protocol_engine.validate_incoming_message(msg, current_time=now)
        if not valid:
            logger.warning(f"Node {self.node_id} rejected peer message {msg.message_id}: {reason}")
            return False, reason, None

        # Step 2: Ensure node has active lease
        if not self.has_active_autonomy(now):
            reject_msg = edge_handover_protocol_engine.create_message(
                message_type=PeerMessageType.HANDOVER_REJECT,
                handover_id=msg.handover_id,
                chain_id=msg.chain_id,
                source_node=self.camera_id,
                destination_node=msg.source_node,
                entity_id=msg.entity_id,
                payload={"reason": "DESTINATION_NODE_LEASE_EXPIRED_OR_LOCKED"},
                timestamp=now
            )
            return False, "DESTINATION_NODE_AUTONOMY_EXPIRED", reject_msg

        # Step 3: Handle message types
        if msg.message_type == PeerMessageType.HANDOVER_PROPOSE:
            priority = msg.payload.get("priority_score", 70.0)
            eta_sec = msg.payload.get("expected_eta_sec", 15.0)

            # Request camera reservation via Arbitration Engine
            ok_res, r_res, res = ptz_arbitration_engine.request_camera_reservation(
                camera_id=self.camera_id,
                entity_id=msg.entity_id,
                priority_score=priority,
                duration_sec=eta_sec + 15.0,
                timestamp=now
            )

            if not ok_res:
                reject_msg = edge_handover_protocol_engine.create_message(
                    message_type=PeerMessageType.HANDOVER_REJECT,
                    handover_id=msg.handover_id,
                    chain_id=msg.chain_id,
                    source_node=self.camera_id,
                    destination_node=msg.source_node,
                    entity_id=msg.entity_id,
                    payload={"reason": f"ARBITRATION_REJECTED_{r_res}"},
                    timestamp=now
                )
                return False, f"REJECTED_{r_res}", reject_msg

            # Slew PTZ locally if capable
            ptz_capable = self.capabilities.get("ptz", False)
            if ptz_capable and not self.global_ptz_lock_active:
                from backend.app.services.predictive_ptz_engine import predictive_ptz_engine
                cued, c_reason, act = predictive_ptz_engine.evaluate_and_precue(
                    prediction_id=msg.prediction_id,
                    entity_id=msg.entity_id,
                    target_camera=self.camera_id,
                    confidence=msg.confidence,
                    eta_sec=eta_sec,
                    timestamp=now
                )
                ptz_ready = cued
            else:
                ptz_ready = True  # Fixed camera or lock active

            self.local_handovers[msg.handover_id] = {
                "handover_id": msg.handover_id,
                "chain_id": msg.chain_id,
                "source_camera": msg.source_node,
                "target_camera": self.camera_id,
                "entity_id": msg.entity_id,
                "state": HandoverState.AUTHORIZED.value,
                "created_at": now
            }

            accept_msg = edge_handover_protocol_engine.create_message(
                message_type=PeerMessageType.HANDOVER_ACCEPT,
                handover_id=msg.handover_id,
                chain_id=msg.chain_id,
                source_node=self.camera_id,
                destination_node=msg.source_node,
                entity_id=msg.entity_id,
                payload={"ptz_ready": ptz_ready},
                timestamp=now
            )

            self.buffer_event("edge.handover_accepted", {
                "handover_id": msg.handover_id,
                "node": self.node_id,
                "camera_id": self.camera_id,
                "ptz_ready": ptz_ready
            }, timestamp=now)

            return True, "PROPOSAL_ACCEPTED", accept_msg

        elif msg.message_type == PeerMessageType.HANDOVER_ACCEPT:
            if msg.handover_id in self.local_handovers:
                self.local_handovers[msg.handover_id]["state"] = HandoverState.PTZ_READY.value
            return True, "PEER_ACCEPTED_CONFIRMED", None

        elif msg.message_type == PeerMessageType.TARGET_ACQUIRED:
            if msg.handover_id in self.local_handovers:
                self.local_handovers[msg.handover_id]["state"] = HandoverState.TARGET_ACQUIRED.value
            return True, "TARGET_ACQUIRED_RECORDED", None

        elif msg.message_type == PeerMessageType.HANDOVER_CONFIRMED:
            if msg.handover_id in self.local_handovers:
                self.local_handovers[msg.handover_id]["state"] = HandoverState.HANDOVER_CONFIRMED.value
            return True, "HANDOVER_CONFIRMED_RECORDED", None

        elif msg.message_type == PeerMessageType.HANDOVER_CANCEL:
            if msg.handover_id in self.local_handovers:
                self.local_handovers[msg.handover_id]["state"] = HandoverState.CANCELLED.value
            ptz_arbitration_engine.release_camera(self.camera_id, reason="PEER_CANCELLED", timestamp=now)
            return True, "HANDOVER_CANCELLED", None

        return True, f"PROCESSED_{msg.message_type.value}", None

    def confirm_local_acquisition(
        self,
        handover_id: str,
        entity_id: str,
        local_track_id: str,
        association_score: float = 0.85,
        source_camera: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str, Optional[PeerHandoverMessage]]:
        """
        Called when local edge vision detector observes the expected entity.
        Locks PTZ locally and dispatches HANDOVER_CONFIRMED back to source peer.
        """
        now = timestamp if timestamp is not None else time.time()
        ptz_arbitration_engine.lock_camera(self.camera_id, entity_id, timestamp=now)

        h_info = self.local_handovers.get(handover_id, {})
        h_info["state"] = HandoverState.HANDOVER_CONFIRMED.value
        h_info["local_track_id"] = local_track_id
        h_info["association_score"] = association_score

        src = source_camera or h_info.get("source_camera")
        confirm_msg = None
        if src:
            confirm_msg = edge_handover_protocol_engine.create_message(
                message_type=PeerMessageType.HANDOVER_CONFIRMED,
                handover_id=handover_id,
                chain_id=h_info.get("chain_id", "CHAIN-UNKNOWN"),
                source_node=self.camera_id,
                destination_node=src,
                entity_id=entity_id,
                confidence=association_score,
                payload={"association_score": association_score, "track_id": local_track_id},
                timestamp=now
            )

        self.buffer_event("edge.handover_confirmed", {
            "handover_id": handover_id,
            "camera_id": self.camera_id,
            "entity_id": entity_id,
            "track_id": local_track_id,
            "association_score": association_score
        }, timestamp=now)

        return True, "ACQUISITION_CONFIRMED", confirm_msg

    def fallback_to_central_handover(
        self,
        entity_id: str,
        source_camera: str,
        target_camera: str,
        prediction_id: str,
        confidence: float,
        expected_eta_sec: float,
        timestamp: Optional[float] = None
    ) -> Tuple[bool, str, Any]:
        """
        Fallback mechanism: When peer communication fails or peer node is offline,
        route handover through the centralized control-plane PTZHandoverMesh.
        """
        logger.info(f"Fallback to central control plane for handover {source_camera} -> {target_camera}")
        ok, reason, hop = ptz_handover_mesh.initiate_handover(
            entity_id=entity_id,
            source_camera=source_camera,
            target_camera=target_camera,
            prediction_id=prediction_id,
            prediction_confidence=confidence,
            expected_eta_sec=expected_eta_sec,
            timestamp=timestamp
        )
        self.buffer_event("edge.central_fallback_invoked", {
            "source_camera": source_camera,
            "target_camera": target_camera,
            "entity_id": entity_id,
            "success": ok,
            "reason": reason
        }, timestamp=timestamp)
        return ok, f"CENTRAL_FALLBACK_{reason}", hop


class EdgeNodeManager:
    """
    Control plane service managing the registry of edge camera nodes,
    heartbeats, global emergency lock, and peer message dispatching.
    """
    def __init__(self):
        self.nodes: Dict[str, EdgeCameraNode] = {}
        self.global_ptz_lock: bool = False
        self._initialize_from_topology()

    def _initialize_from_topology(self):
        topo = camera_topology_service.get_topology_graph()
        for cam_id in topo.get("nodes", {}).keys():
            self.nodes[cam_id] = EdgeCameraNode(cam_id)
        logger.info(f"Initialized EdgeNodeManager with {len(self.nodes)} edge camera nodes.")

    def get_node(self, camera_id: str) -> Optional[EdgeCameraNode]:
        return self.nodes.get(camera_id)

    def set_global_ptz_lock(self, locked: bool):
        self.global_ptz_lock = locked
        for node in self.nodes.values():
            node.global_ptz_lock_active = locked
        logger.warning(f"EMERGENCY GLOBAL PTZ LOCK SET TO: {locked}")
        publish_event("edge.global_lock_changed", {"global_ptz_lock": locked})

    def dispatch_peer_message(self, msg: PeerHandoverMessage, current_time: Optional[float] = None) -> Tuple[bool, str, Optional[PeerHandoverMessage]]:
        """Routes peer message from source edge node to destination edge node."""
        dest_node = self.nodes.get(msg.destination_node)
        if not dest_node:
            return False, f"DESTINATION_NODE_{msg.destination_node}_NOT_FOUND", None
        return dest_node.handle_incoming_peer_message(msg, current_time=current_time)

    def set_network_partition_mode(self, mode: NetworkPartitionMode):
        for node in self.nodes.values():
            node.network_state = mode
        publish_event("edge.network_partition_changed", {"mode": mode.value})

    def reset(self):
        self.nodes.clear()
        self.global_ptz_lock = False
        self._initialize_from_topology()
        edge_handover_protocol_engine.reset()


edge_node_manager = EdgeNodeManager()
