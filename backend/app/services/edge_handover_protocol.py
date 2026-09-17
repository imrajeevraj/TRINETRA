"""
TRINETRA — Edge Handover Protocol & Peer Message Engine (Phase XI)
Implements peer-to-peer message interchange, topological authorization,
idempotent deduplication, sequence replay protection, and rate limiting.
"""

from __future__ import annotations
import time
import hmac
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set

from backend.app.services.camera_topology import camera_topology_service

logger = logging.getLogger("EdgeHandoverProtocol")

DEFAULT_SECRET_KEY = b"ibvap_edge_mesh_psk_secret_2026"
PROTOCOL_VERSION = "1.0.0"


class PeerMessageType(str, Enum):
    HANDOVER_PROPOSE = "HANDOVER_PROPOSE"
    HANDOVER_ACCEPT = "HANDOVER_ACCEPT"
    HANDOVER_REJECT = "HANDOVER_REJECT"
    HANDOVER_READY = "HANDOVER_READY"
    TARGET_ACQUIRED = "TARGET_ACQUIRED"
    HANDOVER_CONFIRMED = "HANDOVER_CONFIRMED"
    HANDOVER_CANCEL = "HANDOVER_CANCEL"
    HANDOVER_EXPIRED = "HANDOVER_EXPIRED"


@dataclass
class PeerHandoverMessage:
    message_id: str
    message_type: PeerMessageType
    handover_id: str
    chain_id: str
    source_node: str
    destination_node: str
    entity_id: str
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 1
    prediction_id: str = ""
    confidence: float = 1.0
    ttl_sec: float = 20.0
    protocol_version: str = PROTOCOL_VERSION
    signature: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def compute_signature(self, secret: bytes = DEFAULT_SECRET_KEY) -> str:
        data = f"{self.message_id}:{self.message_type.value}:{self.source_node}:{self.destination_node}:{self.entity_id}:{self.sequence_number}:{self.timestamp:.3f}"
        return hmac.new(secret, data.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_signature(self, secret: bytes = DEFAULT_SECRET_KEY) -> bool:
        if not self.signature:
            return False
        expected = self.compute_signature(secret)
        return hmac.compare_digest(self.signature, expected)


class PeerRateLimiter:
    """Token bucket rate limiter per peer channel."""
    def __init__(self, rate_per_sec: float = 100.0, capacity: float = 200.0):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    def allow(self, peer_id: str, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        tokens = self.tokens.get(peer_id, self.capacity)
        last = self.last_update.get(peer_id, now)

        delta = max(0.0, now - last)
        tokens = min(self.capacity, tokens + delta * self.rate)
        self.last_update[peer_id] = now

        if tokens >= 1.0:
            self.tokens[peer_id] = tokens - 1.0
            return True
        self.tokens[peer_id] = tokens
        return False

    def reset(self):
        self.tokens.clear()
        self.last_update.clear()


class EdgeHandoverProtocolEngine:
    """
    Validates, signs, deduplicates, and rate-limits peer handover messages.
    Guarantees that messages:
    1. Are authenticated via cryptographic HMAC signature.
    2. Adhere to topological adjacency (no arbitrary hops).
    3. Are processed idempotently without replaying duplicate message_ids.
    4. Maintain monotonic sequence numbers.
    5. Fall within the allowed Time-To-Live (TTL).
    """
    def __init__(self, secret: bytes = DEFAULT_SECRET_KEY):
        self.secret = secret
        self.seen_messages: Set[str] = set()
        self.sent_sequences: Dict[Tuple[str, str], int] = {}       # (source, dest) -> last_sent_seq
        self.received_sequences: Dict[Tuple[str, str], int] = {}   # (source, dest) -> last_received_seq
        self.rate_limiter = PeerRateLimiter()
        self._msg_counter = 0

    def create_message(
        self,
        message_type: PeerMessageType,
        handover_id: str,
        chain_id: str,
        source_node: str,
        destination_node: str,
        entity_id: str,
        prediction_id: str = "",
        confidence: float = 1.0,
        ttl_sec: float = 20.0,
        payload: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> PeerHandoverMessage:
        self._msg_counter += 1
        now = timestamp if timestamp is not None else time.time()
        msg_id = f"MSG-{source_node}-{int(now * 1000)}-{self._msg_counter:04d}"

        channel = (source_node, destination_node)
        seq = self.sent_sequences.get(channel, 0) + 1
        self.sent_sequences[channel] = seq

        msg = PeerHandoverMessage(
            message_id=msg_id,
            message_type=message_type,
            handover_id=handover_id,
            chain_id=chain_id,
            source_node=source_node,
            destination_node=destination_node,
            entity_id=entity_id,
            timestamp=now,
            sequence_number=seq,
            prediction_id=prediction_id,
            confidence=confidence,
            ttl_sec=ttl_sec,
            payload=payload or {}
        )
        msg.signature = msg.compute_signature(self.secret)
        return msg

    def validate_incoming_message(
        self,
        msg: PeerHandoverMessage,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str]:
        now = current_time if current_time is not None else time.time()

        # 1. Check Protocol Version
        if msg.protocol_version != PROTOCOL_VERSION:
            return False, f"PROTOCOL_VERSION_MISMATCH_{msg.protocol_version}"

        # 2. Check Idempotency (Already Seen)
        if msg.message_id in self.seen_messages:
            return False, "DUPLICATE_MESSAGE_IGNORED"

        # 3. Check TTL Expiration
        if (now - msg.timestamp) > msg.ttl_sec:
            return False, f"MESSAGE_TTL_EXPIRED_{now - msg.timestamp:.1f}S_MAX_{msg.ttl_sec}S"

        # 4. Check Signature Authentication
        if not msg.verify_signature(self.secret):
            return False, "AUTHENTICATION_FAILED_INVALID_SIGNATURE"

        # 5. Check Topological Peer Authorization
        authorized_peers = camera_topology_service.get_authorized_peers(msg.source_node)
        if msg.destination_node not in authorized_peers:
            return False, f"UNAUTHORIZED_PEER_LINK_{msg.source_node}_TO_{msg.destination_node}"

        # 6. Check Rate Limiter
        if not self.rate_limiter.allow(msg.source_node, current_time=now):
            return False, f"PEER_RATE_LIMIT_EXCEEDED_{msg.source_node}"

        # 7. Check Sequence Monotonicity (Replay Protection)
        channel = (msg.source_node, msg.destination_node)
        last_rx_seq = self.received_sequences.get(channel, 0)
        if msg.sequence_number <= last_rx_seq and msg.message_type == PeerMessageType.HANDOVER_PROPOSE:
            return False, f"REPLAY_DETECTED_SEQ_{msg.sequence_number}_LE_{last_rx_seq}"

        # Mark seen and update sequence
        self.seen_messages.add(msg.message_id)
        self.received_sequences[channel] = max(last_rx_seq, msg.sequence_number)
        return True, "VALID"

    def reset(self):
        self.seen_messages.clear()
        self.sent_sequences.clear()
        self.received_sequences.clear()
        self.rate_limiter.reset()
        self._msg_counter = 0


edge_handover_protocol_engine = EdgeHandoverProtocolEngine()
