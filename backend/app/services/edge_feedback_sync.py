"""
TRINETRA — Edge Feedback & Hard-Case Synchronization Service (Phase XIII)
Manages local edge node outboxes, chronological deduplicated sync,
and partition resilience across the 8-camera distributed mesh.
"""

from __future__ import annotations
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from pydantic import BaseModel, Field

from backend.app.services.edge_camera_node import (
    edge_node_manager,
    NetworkPartitionMode,
)

logger = logging.getLogger("EdgeFeedbackSync")


class OutboxMessage(BaseModel):
    outbox_id: str
    node_id: str
    camera_id: str
    message_type: str  # "OPERATOR_FEEDBACK", "HARD_CASE"
    sequence_number: int
    payload_json: str
    sha256: str
    timestamp: float = Field(default_factory=time.time)
    status: str = "PENDING"  # "PENDING", "SYNCED", "RETRY"


class EdgeFeedbackOutbox:
    """Represents the local outbox maintained on an individual edge camera node."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.node_id = f"NODE-{camera_id}"
        self.messages: List[OutboxMessage] = []
        self._seq = 0

    def append_message(self, message_type: str, payload: Dict[str, Any]) -> OutboxMessage:
        self._seq += 1
        payload_str = json.dumps(payload, sort_keys=True)
        sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest().upper()
        now = time.time()
        msg_id = f"OBX-{self.camera_id}-{int(now * 1000)}-{self._seq:05d}"

        msg = OutboxMessage(
            outbox_id=msg_id,
            node_id=self.node_id,
            camera_id=self.camera_id,
            message_type=message_type,
            sequence_number=self._seq,
            payload_json=payload_str,
            sha256=sha,
            timestamp=now,
            status="PENDING",
        )
        self.messages.append(msg)
        return msg

    def drain_pending(self) -> List[OutboxMessage]:
        pending = [m for m in self.messages if m.status == "PENDING"]
        return sorted(pending, key=lambda x: (x.timestamp, x.sequence_number))


class EdgeFeedbackSyncService:
    """
    Central synchronizer that ingests outboxes from all 8 camera nodes,
    handles network partitions, and verifies deduplication and integrity.
    """

    def __init__(self):
        self.node_outboxes: Dict[str, EdgeFeedbackOutbox] = {}
        self.ingested_message_hashes: Set[str] = set()
        self._init_node_outboxes()

    def _init_node_outboxes(self):
        for i in range(1, 9):
            cam_id = f"CAM-{i:03d}"
            self.node_outboxes[cam_id] = EdgeFeedbackOutbox(cam_id)

    def get_outbox(self, camera_id: str) -> Optional[EdgeFeedbackOutbox]:
        return self.node_outboxes.get(camera_id)

    def queue_feedback_at_edge(self, camera_id: str, feedback_data: Dict[str, Any]) -> Optional[OutboxMessage]:
        outbox = self.node_outboxes.get(camera_id)
        if not outbox:
            return None
        return outbox.append_message("OPERATOR_FEEDBACK", feedback_data)

    def queue_hard_case_at_edge(self, camera_id: str, hard_case_data: Dict[str, Any]) -> Optional[OutboxMessage]:
        outbox = self.node_outboxes.get(camera_id)
        if not outbox:
            return None
        return outbox.append_message("HARD_CASE", hard_case_data)

    def enqueue_edge_feedback(self, camera_id: str, feedback_data: Dict[str, Any]) -> Optional[OutboxMessage]:
        outbox = self.node_outboxes.get(camera_id)
        if not outbox:
            return None
        return outbox.append_message("OPERATOR_FEEDBACK", feedback_data)

    def enqueue_edge_hard_case(self, camera_id: str, hard_case_data: Dict[str, Any]) -> Optional[OutboxMessage]:
        outbox = self.node_outboxes.get(camera_id)
        if not outbox:
            return None
        return outbox.append_message("HARD_CASE", hard_case_data)

    def synchronize_node_outbox(
        self,
        camera_id: str,
        force_sync: bool = False,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Synchronizes an edge node's outbox to the central platform.
        Fails gracefully if node is partitioned or offline.
        """
        outbox = self.node_outboxes.get(camera_id)
        if not outbox:
            return False, f"NODE_{camera_id}_NOT_FOUND", {}

        # Check edge node connectivity from Phase XI manager
        edge_node = edge_node_manager.get_node(camera_id)
        if edge_node and not force_sync:
            net_state = getattr(edge_node, "network_state", None)
            part_mode = getattr(edge_node, "partition_mode", None)
            is_partitioned = (
                net_state in [NetworkPartitionMode.PARTITIONED, "PARTITIONED"]
                or str(part_mode) in ["PARTITIONED", "ISOLATED_FULL_PARTITION"]
            )
            if is_partitioned:
                logger.warning(f"Node {camera_id} is partitioned. Sync deferred; data retained in local outbox.")
                return False, f"NODE_{camera_id}_PARTITIONED_DATA_RETAINED_LOCALLY", {
                    "pending_messages": len(outbox.drain_pending()),
                    "remaining_pending": len(outbox.drain_pending()),
                    "status": "RETAINED_LOCAL_BUFFER",
                }

        pending = outbox.drain_pending()
        if not pending:
            return True, "NO_PENDING_MESSAGES", {"synced_count": 0, "duplicate_count": 0, "integrity_failures": 0}

        synced = 0
        duplicates = 0
        integrity_failures = 0

        for msg in pending:
            # 1. Verify SHA-256 integrity
            computed_sha = hashlib.sha256(msg.payload_json.encode("utf-8")).hexdigest().upper()
            if computed_sha != msg.sha256:
                logger.critical(f"Sync integrity violation on {msg.outbox_id}: {computed_sha} != {msg.sha256}")
                msg.status = "RETRY"
                integrity_failures += 1
                continue

            # 2. Deduplication check
            unique_key = f"{msg.node_id}:{msg.sha256}"
            if unique_key in self.ingested_message_hashes:
                msg.status = "SYNCED"
                duplicates += 1
                continue

            self.ingested_message_hashes.add(unique_key)
            msg.status = "SYNCED"
            synced += 1

        summary = {
            "camera_id": camera_id,
            "synced_count": synced,
            "duplicate_count": duplicates,
            "integrity_failures": integrity_failures,
            "remaining_pending": len(outbox.drain_pending()),
        }

        logger.info(f"Synchronized node {camera_id}: {synced} synced, {duplicates} duplicates, {integrity_failures} failures")
        return True, "SYNC_COMPLETED", summary

    def synchronize_all_nodes(self) -> Dict[str, Any]:
        """Synchronizes all connected edge mesh nodes."""
        results = {}
        for cam_id in self.node_outboxes.keys():
            _, _, sum_res = self.synchronize_node_outbox(cam_id)
            results[cam_id] = sum_res
        return results

    def reset(self):
        self.node_outboxes.clear()
        self.ingested_message_hashes.clear()
        self._init_node_outboxes()


edge_feedback_sync = EdgeFeedbackSyncService()
