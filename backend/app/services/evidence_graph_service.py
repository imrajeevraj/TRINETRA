"""
TRINETRA — Unified Evidence & Entity Graph Service (Phase VIII)
Maintains queryable graph topology of incidents, entities, cameras, tracks,
and cryptographically verified evidence artifacts.
"""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.app.services.incident_correlation_engine import incident_correlation_engine
from backend.app.services.entity_association_engine import entity_association_engine

logger = logging.getLogger("EvidenceGraphService")


class EvidenceGraphService:
    def __init__(self):
        pass

    def verify_evidence_hash(self, file_path: str, expected_hash: str) -> bool:
        p = Path(file_path)
        if not p.exists():
            return False
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
            return h == expected_hash.upper()
        except Exception as e:
            logger.error(f"Error reading evidence {file_path}: {e}")
            return False

    def build_incident_graph(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Builds a queryable nodes-and-links graph representation for an incident.
        """
        inc = incident_correlation_engine.get_incident(incident_id)
        if not inc:
            return None

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids = set()

        def add_node(nid: str, label: str, node_type: str, metadata: Optional[Dict[str, Any]] = None):
            if nid not in node_ids:
                node_ids.add(nid)
                nodes.append({
                    "id": nid,
                    "label": label,
                    "type": node_type,
                    "metadata": metadata or {}
                })

        def add_edge(source: str, target: str, relationship: str, metadata: Optional[Dict[str, Any]] = None):
            edges.append({
                "source": source,
                "target": target,
                "relationship": relationship,
                "metadata": metadata or {}
            })

        # 1. Incident Root Node
        add_node(
            nid=inc.incident_id,
            label=f"Incident {inc.incident_id}",
            node_type="INCIDENT",
            metadata={
                "severity": inc.severity.value,
                "status": inc.status.value,
                "risk_score": inc.risk_score
            }
        )

        # 2. Camera Nodes
        for cam in inc.cameras_involved:
            add_node(nid=f"CAM_{cam}", label=cam, node_type="CAMERA")
            add_edge(source=inc.incident_id, target=f"CAM_{cam}", relationship="OBSERVED_BY")

        # 3. Global Entity Nodes
        for gid in inc.global_entities:
            add_node(nid=gid, label=gid, node_type="GLOBAL_ENTITY")
            add_edge(source=inc.incident_id, target=gid, relationship="ASSOCIATED_WITH")

            # Link observations
            entity_timeline = entity_association_engine.get_entity_timeline(gid)
            if entity_timeline:
                for obs in entity_timeline:
                    tid = f"TRK_{obs['track_id']}"
                    add_node(nid=tid, label=obs['track_id'], node_type="TRACK", metadata={"confidence": obs["confidence"]})
                    add_edge(source=gid, target=tid, relationship="TRACKED_AS")
                    add_edge(source=tid, target=f"CAM_{obs['camera_id']}", relationship="CAPTURED_BY")

        # 4. Evidence Frame Nodes
        for path_str, h in inc.evidence_hashes.items():
            ev_id = f"EV_{h[:10]}"
            add_node(
                nid=ev_id,
                label=f"Evidence {h[:8]}...",
                node_type="EVIDENCE",
                metadata={"file_path": path_str, "sha256": h}
            )
            add_edge(source=inc.incident_id, target=ev_id, relationship="SUPPORTED_BY")

        # 5. Timeline Sequence Edges (MOVED_TO)
        prev_node: Optional[str] = None
        for tl in inc.timeline:
            step_id = f"STEP_{tl.timestamp:.1f}_{tl.camera_id}"
            add_node(
                nid=step_id,
                label=f"{tl.event_type} on {tl.camera_id}",
                node_type="EVENT",
                metadata={"desc": tl.description, "risk_delta": tl.risk_delta}
            )
            add_edge(source=inc.incident_id, target=step_id, relationship="TRIGGERED")
            if prev_node:
                add_edge(source=prev_node, target=step_id, relationship="MOVED_TO")
            prev_node = step_id

        return {
            "incident_id": inc.incident_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges
        }


evidence_graph_service = EvidenceGraphService()
