"""
TRINETRA — Camera Topology & Transition Prediction Service (Phase VIII)
Maintains logical spatial adjacency, travel-time feasibility windows,
and candidate next-camera predictive forecasting.
"""

from __future__ import annotations
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("CameraTopology")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "camera_topology.yaml"


class CameraTopologyService:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.mode = "LOGICAL_TOPOLOGY"
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.transitions: List[Dict[str, Any]] = []
        self._adjacency: Dict[str, List[Dict[str, Any]]] = {}
        self.load_topology()

    def load_topology(self):
        if not self.config_path.exists():
            logger.warning(f"Topology config not found at {self.config_path}, using defaults.")
            self._load_defaults()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            meta = data.get("topology_metadata", {})
            self.mode = meta.get("mode", "LOGICAL_TOPOLOGY")
            self.nodes = data.get("nodes", {})
            self.transitions = data.get("transitions", [])

            self._adjacency = {}
            for t in self.transitions:
                f_cam = t["from_camera"]
                if f_cam not in self._adjacency:
                    self._adjacency[f_cam] = []
                self._adjacency[f_cam].append(t)

            logger.info(f"Loaded camera topology: {len(self.nodes)} nodes, {len(self.transitions)} transitions ({self.mode}).")
        except Exception as e:
            logger.error(f"Failed to parse topology config: {e}. Falling back to default.")
            self._load_defaults()

    def _load_defaults(self):
        self.mode = "LOGICAL_TOPOLOGY"
        self.nodes = {
            "CAM-001": {"name": "Perimeter North PTZ", "zone_id": "ZONE-PERIMETER-NORTH"},
            "CAM-002": {"name": "Logistics Gate", "zone_id": "ZONE-CARGO-BAY"},
            "CAM-003": {"name": "Perimeter South PTZ", "zone_id": "ZONE-PERIMETER-SOUTH", "sensor_type": "OPTICAL", "ptz": True},
            "CAM-004": {"name": "Airspace Tower", "zone_id": "ZONE-AIRSPACE", "sensor_type": "OPTICAL", "ptz": False},
            "CAM-005": {"name": "North Outpost Thermal", "zone_id": "ZONE-PERIMETER-NORTH-T", "sensor_type": "THERMAL", "ptz": False},
            "CAM-006": {"name": "East Corridor Thermal", "zone_id": "ZONE-CORRIDOR-EAST-T", "sensor_type": "THERMAL", "ptz": False},
            "CAM-007": {"name": "South Sector Dual PTZ", "zone_id": "ZONE-PERIMETER-SOUTH-T", "sensor_type": "THERMAL", "ptz": True},
            "CAM-008": {"name": "Airbase Boundary Mast", "zone_id": "ZONE-AIRBASE-BOUNDARY", "sensor_type": "OPTICAL", "ptz": False}
        }
        self.transitions = [
            {"from_camera": "CAM-001", "to_camera": "CAM-002", "min_travel_time_sec": 5.0, "max_travel_time_sec": 30.0, "direction": "SOUTHWARD"},
            {"from_camera": "CAM-002", "to_camera": "CAM-001", "min_travel_time_sec": 5.0, "max_travel_time_sec": 30.0, "direction": "NORTHWARD"},
            {"from_camera": "CAM-002", "to_camera": "CAM-003", "min_travel_time_sec": 10.0, "max_travel_time_sec": 45.0, "direction": "SOUTHWARD"},
            {"from_camera": "CAM-003", "to_camera": "CAM-002", "min_travel_time_sec": 10.0, "max_travel_time_sec": 45.0, "direction": "NORTHWARD"},
            {"from_camera": "CAM-003", "to_camera": "CAM-004", "min_travel_time_sec": 15.0, "max_travel_time_sec": 60.0, "direction": "EASTWARD"},
            {"from_camera": "CAM-004", "to_camera": "CAM-003", "min_travel_time_sec": 15.0, "max_travel_time_sec": 60.0, "direction": "WESTWARD"},
            {"from_camera": "CAM-001", "to_camera": "CAM-005", "min_travel_time_sec": 2.0, "max_travel_time_sec": 20.0, "direction": "NORTH-WEST"},
            {"from_camera": "CAM-005", "to_camera": "CAM-001", "min_travel_time_sec": 2.0, "max_travel_time_sec": 20.0, "direction": "SOUTH-EAST"},
            {"from_camera": "CAM-003", "to_camera": "CAM-007", "min_travel_time_sec": 3.0, "max_travel_time_sec": 25.0, "direction": "SOUTH-WEST"},
            {"from_camera": "CAM-007", "to_camera": "CAM-003", "min_travel_time_sec": 3.0, "max_travel_time_sec": 25.0, "direction": "NORTH-EAST"}
        ]
        self._adjacency = {}
        for t in self.transitions:
            f_cam = t["from_camera"]
            if f_cam not in self._adjacency:
                self._adjacency[f_cam] = []
            self._adjacency[f_cam].append(t)

    def is_transition_feasible(
        self,
        from_cam: str,
        to_cam: str,
        elapsed_sec: float
    ) -> Tuple[bool, str]:
        """
        Validates whether target movement from from_cam to to_cam is physically plausible.
        Returns: (is_feasible: bool, reason: str)
        """
        if from_cam == to_cam:
            # Same camera transition is continuous observation
            return True, "CONTINUOUS_SAME_CAMERA"

        adjacent = self._adjacency.get(from_cam, [])
        matching = [t for t in adjacent if t["to_camera"] == to_cam]

        if not matching:
            # Multi-hop or disconnected cameras
            return False, f"NO_DIRECT_TOPOLOGY_EDGE_{from_cam}_TO_{to_cam}"

        trans = matching[0]
        min_t = trans["min_travel_time_sec"]
        max_t = trans["max_travel_time_sec"]

        if elapsed_sec < min_t:
            return False, f"TRANSITION_TOO_FAST_{elapsed_sec:.1f}S_MIN_{min_t}S"
        elif elapsed_sec > max_t:
            return False, f"TRANSITION_TOO_SLOW_{elapsed_sec:.1f}S_MAX_{max_t}S"

        return True, f"FEASIBLE_WINDOW_{min_t}S_TO_{max_t}S"

    def predict_next_cameras(self, current_cam: str) -> List[Dict[str, Any]]:
        """Forecasts candidate next cameras based on spatial adjacency."""
        candidates = []
        for edge in self._adjacency.get(current_cam, []):
            candidates.append({
                "to_camera": edge["to_camera"],
                "direction": edge.get("direction", "UNKNOWN"),
                "min_eta_sec": edge["min_travel_time_sec"],
                "max_eta_sec": edge["max_travel_time_sec"],
                "typical_eta_sec": edge.get("typical_travel_time_sec", (edge["min_travel_time_sec"] + edge["max_travel_time_sec"]) / 2.0)
            })
        return candidates

    def get_topology_graph(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "nodes": self.nodes,
            "transitions": self.transitions
        }

    def get_sensor_type(self, camera_id: str) -> str:
        """Returns OPTICAL, THERMAL, or UNKNOWN."""
        node = self.nodes.get(camera_id, {})
        return node.get("sensor_type", "OPTICAL")

    def get_camera_capabilities(self, camera_id: str) -> Dict[str, Any]:
        """Returns node capabilities dictionary."""
        return self.nodes.get(camera_id, {
            "sensor_type": "OPTICAL",
            "ptz": False,
            "thermal": False,
            "optical": True,
            "night_capable": False,
            "supported_operations": ["DETECT", "TRACK"]
        })

    def is_cross_spectral_pair(self, cam_a: str, cam_b: str) -> bool:
        """Determines if transition between cam_a and cam_b constitutes an optical-thermal transition."""
        type_a = self.get_sensor_type(cam_a)
        type_b = self.get_sensor_type(cam_b)
        return (type_a == "OPTICAL" and type_b == "THERMAL") or (type_a == "THERMAL" and type_b == "OPTICAL")

    def get_authorized_peers(self, camera_id: str) -> List[str]:
        """Returns list of adjacent camera IDs authorized for peer-to-peer handover."""
        edges = self._adjacency.get(camera_id, [])
        return [e["to_camera"] for e in edges]


camera_topology_service = CameraTopologyService()
