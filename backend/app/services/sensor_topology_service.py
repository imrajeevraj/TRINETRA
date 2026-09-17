"""
TRINETRA Phase XIV — Sensor Topology Service
Extends Phase XI 8-Camera distributed mesh with detailed sensor topology nodes.
Tracks sensor-to-camera bindings, physical/logical positions, orientation,
clock sources, and calibration references.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from backend.app.services.sensor_abstraction import SensorModality, DataOrigin

logger = logging.getLogger("SensorTopology")


@dataclass
class SensorTopologyNode:
    camera_id: str
    sensor_id: str
    modality: SensorModality
    position_reference: str = "LOGICAL / UNKNOWN"
    orientation_reference: str = "FIXED_NORTH"
    intrinsics_reference: str = ""
    extrinsics_reference: str = ""
    resolution: str = "1920x1080"
    frame_rate: int = 30
    timestamp_source: str = "PTP_IEEE_1588"
    synchronization_status: str = "SYNCED"
    calibration_status: str = "VALIDATED"
    is_co_located: bool = False
    co_located_with: Optional[str] = None


class SensorTopologyService:
    """
    Manages multimodal topology for the 8-camera mesh.
    Guarantees no fabricated GPS or physical coordinates:
    position_reference strictly reports 'LOGICAL / UNKNOWN' when unmeasured.
    """

    def __init__(self):
        self._nodes: Dict[str, SensorTopologyNode] = {}
        self._initialize_topology()

    def _initialize_topology(self):
        # CAM-001 (Optical PTZ)
        self._add_node(
            camera_id="CAM-001",
            sensor_id="SNS-CAM001-RGB",
            modality=SensorModality.OPTICAL_RGB,
            orientation_reference="PTZ_SLEW_360",
            intrinsics_reference="INT-OPT-001",
            resolution="1920x1080",
            frame_rate=30,
        )

        # CAM-002 (Optical Bullet)
        self._add_node(
            camera_id="CAM-002",
            sensor_id="SNS-CAM002-RGB",
            modality=SensorModality.OPTICAL_RGB,
            orientation_reference="FIXED_SOUTH",
            intrinsics_reference="INT-OPT-002",
            resolution="1920x1080",
            frame_rate=30,
        )

        # CAM-003 (Optical PTZ)
        self._add_node(
            camera_id="CAM-003",
            sensor_id="SNS-CAM003-RGB",
            modality=SensorModality.OPTICAL_RGB,
            orientation_reference="PTZ_SLEW_360",
            intrinsics_reference="INT-OPT-003",
            resolution="1920x1080",
            frame_rate=30,
        )

        # CAM-004 (Airspace Mast 4K)
        self._add_node(
            camera_id="CAM-004",
            sensor_id="SNS-CAM004-RGB",
            modality=SensorModality.OPTICAL_RGB,
            orientation_reference="AIRSPACE_ELEVATED_45",
            intrinsics_reference="INT-OPT-004",
            resolution="3840x2160",
            frame_rate=60,
        )

        # CAM-005 (Thermal Fixed Sentry)
        self._add_node(
            camera_id="CAM-005",
            sensor_id="SNS-CAM005-LWIR",
            modality=SensorModality.THERMAL_LWIR,
            orientation_reference="FIXED_NORTH_OUTPOST",
            intrinsics_reference="INT-THM-005",
            resolution="640x512",
            frame_rate=30,
        )

        # CAM-006 (Thermal Fixed Sentry)
        self._add_node(
            camera_id="CAM-006",
            sensor_id="SNS-CAM006-LWIR",
            modality=SensorModality.THERMAL_LWIR,
            orientation_reference="FIXED_EAST_CORRIDOR",
            intrinsics_reference="INT-THM-006",
            resolution="640x512",
            frame_rate=30,
        )

        # CAM-007 (Dual Head Optical + Thermal PTZ)
        self._add_node(
            camera_id="CAM-007",
            sensor_id="SNS-CAM007-RGB",
            modality=SensorModality.OPTICAL_RGB,
            orientation_reference="PTZ_SLEW_360_DUAL",
            intrinsics_reference="INT-OPT-007",
            extrinsics_reference="EXT-DUAL-007",
            resolution="1920x1080",
            frame_rate=30,
            is_co_located=True,
            co_located_with="SNS-CAM007-LWIR",
        )
        self._add_node(
            camera_id="CAM-007",
            sensor_id="SNS-CAM007-LWIR",
            modality=SensorModality.THERMAL_LWIR,
            orientation_reference="PTZ_SLEW_360_DUAL",
            intrinsics_reference="INT-THM-007",
            extrinsics_reference="EXT-DUAL-007",
            resolution="640x512",
            frame_rate=30,
            is_co_located=True,
            co_located_with="SNS-CAM007-RGB",
        )

        # CAM-008 (Optical Boundary Bullet)
        self._add_node(
            camera_id="CAM-008",
            sensor_id="SNS-CAM008-RGB",
            modality=SensorModality.OPTICAL_RGB,
            orientation_reference="FIXED_AIRBASE_BOUNDARY",
            intrinsics_reference="INT-OPT-008",
            resolution="1920x1080",
            frame_rate=30,
        )

    def _add_node(self, camera_id: str, sensor_id: str, modality: SensorModality, **kwargs):
        node = SensorTopologyNode(
            camera_id=camera_id,
            sensor_id=sensor_id,
            modality=modality,
            position_reference="LOGICAL / UNKNOWN",
            **kwargs,
        )
        self._nodes[sensor_id] = node

    def get_node(self, sensor_id: str) -> Optional[SensorTopologyNode]:
        return self._nodes.get(sensor_id)

    def list_nodes(self, camera_id: Optional[str] = None) -> List[SensorTopologyNode]:
        nodes = list(self._nodes.values())
        if camera_id:
            nodes = [n for n in nodes if n.camera_id == camera_id]
        return nodes

    def get_co_located_sensors(self, camera_id: str) -> List[SensorTopologyNode]:
        return [n for n in self._nodes.values() if n.camera_id == camera_id and n.is_co_located]

    def reset(self):
        self._nodes.clear()
        self._initialize_topology()


sensor_topology_service = SensorTopologyService()
