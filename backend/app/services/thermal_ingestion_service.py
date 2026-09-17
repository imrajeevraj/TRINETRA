"""
TRINETRA Phase XIV — Native Thermal Data Ingestion Service
Handles ingestion of raw LWIR frame buffers and radiometric temperature arrays.
Distinguishes standard AGC thermal imagery from true radiometric temperature metadata.
"""

from __future__ import annotations
import hashlib
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.sensor_abstraction import DataOrigin

logger = logging.getLogger("ThermalIngestion")


class ThermalPayloadType(str, Enum):
    THERMAL_IMAGE = "THERMAL_IMAGE"                     # 8-bit normalized grayscale / pseudo-color
    RADIOMETRIC_THERMAL_DATA = "RADIOMETRIC_THERMAL_DATA" # 14/16-bit raw sensor / Kelvin temperature matrix


@dataclass
class RadiometricMetadata:
    is_radiometric: bool = False
    emissivity: float = 0.95
    ambient_temp_c: float = 22.0
    min_scene_temp_c: Optional[float] = None
    max_scene_temp_c: Optional[float] = None
    mean_scene_temp_c: Optional[float] = None
    thermal_sensitivity_netd_mk: float = 40.0
    temperature_unit: str = "CELSIUS"


@dataclass
class IngestedThermalFrame:
    frame_id: str
    sensor_id: str
    camera_id: str
    payload_type: ThermalPayloadType
    resolution: Tuple[int, int]
    timestamp: float
    sequence_number: int
    frame_hash: str
    data_origin: DataOrigin
    radiometric_meta: RadiometricMetadata
    calibration_id: Optional[str] = None
    raw_payload_bytes: bytes = field(default_factory=bytes)
    is_saturated: bool = False
    dead_pixel_count: int = 0


class ThermalIngestionService:
    """
    Ingests and validates raw thermal streams from physical or simulated LWIR sensors.
    """

    def __init__(self):
        self._frames: Dict[str, IngestedThermalFrame] = {}
        self._seq_counter = 0

    def ingest_frame(
        self,
        sensor_id: str,
        camera_id: str,
        payload_bytes: bytes,
        payload_type: ThermalPayloadType = ThermalPayloadType.THERMAL_IMAGE,
        resolution: Tuple[int, int] = (640, 512),
        timestamp: Optional[float] = None,
        data_origin: DataOrigin = DataOrigin.SIMULATED,
        radiometric_meta: Optional[RadiometricMetadata] = None,
        calibration_id: Optional[str] = None,
    ) -> IngestedThermalFrame:
        """
        Ingests a thermal frame, computes SHA-256 hash, and structures radiometric metadata.
        """
        self._seq_counter += 1
        capture_time = timestamp if timestamp is not None else time.time()
        frame_hash = hashlib.sha256(payload_bytes).hexdigest().upper()
        frame_id = f"FR-THM-{sensor_id}-{self._seq_counter:06d}"

        rad_meta = radiometric_meta or RadiometricMetadata()

        # Simple hardware saturation inspection
        is_sat = False
        if payload_bytes and len(payload_bytes) > 100:
            # Check for high percentage of 255s (pure white saturation)
            sat_count = sum(1 for b in payload_bytes[:1000] if b >= 254)
            if sat_count > 250:  # >25% in sample
                is_sat = True

        frame = IngestedThermalFrame(
            frame_id=frame_id,
            sensor_id=sensor_id,
            camera_id=camera_id,
            payload_type=payload_type,
            resolution=resolution,
            timestamp=capture_time,
            sequence_number=self._seq_counter,
            frame_hash=frame_hash,
            data_origin=data_origin,
            radiometric_meta=rad_meta,
            calibration_id=calibration_id,
            raw_payload_bytes=payload_bytes,
            is_saturated=is_sat,
        )

        self._frames[frame_id] = frame
        return frame

    def get_frame(self, frame_id: str) -> Optional[IngestedThermalFrame]:
        return self._frames.get(frame_id)

    def reset(self):
        self._frames.clear()
        self._seq_counter = 0


thermal_ingestion_service = ThermalIngestionService()
