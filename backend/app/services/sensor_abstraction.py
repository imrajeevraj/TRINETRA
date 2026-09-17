"""
TRINETRA Phase XIV — Multimodal Sensor Abstraction Service
Provides generic sensor abstraction across 5 modalities:
OPTICAL_RGB, THERMAL_LWIR, DEPTH, RADAR, OTHER.
Enforces strict provenance tagging (REAL_SENSOR, SIMULATED, SYNTHETIC)
and cryptographic SHA-256 frame integrity.
"""

from __future__ import annotations
import hashlib
import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("SensorAbstraction")


class SensorModality(str, Enum):
    OPTICAL_RGB = "OPTICAL_RGB"
    THERMAL_LWIR = "THERMAL_LWIR"
    DEPTH = "DEPTH"
    RADAR = "RADAR"
    OTHER = "OTHER"


class DataOrigin(str, Enum):
    REAL_SENSOR = "REAL_SENSOR"
    RECORDED_REAL_SENSOR = "RECORDED_REAL_SENSOR"
    SIMULATED = "SIMULATED"
    SYNTHETIC = "SYNTHETIC"


@dataclass
class SensorObservation:
    sensor_id: str
    camera_id: str
    modality: SensorModality
    timestamp: float
    ingestion_timestamp: float
    frame_id: str
    sequence_number: int
    resolution: Tuple[int, int]
    frame_hash: str
    data_origin: DataOrigin
    calibration_id: Optional[str] = None
    capture_metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload_bytes: Optional[bytes] = None
    monotonic_timestamp: float = 0.0
    pixel_format: str = "MONO8"
    sensor_mode: str = "DEFAULT"
    integration_time_us: Optional[float] = None
    sensor_gain_db: Optional[float] = None
    temperature_k: Optional[float] = None
    quality_status: str = "UNVALIDATED"
    rejection_reason: Optional[str] = None
    source_type: str = "EMBEDDED"


@dataclass
class SensorDevice:
    sensor_id: str
    camera_id: str
    modality: SensorModality
    resolution: Tuple[int, int]
    fps: int
    data_origin: DataOrigin
    is_active: bool = True
    position_status: str = "LOGICAL / UNKNOWN"
    calibration_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class SensorAbstractionService:
    """
    Manages physical and logical multimodal sensors, enforces data origin tagging,
    and calculates cryptographic frame digests.
    """

    def __init__(self):
        self._sensors: Dict[str, SensorDevice] = {}
        self._observation_history: Dict[str, List[SensorObservation]] = {}
        self._sequence_counters: Dict[str, int] = {}
        self._initialize_mesh_sensors()

    def _initialize_mesh_sensors(self):
        """Initializes default 8-camera mesh sensor abstraction (CAM-001..CAM-008)."""
        # Optical sensors across CAM-001 to CAM-008
        for i in range(1, 9):
            cam_id = f"CAM-{i:03d}"
            sns_id = f"SNS-CAM{i:03d}-RGB"
            res = (3840, 2160) if i == 4 else (1920, 1080)
            fps = 60 if i == 4 else 30
            self.register_sensor(
                sensor_id=sns_id,
                camera_id=cam_id,
                modality=SensorModality.OPTICAL_RGB,
                resolution=res,
                fps=fps,
                data_origin=DataOrigin.SIMULATED,
                calibration_id=f"CAL-OPT-{i:03d}",
            )

        # Thermal LWIR sensors on CAM-005, CAM-006, CAM-007
        thermal_cams = ["CAM-005", "CAM-006", "CAM-007"]
        for cam_id in thermal_cams:
            idx = cam_id.split("-")[1]
            sns_id = f"SNS-{cam_id.replace('-', '')}-LWIR"
            self.register_sensor(
                sensor_id=sns_id,
                camera_id=cam_id,
                modality=SensorModality.THERMAL_LWIR,
                resolution=(640, 512),
                fps=30,
                data_origin=DataOrigin.SIMULATED,
                calibration_id=f"CAL-THM-{idx}",
            )

    def register_sensor(
        self,
        sensor_id: str,
        camera_id: str,
        modality: SensorModality,
        resolution: Tuple[int, int],
        fps: int,
        data_origin: DataOrigin,
        calibration_id: Optional[str] = None,
        position_status: str = "LOGICAL / UNKNOWN",
    ) -> SensorDevice:
        """Registers a sensor into the multimodal abstraction inventory."""
        dev = SensorDevice(
            sensor_id=sensor_id,
            camera_id=camera_id,
            modality=modality,
            resolution=resolution,
            fps=fps,
            data_origin=data_origin,
            calibration_id=calibration_id,
            position_status=position_status,
        )
        self._sensors[sensor_id] = dev
        if sensor_id not in self._observation_history:
            self._observation_history[sensor_id] = []
        if sensor_id not in self._sequence_counters:
            self._sequence_counters[sensor_id] = 0
        logger.info(f"Registered sensor {sensor_id} ({modality.value}) on {camera_id} [Origin: {data_origin.value}]")
        return dev

    def get_sensor(self, sensor_id: str) -> Optional[SensorDevice]:
        return self._sensors.get(sensor_id)

    def list_sensors(self, camera_id: Optional[str] = None, modality: Optional[SensorModality] = None) -> List[SensorDevice]:
        sensors = list(self._sensors.values())
        if camera_id:
            sensors = [s for s in sensors if s.camera_id == camera_id]
        if modality:
            sensors = [s for s in sensors if s.modality == modality]
        return sensors

    def create_observation(
        self,
        sensor_id: str,
        frame_id: str,
        payload_bytes: bytes,
        timestamp: Optional[float] = None,
        capture_metadata: Optional[Dict[str, Any]] = None,
        forced_data_origin: Optional[DataOrigin] = None,
        monotonic_timestamp: Optional[float] = None,
        pixel_format: str = "MONO8",
        sensor_mode: str = "DEFAULT",
        integration_time_us: Optional[float] = None,
        sensor_gain_db: Optional[float] = None,
        temperature_k: Optional[float] = None,
        quality_status: str = "UNVALIDATED",
        rejection_reason: Optional[str] = None,
        source_type: str = "EMBEDDED",
    ) -> SensorObservation:
        """
        Creates a verified sensor observation with SHA-256 hash and strict provenance tagging.
        """
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            raise ValueError(f"Sensor {sensor_id} is not registered.")

        ingest_time = time.time()
        capture_time = timestamp if timestamp is not None else ingest_time
        mono_time = monotonic_timestamp if monotonic_timestamp is not None else time.monotonic()
        frame_hash = hashlib.sha256(payload_bytes).hexdigest().upper()
        origin = forced_data_origin or sensor.data_origin

        self._sequence_counters[sensor_id] = self._sequence_counters.get(sensor_id, 0) + 1
        seq_num = self._sequence_counters[sensor_id]

        obs = SensorObservation(
            sensor_id=sensor_id,
            camera_id=sensor.camera_id,
            modality=sensor.modality,
            timestamp=capture_time,
            ingestion_timestamp=ingest_time,
            frame_id=frame_id,
            sequence_number=seq_num,
            resolution=sensor.resolution,
            frame_hash=frame_hash,
            data_origin=origin,
            calibration_id=sensor.calibration_id,
            capture_metadata=capture_metadata or {},
            raw_payload_bytes=payload_bytes,
            monotonic_timestamp=mono_time,
            pixel_format=pixel_format,
            sensor_mode=sensor_mode,
            integration_time_us=integration_time_us,
            sensor_gain_db=sensor_gain_db,
            temperature_k=temperature_k,
            quality_status=quality_status,
            rejection_reason=rejection_reason,
            source_type=source_type,
        )

        history = self._observation_history[sensor_id]
        history.append(obs)
        if len(history) > 200:
            history.pop(0)

        return obs

    def get_latest_observation(self, sensor_id: str) -> Optional[SensorObservation]:
        history = self._observation_history.get(sensor_id, [])
        return history[-1] if history else None

    def reset(self):
        """Resets all state to clean initial defaults."""
        self._sensors.clear()
        self._observation_history.clear()
        self._sequence_counters.clear()
        self._initialize_mesh_sensors()


sensor_abstraction_service = SensorAbstractionService()
