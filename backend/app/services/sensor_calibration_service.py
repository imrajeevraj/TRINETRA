"""
TRINETRA Phase XIV — Sensor Calibration Framework Service
Maintains optical intrinsics, thermal sensor profiles, and cross-spectral extrinsics.
Enforces fail-safe degradation: without validated calibration, geometric fusion
is strictly disabled or degraded to non-geometric/temporal-only modes.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from backend.app.services.sensor_abstraction import SensorModality

logger = logging.getLogger("SensorCalibration")


class CalibrationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    UNVALIDATED = "UNVALIDATED"
    MISSING = "MISSING"
    EXPIRED = "EXPIRED"


@dataclass
class OpticalIntrinsics:
    fx: float = 1420.5
    fy: float = 1420.5
    cx: float = 960.0
    cy: float = 540.0
    k1: float = -0.05
    k2: float = 0.01
    p1: float = 0.001
    p2: float = 0.001
    resolution: tuple[int, int] = (1920, 1080)


@dataclass
class ThermalIntrinsics:
    fx: float = 612.3
    fy: float = 612.3
    cx: float = 320.0
    cy: float = 256.0
    sensor_profile: str = "Uncooled VOx Microbolometer 8-14um"
    netd_mk: float = 38.0
    pixel_pitch_um: float = 12.0
    resolution: tuple[int, int] = (640, 512)


@dataclass
class CrossSpectralExtrinsics:
    relative_rotation_euler: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0]) # roll, pitch, yaw deg
    relative_translation_m: list[float] = field(default_factory=lambda: [0.12, 0.0, 0.0])  # baseline offset meters
    rmse_alignment_px: float = 0.85
    alignment_version: str = "EXT-CAL-v1.0"


@dataclass
class CalibrationProfile:
    calibration_id: str
    sensor_id: str
    modality: SensorModality
    status: CalibrationStatus
    optical_intrinsics: Optional[OpticalIntrinsics] = None
    thermal_intrinsics: Optional[ThermalIntrinsics] = None
    cross_spectral_extrinsics: Optional[CrossSpectralExtrinsics] = None
    paired_sensor_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    notes: str = ""


class SensorCalibrationService:
    """
    Manages calibration profiles and enforces strict geometric validation rules.
    """

    def __init__(self):
        self._profiles: Dict[str, CalibrationProfile] = {}
        self._initialize_default_calibrations()

    def _initialize_default_calibrations(self):
        # Optical profiles for CAM-001..CAM-008
        for i in range(1, 9):
            cal_id = f"CAL-OPT-{i:03d}"
            sns_id = f"SNS-CAM{i:03d}-RGB"
            self._profiles[cal_id] = CalibrationProfile(
                calibration_id=cal_id,
                sensor_id=sns_id,
                modality=SensorModality.OPTICAL_RGB,
                status=CalibrationStatus.VALIDATED,
                optical_intrinsics=OpticalIntrinsics(),
                notes="Standard optical factory calibration",
            )

        # Thermal profiles for CAM-005, CAM-006
        for i in [5, 6]:
            cal_id = f"CAL-THM-{i:03d}"
            sns_id = f"SNS-CAM{i:03d}-LWIR"
            self._profiles[cal_id] = CalibrationProfile(
                calibration_id=cal_id,
                sensor_id=sns_id,
                modality=SensorModality.THERMAL_LWIR,
                status=CalibrationStatus.VALIDATED,
                thermal_intrinsics=ThermalIntrinsics(),
                notes="LWIR microbolometer bench calibration",
            )

        # Dual-camera CAM-007 co-located calibration
        cal_id_7 = "CAL-DUAL-007"
        self._profiles[cal_id_7] = CalibrationProfile(
            calibration_id=cal_id_7,
            sensor_id="SNS-CAM007-RGB",
            paired_sensor_id="SNS-CAM007-LWIR",
            modality=SensorModality.OPTICAL_RGB,
            status=CalibrationStatus.VALIDATED,
            optical_intrinsics=OpticalIntrinsics(),
            thermal_intrinsics=ThermalIntrinsics(),
            cross_spectral_extrinsics=CrossSpectralExtrinsics(
                relative_rotation_euler=[0.1, -0.2, 0.05],
                relative_translation_m=[0.085, 0.01, 0.0],
                rmse_alignment_px=0.74,
            ),
            notes="Co-located dual-sensor rig calibration verified",
        )

    def register_profile(self, profile: CalibrationProfile) -> CalibrationProfile:
        self._profiles[profile.calibration_id] = profile
        logger.info(f"Registered calibration profile {profile.calibration_id} for {profile.sensor_id} [{profile.status.value}]")
        return profile

    def get_profile(self, calibration_id: str) -> Optional[CalibrationProfile]:
        return self._profiles.get(calibration_id)

    def get_sensor_profile(self, sensor_id: str) -> Optional[CalibrationProfile]:
        for p in self._profiles.values():
            if p.sensor_id == sensor_id or p.paired_sensor_id == sensor_id:
                return p
        return None

    def get_paired_profile(self, sensor_a: str, sensor_b: str) -> Optional[CalibrationProfile]:
        for p in self._profiles.values():
            if (p.sensor_id == sensor_a and p.paired_sensor_id == sensor_b) or \
               (p.sensor_id == sensor_b and p.paired_sensor_id == sensor_a):
                return p
        return None

    def can_perform_geometric_fusion(self, optical_sensor_id: str, thermal_sensor_id: str) -> tuple[bool, str]:
        """
        Determines if cross-spectral geometric fusion is legally permitted.
        Requires validated extrinsics between the two sensors.
        """
        # Search for shared/dual profile or paired profiles
        profile = None
        for p in self._profiles.values():
            if (p.sensor_id == optical_sensor_id and p.paired_sensor_id == thermal_sensor_id) or \
               (p.sensor_id == thermal_sensor_id and p.paired_sensor_id == optical_sensor_id):
                profile = p
                break

        if not profile:
            return False, "MISSING_EXTRINSICS: No cross-spectral extrinsic calibration found between sensors."

        if profile.status == CalibrationStatus.EXPIRED:
            return False, "EXPIRED_CALIBRATION: Cross-spectral calibration has expired."

        if profile.status != CalibrationStatus.VALIDATED:
            return False, f"UNVALIDATED_CALIBRATION: Extrinsic calibration is {profile.status.value}; geometric fusion disabled."

        if not profile.cross_spectral_extrinsics:
            return False, "NULL_EXTRINSICS: Extrinsics block is empty."

        if profile.cross_spectral_extrinsics.rmse_alignment_px > 3.0:
            return False, f"HIGH_RMSE_ERROR: Calibration alignment error {profile.cross_spectral_extrinsics.rmse_alignment_px}px exceeds 3.0px threshold."

        return True, "VALIDATED_GEOMETRIC_FUSION_PERMITTED"

    def reset(self):
        self._profiles.clear()
        self._initialize_default_calibrations()


sensor_calibration_service = SensorCalibrationService()
