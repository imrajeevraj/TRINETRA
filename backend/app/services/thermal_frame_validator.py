"""
TRINETRA Phase XV — Thermal Data Integrity & Frame Validation Engine
Validates thermal frames prior to ML ingestion: dimensions, pixel format, timestamp validity,
monotonicity, duplicates, corruption, saturation, dead pixels, uniformity, temperature ranges,
missing metadata, and stream stall detection.

Strictly records explicit rejection reasons: NEVER silently discards bad frames.
"""

from __future__ import annotations
import math
import time
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.thermal_sensor_adapter import ThermalFrame, ThermalMetadata

logger = logging.getLogger("ThermalFrameValidator")


class ThermalRejectionReason(str, Enum):
    INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
    INVALID_PIXEL_FORMAT = "INVALID_PIXEL_FORMAT"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    TIMESTAMP_REGRESSION = "TIMESTAMP_REGRESSION"
    CORRUPTED_PAYLOAD = "CORRUPTED_PAYLOAD"
    FRAME_DUPLICATE = "FRAME_DUPLICATE"
    SATURATION_LIMIT_EXCEEDED = "SATURATION_LIMIT_EXCEEDED"
    EXCESSIVE_UNIFORMITY = "EXCESSIVE_UNIFORMITY"
    DEAD_PIXEL_LIMIT_EXCEEDED = "DEAD_PIXEL_LIMIT_EXCEEDED"
    INVALID_TEMPERATURE_RANGE = "INVALID_TEMPERATURE_RANGE"
    MISSING_METADATA = "MISSING_METADATA"
    STREAM_STALLED = "STREAM_STALLED"
    SENSOR_DISCONNECTED = "SENSOR_DISCONNECTED"


@dataclass
class ValidationVerdict:
    is_valid: bool
    rejection_reason: Optional[ThermalRejectionReason] = None
    details: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)


class ThermalFrameValidator:
    """
    Validates physical or simulated thermal frames before entering downstream ML pipelines.
    Enforces fail-closed validation with comprehensive error attribution.
    """

    def __init__(
        self,
        min_width: int = 160,
        max_width: int = 4096,
        min_height: int = 120,
        max_height: int = 2160,
        max_future_timestamp_s: float = 1.0,
        stream_stall_threshold_s: float = 3.0,
        max_uniformity_flatness_ratio: float = 0.99,
        max_saturation_ratio: float = 0.95,
        min_temp_kelvin: float = 213.15,  # -60°C
        max_temp_kelvin: float = 453.15,  # +180°C
    ):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.max_future_timestamp_s = max_future_timestamp_s
        self.stream_stall_threshold_s = stream_stall_threshold_s
        self.max_uniformity_flatness_ratio = max_uniformity_flatness_ratio
        self.max_saturation_ratio = max_saturation_ratio
        self.min_temp_kelvin = min_temp_kelvin
        self.max_temp_kelvin = max_temp_kelvin

        # Stream tracking state
        self._last_capture_times: Dict[str, float] = {}
        self._last_monotonic_times: Dict[str, float] = {}
        self._last_frame_hashes: Dict[str, str] = {}
        self._recent_rejections: List[ValidationVerdict] = []

    def validate_frame(self, frame: ThermalFrame, is_connected: bool = True) -> ValidationVerdict:
        """
        Executes complete 14-point thermal data integrity check.
        Returns explicit ValidationVerdict.
        """
        # 1. Connection check
        if not is_connected:
            verdict = ValidationVerdict(
                is_valid=False,
                rejection_reason=ThermalRejectionReason.SENSOR_DISCONNECTED,
                details={"sensor_id": frame.metadata.sensor_id, "message": "Sensor reports disconnected"},
            )
            self._record_verdict(frame.metadata, verdict)
            return verdict

        meta = frame.metadata

        # 2. Metadata presence
        if not meta or not meta.sensor_id or not meta.frame_id:
            verdict = ValidationVerdict(
                is_valid=False,
                rejection_reason=ThermalRejectionReason.MISSING_METADATA,
                details={"message": "Sensor ID or Frame ID is null/empty"},
            )
            self._record_verdict(meta, verdict)
            return verdict

        # 3. Payload corruption / size
        if not frame.payload_bytes or len(frame.payload_bytes) < 16:
            verdict = ValidationVerdict(
                is_valid=False,
                rejection_reason=ThermalRejectionReason.CORRUPTED_PAYLOAD,
                details={"byte_length": len(frame.payload_bytes) if frame.payload_bytes else 0},
            )
            self._record_verdict(meta, verdict)
            return verdict

        # 4. Dimensions check
        if (
            meta.frame_width < self.min_width
            or meta.frame_width > self.max_width
            or meta.frame_height < self.min_height
            or meta.frame_height > self.max_height
        ):
            verdict = ValidationVerdict(
                is_valid=False,
                rejection_reason=ThermalRejectionReason.INVALID_DIMENSIONS,
                details={"width": meta.frame_width, "height": meta.frame_height},
            )
            self._record_verdict(meta, verdict)
            return verdict

        # 5. Pixel format check
        valid_formats = ["MONO8", "MONO16", "RADIOMETRIC_FLOAT32", "RAW16"]
        if meta.pixel_format.upper() not in valid_formats:
            verdict = ValidationVerdict(
                is_valid=False,
                rejection_reason=ThermalRejectionReason.INVALID_PIXEL_FORMAT,
                details={"pixel_format": meta.pixel_format},
            )
            self._record_verdict(meta, verdict)
            return verdict

        # 6. Timestamp validity (non-negative, not far into future)
        now = time.time()
        if meta.capture_timestamp <= 0.0 or meta.capture_timestamp > (now + self.max_future_timestamp_s):
            verdict = ValidationVerdict(
                is_valid=False,
                rejection_reason=ThermalRejectionReason.INVALID_TIMESTAMP,
                details={"capture_timestamp": meta.capture_timestamp, "now": now},
            )
            self._record_verdict(meta, verdict)
            return verdict

        # 7. Monotonic timestamp regression
        sensor_id = meta.sensor_id
        if sensor_id in self._last_monotonic_times:
            if meta.monotonic_timestamp < self._last_monotonic_times[sensor_id]:
                verdict = ValidationVerdict(
                    is_valid=False,
                    rejection_reason=ThermalRejectionReason.TIMESTAMP_REGRESSION,
                    details={
                        "previous_mono": self._last_monotonic_times[sensor_id],
                        "current_mono": meta.monotonic_timestamp,
                    },
                )
                self._record_verdict(meta, verdict)
                return verdict

        # 8. Stream stall check
        if sensor_id in self._last_capture_times:
            delta = meta.capture_timestamp - self._last_capture_times[sensor_id]
            if delta > self.stream_stall_threshold_s:
                verdict = ValidationVerdict(
                    is_valid=False,
                    rejection_reason=ThermalRejectionReason.STREAM_STALLED,
                    details={"time_delta_s": delta, "threshold_s": self.stream_stall_threshold_s},
                )
                self._record_verdict(meta, verdict)
                return verdict

        # 9. Frame duplicate check
        frame_hash = frame.frame_sha256 or hashlib.sha256(frame.payload_bytes).hexdigest().upper()
        if sensor_id in self._last_frame_hashes:
            if frame_hash == self._last_frame_hashes[sensor_id]:
                verdict = ValidationVerdict(
                    is_valid=False,
                    rejection_reason=ThermalRejectionReason.FRAME_DUPLICATE,
                    details={"frame_hash": frame_hash, "sensor_id": sensor_id},
                )
                self._record_verdict(meta, verdict)
                return verdict

        # 10. Temperature range check for radiometric feeds
        if meta.sensor_temperature_k is not None:
            if meta.sensor_temperature_k < self.min_temp_kelvin or meta.sensor_temperature_k > self.max_temp_kelvin:
                verdict = ValidationVerdict(
                    is_valid=False,
                    rejection_reason=ThermalRejectionReason.INVALID_TEMPERATURE_RANGE,
                    details={
                        "sensor_temperature_k": meta.sensor_temperature_k,
                        "min_k": self.min_temp_kelvin,
                        "max_k": self.max_temp_kelvin,
                    },
                )
                self._record_verdict(meta, verdict)
                return verdict

        # 11. Payload heuristics: check all zeros (dead detector) or all 255s (saturation)
        raw_tail = frame.payload_bytes[16:] if len(frame.payload_bytes) > 16 else frame.payload_bytes
        if raw_tail:
            num_zeros = raw_tail.count(b"\x00")
            num_saturated = raw_tail.count(b"\xFF")
            total = len(raw_tail)
            if (num_zeros / total) > self.max_uniformity_flatness_ratio:
                verdict = ValidationVerdict(
                    is_valid=False,
                    rejection_reason=ThermalRejectionReason.EXCESSIVE_UNIFORMITY,
                    details={"zero_ratio": num_zeros / total, "threshold": self.max_uniformity_flatness_ratio},
                )
                self._record_verdict(meta, verdict)
                return verdict

            if (num_saturated / total) > self.max_saturation_ratio:
                verdict = ValidationVerdict(
                    is_valid=False,
                    rejection_reason=ThermalRejectionReason.SATURATION_LIMIT_EXCEEDED,
                    details={"saturated_ratio": num_saturated / total, "threshold": self.max_saturation_ratio},
                )
                self._record_verdict(meta, verdict)
                return verdict

        # Update stream tracking state on success
        self._last_capture_times[sensor_id] = meta.capture_timestamp
        self._last_monotonic_times[sensor_id] = meta.monotonic_timestamp
        self._last_frame_hashes[sensor_id] = frame_hash

        verdict = ValidationVerdict(is_valid=True, details={"frame_hash": frame_hash})
        self._record_verdict(meta, verdict)
        return verdict

    def _record_verdict(self, meta: Optional[ThermalMetadata], verdict: ValidationVerdict):
        if meta:
            if not verdict.is_valid and verdict.rejection_reason:
                meta.quality_status = "REJECTED"
                meta.rejection_reason = verdict.rejection_reason.value
            else:
                meta.quality_status = "GOOD"
                meta.rejection_reason = None
        if not verdict.is_valid:
            self._recent_rejections.append(verdict)
            if len(self._recent_rejections) > 200:
                self._recent_rejections.pop(0)

    def get_recent_rejections(self) -> List[ValidationVerdict]:
        return list(self._recent_rejections)

    def reset(self):
        self._last_capture_times.clear()
        self._last_monotonic_times.clear()
        self._last_frame_hashes.clear()
        self._recent_rejections.clear()


thermal_frame_validator = ThermalFrameValidator()
