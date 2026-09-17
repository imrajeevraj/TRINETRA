"""
TRINETRA Phase XIV — Multimodal Sensor Time Synchronization Service
Manages temporal alignment across optical and thermal sensor observations.
Provides bounded temporal association windows, handles frame delays, drops,
and out-of-order deliveries, and assigns synchronization statuses.
"""

from __future__ import annotations
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from backend.app.services.sensor_abstraction import SensorObservation

logger = logging.getLogger("SensorSync")


class SynchronizationStatus(str, Enum):
    SYNCED = "SYNCED"
    PARTIALLY_SYNCED = "PARTIALLY_SYNCED"
    UNSYNCED = "UNSYNCED"
    STALE = "STALE"


@dataclass
class SyncEvaluationResult:
    optical_sensor_id: str
    thermal_sensor_id: str
    optical_timestamp: float
    thermal_timestamp: float
    time_delta_ms: float
    status: SynchronizationStatus
    clock_source: str
    frame_sequence_delta: int
    is_aligned: bool
    reason: str = ""


class SensorSyncService:
    """
    Evaluates sub-second temporal alignment between optical and thermal streams.
    Does not assume perfect synchronization.
    """

    def __init__(self, max_sync_window_ms: float = 50.0, stale_threshold_ms: float = 1000.0):
        self.max_sync_window_ms = max_sync_window_ms
        self.stale_threshold_ms = stale_threshold_ms
        # Sensor observations buffers: sensor_id -> list of observations
        self._buffers: Dict[str, List[SensorObservation]] = {}
        self._sync_history: List[SyncEvaluationResult] = []

    def buffer_observation(self, obs: SensorObservation, max_buffer_size: int = 100):
        if obs.sensor_id not in self._buffers:
            self._buffers[obs.sensor_id] = []
        buf = self._buffers[obs.sensor_id]
        buf.append(obs)
        # Keep sorted by timestamp to handle out-of-order frames
        buf.sort(key=lambda x: x.timestamp)
        if len(buf) > max_buffer_size:
            buf.pop(0)

    def evaluate_pair(
        self,
        optical_obs: SensorObservation,
        thermal_obs: SensorObservation,
        clock_source: str = "PTP_IEEE_1588",
    ) -> SyncEvaluationResult:
        """
        Evaluates temporal difference between optical and thermal observations.
        """
        now = time.time()
        opt_age_ms = (now - optical_obs.timestamp) * 1000.0
        thm_age_ms = (now - thermal_obs.timestamp) * 1000.0

        delta_sec = abs(optical_obs.timestamp - thermal_obs.timestamp)
        delta_ms = round(delta_sec * 1000.0, 2)
        seq_delta = abs(optical_obs.sequence_number - thermal_obs.sequence_number)

        if opt_age_ms > self.stale_threshold_ms or thm_age_ms > self.stale_threshold_ms:
            status = SynchronizationStatus.STALE
            is_aligned = False
            reason = f"Observations are stale (Opt age: {opt_age_ms:.1f}ms, Thm age: {thm_age_ms:.1f}ms)"
        elif delta_ms <= self.max_sync_window_ms:
            status = SynchronizationStatus.SYNCED
            is_aligned = True
            reason = f"Well-aligned within {self.max_sync_window_ms}ms window (delta: {delta_ms:.2f}ms)"
        elif delta_ms <= (self.max_sync_window_ms * 2.0):
            status = SynchronizationStatus.PARTIALLY_SYNCED
            is_aligned = True
            reason = f"Partially aligned (delta: {delta_ms:.2f}ms)"
        else:
            status = SynchronizationStatus.UNSYNCED
            is_aligned = False
            reason = f"Synchronization drift exceeded limits (delta: {delta_ms:.2f}ms)"

        res = SyncEvaluationResult(
            optical_sensor_id=optical_obs.sensor_id,
            thermal_sensor_id=thermal_obs.sensor_id,
            optical_timestamp=optical_obs.timestamp,
            thermal_timestamp=thermal_obs.timestamp,
            time_delta_ms=delta_ms,
            status=status,
            clock_source=clock_source,
            frame_sequence_delta=seq_delta,
            is_aligned=is_aligned,
            reason=reason,
        )
        self._sync_history.append(res)
        if len(self._sync_history) > 500:
            self._sync_history.pop(0)
        return res

    def find_best_temporal_match(
        self,
        optical_sensor_id: str,
        thermal_sensor_id: str,
        target_timestamp: float,
        tolerance_ms: Optional[float] = None,
    ) -> Optional[Tuple[SensorObservation, SensorObservation, SyncEvaluationResult]]:
        """
        Finds temporally nearest matching pair within bounded association window.
        """
        tol_ms = tolerance_ms if tolerance_ms is not None else self.max_sync_window_ms
        opt_buf = self._buffers.get(optical_sensor_id, [])
        thm_buf = self._buffers.get(thermal_sensor_id, [])

        if not opt_buf or not thm_buf:
            return None

        # Find closest optical
        best_opt = min(opt_buf, key=lambda o: abs(o.timestamp - target_timestamp))
        # Find closest thermal to that optical
        best_thm = min(thm_buf, key=lambda t: abs(t.timestamp - best_opt.timestamp))

        eval_res = self.evaluate_pair(best_opt, best_thm)
        if eval_res.time_delta_ms <= tol_ms:
            return best_opt, best_thm, eval_res
        return None

    def get_latest_sync_status(self, optical_sensor_id: str, thermal_sensor_id: str) -> SynchronizationStatus:
        for item in reversed(self._sync_history):
            if item.optical_sensor_id == optical_sensor_id and item.thermal_sensor_id == thermal_sensor_id:
                return item.status
        return SynchronizationStatus.SYNCED

    def reset(self):
        self._buffers.clear()
        self._sync_history.clear()


sensor_sync_service = SensorSyncService()
