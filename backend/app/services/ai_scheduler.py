import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
import numpy as np

from backend.app.services.detection_service import detection_service
from backend.app.services.tracking_service import tracking_service
from backend.app.services.border_rules_service import border_rules_service
from backend.app.services.anpr.anpr_service import anpr_service
from backend.app.services.behavior_service import behavior_service
from backend.app.services.face_service import face_service

logger = logging.getLogger("AIScheduler")


class CameraAIState:
    """State and metrics tracking for a single camera stream."""

    def __init__(self, camera_id: str, target_ai_fps: float = 8.0):
        self.camera_id = camera_id
        self.target_ai_fps = target_ai_fps
        self.min_interval = 1.0 / target_ai_fps
        self.last_inference_time = 0.0

        # Bounded buffer: stores only the latest frame
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.has_new_frame = False

        # Current active tracked detections
        self.current_detections: List[Dict[str, Any]] = []
        self.detections_lock = threading.Lock()
        self.is_processing = False

        # O-04: Fine-grained Stage Profiling Metrics (ms)
        self.capture_ms = 0.0
        self.inference_ms = 0.0
        self.tracking_ms = 0.0
        self.zone_ms = 0.0
        self.anpr_ms = 0.0
        self.behavior_ms = 0.0
        self.face_ms = 0.0
        self.total_pipeline_ms = 0.0

        self.detector_fps = 0.0
        self.inference_count = 0
        self.fps_timer = time.time()
        self.last_ai_result_at: Optional[datetime] = None

    def submit_frame(self, frame: np.ndarray, capture_ms: float = 0.0):
        """Latest-Frame-Wins submission: immediately overwrites older unconsumed frames."""
        with self.frame_lock:
            self.latest_frame = frame
            self.has_new_frame = True
            self.capture_ms = capture_ms

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            if self.has_new_frame and self.latest_frame is not None:
                self.has_new_frame = False
                return self.latest_frame
            return None

    def update_stage_metrics(
        self,
        inference_ms: float,
        tracking_ms: float,
        zone_ms: float,
        anpr_ms: float,
        behavior_ms: float,
        face_ms: float,
        total_ms: float,
    ):
        """O-04: Update granular stage profiling timings."""
        self.inference_ms = round(inference_ms, 2)
        self.tracking_ms = round(tracking_ms, 2)
        self.zone_ms = round(zone_ms, 2)
        self.anpr_ms = round(anpr_ms, 2)
        self.behavior_ms = round(behavior_ms, 2)
        self.face_ms = round(face_ms, 2)
        self.total_pipeline_ms = round(total_ms, 2)
        self.last_ai_result_at = datetime.now(timezone.utc)
        self.inference_count += 1

        now = time.time()
        elapsed = now - self.fps_timer
        if elapsed >= 2.0:
            self.detector_fps = round(self.inference_count / elapsed, 1)
            self.inference_count = 0
            self.fps_timer = now


class AIScheduler:
    """Central asynchronous multi-camera AI scheduler with parallel worker pool.

    Pulls latest frames from camera buffers, runs GPU inference and multi-stage
    tracking/rules evaluation concurrently without blocking video playback.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cameras: Dict[str, CameraAIState] = {}
            cls._instance.running = False
            cls._instance.worker_thread = None
            cls._instance.lock = threading.Lock()
            # Dedicated ThreadPoolExecutor for multi-camera parallel inference (tuned to prevent GIL/CPU starvation)
            cls._instance.executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="AISchedulerWorker"
            )
        return cls._instance

    def register_camera(self, camera_id: str, target_ai_fps: float = 3.0):
        with self.lock:
            if camera_id not in self.cameras:
                self.cameras[camera_id] = CameraAIState(camera_id, target_ai_fps)
                logger.info(
                    f"Registered camera {camera_id} with target AI rate: {target_ai_fps} FPS"
                )

    def unregister_camera(self, camera_id: str):
        with self.lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                logger.info(f"Unregistered camera {camera_id}")

    def submit_frame(self, camera_id: str, frame: np.ndarray, capture_ms: float = 0.0):
        if camera_id in self.cameras:
            self.cameras[camera_id].submit_frame(frame, capture_ms)

    def get_detections(self, camera_id: str) -> List[Dict[str, Any]]:
        state = self.cameras.get(camera_id)
        if state:
            with state.detections_lock:
                return list(state.current_detections)
        return []

    def get_metrics(self, camera_id: str) -> Dict[str, Any]:
        state = self.cameras.get(camera_id)
        if state:
            age_sec = None
            if state.last_ai_result_at:
                age_sec = round(
                    (
                        datetime.now(timezone.utc) - state.last_ai_result_at
                    ).total_seconds(),
                    1,
                )
            return {
                "detector_fps": state.detector_fps,
                "capture_ms": state.capture_ms,
                "inference_ms": state.inference_ms,
                "tracking_ms": state.tracking_ms,
                "zone_ms": state.zone_ms,
                "anpr_ms": state.anpr_ms,
                "behavior_ms": state.behavior_ms,
                "face_ms": state.face_ms,
                "total_pipeline_ms": state.total_pipeline_ms,
                "last_ai_result_at": state.last_ai_result_at.isoformat()
                if state.last_ai_result_at
                else None,
                "age_seconds": age_sec,
            }
        return {}

    def start(self):
        if self.running:
            return
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._dispatch_loop, name="Central-AI-Dispatcher", daemon=True
        )
        self.worker_thread.start()
        logger.info("Central AI Worker Thread & Parallel ThreadPool started.")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=False)

    def _process_camera(self, state: CameraAIState, frame: np.ndarray):
        """P-06: Independent camera inference execution worker task."""
        pipe_start = time.perf_counter()
        state.is_processing = True
        try:
            # 1. Detection Forward Pass
            raw_dets, inference_ms = detection_service.predict_raw(frame, imgsz=640)

            # 2. Tracking Association
            t0_track = time.perf_counter()
            tracker = tracking_service.get_tracker(state.camera_id)
            tracked_dets = tracker.update_detections(raw_dets)
            tracking_ms = (time.perf_counter() - t0_track) * 1000.0

            with state.detections_lock:
                state.current_detections = tracked_dets

            # 3. Security Rules & Behavioral Evaluation (O-04: Profiling)
            t0_zone = time.perf_counter()
            border_rules_service.process_detections(
                state.camera_id, tracked_dets, frame
            )
            zone_ms = (time.perf_counter() - t0_zone) * 1000.0

            t0_anpr = time.perf_counter()
            anpr_service.process_frame(frame, state.camera_id, tracked_dets)
            anpr_ms = (time.perf_counter() - t0_anpr) * 1000.0

            t0_beh = time.perf_counter()
            behavior_service.process_detections(state.camera_id, tracked_dets, frame)
            behavior_ms = (time.perf_counter() - t0_beh) * 1000.0

            t0_face = time.perf_counter()
            face_service.process_detections(state.camera_id, tracked_dets, frame)
            face_ms = (time.perf_counter() - t0_face) * 1000.0

            total_pipe_ms = (time.perf_counter() - pipe_start) * 1000.0
            state.update_stage_metrics(
                inference_ms=inference_ms,
                tracking_ms=tracking_ms,
                zone_ms=zone_ms,
                anpr_ms=anpr_ms,
                behavior_ms=behavior_ms,
                face_ms=face_ms,
                total_ms=total_pipe_ms,
            )
        except Exception as exc:
            logger.error(
                f"[{state.camera_id}] Error in AI pipeline: {exc}", exc_info=False
            )
        finally:
            state.is_processing = False

    def _dispatch_loop(self):
        """Continuous parallel scheduling loop across all active cameras."""
        log_timer = time.time()

        while self.running:
            now = time.time()
            work_dispatched = 0

            with self.lock:
                camera_states = list(self.cameras.values())

            for state in camera_states:
                if state.is_processing:
                    continue

                # Rate-limiting check: only process if time since last inference exceeds min_interval
                if now - state.last_inference_time < state.min_interval:
                    continue

                frame = state.get_latest_frame()
                if frame is None:
                    continue

                state.last_inference_time = now
                work_dispatched += 1
                # P-06: Dispatch camera task to worker pool
                self.executor.submit(self._process_camera, state, frame)

            # Periodic structured logging every 10 seconds
            if now - log_timer >= 10.0:
                log_timer = now
                summary = []
                for state in camera_states:
                    summary.append(
                        f"[{state.camera_id}] AI_FPS={state.detector_fps:.1f} "
                        f"Inf={state.inference_ms:.1f}ms Trk={state.tracking_ms:.1f}ms "
                        f"Zone={state.zone_ms:.1f}ms ANPR={state.anpr_ms:.1f}ms "
                        f"Tot={state.total_pipeline_ms:.1f}ms"
                    )
                if summary:
                    logger.info("AI PIPELINE TELEMETRY: " + " | ".join(summary))

            # Small sleep to prevent 100% CPU spinning when no cameras need frames
            if work_dispatched == 0:
                time.sleep(0.005)


ai_scheduler = AIScheduler()
