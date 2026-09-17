import os
import cv2
import time
import threading
import logging
import numpy as np
from urllib.parse import urlparse
from typing import Dict, Optional, Generator
from sqlalchemy.orm import Session
from datetime import datetime

from backend.app.core.config import get_cameras_config
from backend.app.core.database import SessionLocal
from backend.app.models.camera import Camera
from backend.app.services.detection_service import detection_service
from backend.app.services.tracking_service import tracking_service
from backend.app.services.ai_scheduler import ai_scheduler
from backend.app.services.evidence_service import evidence_service

logger = logging.getLogger("CameraManager")


class CameraStreamThread(threading.Thread):
    """
    Dedicated video ingestion and streaming thread for a single camera.
    Decoupled from AI inference:
    - Ingests frames at native 30 FPS with zero blocking.
    - Hands off frames to the central AI scheduler via bounded Latest-Frame-Wins queue.
    - Generates real-time display predictions at 30 FPS for buttery-smooth bounding box motion.
    - Uses on-demand JPEG encoding to avoid wasting CPU cycles when no client is streaming.
    """

    def __init__(self, camera_id: str, name: str, source: str, target_fps: int = 15):
        super().__init__()
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.target_fps = target_fps
        self.daemon = True
        self.stop_event = threading.Event()

        self.running = False
        self.raw_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Display cache
        self._cached_jpeg: Optional[bytes] = None
        self._cached_jpeg_time: float = 0.0

        # Telemetry
        self.fps = 0.0
        self.resolution = "0x0"
        self.status = "OFFLINE"
        self.frame_count = 0
        self.last_update = time.time()

        self.abs_source = self._validate_source(source)
        # Register in AI scheduler with 1.0 FPS target to maintain responsive CPU and prevent OOM
        ai_scheduler.register_camera(camera_id, target_ai_fps=1.0)

    @staticmethod
    def _validate_source(source: str):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Camera source must be a non-empty string")
        source = source.strip()
        if source.isdigit():
            return int(source)

        parsed = urlparse(source)
        if parsed.scheme:
            if parsed.scheme != "rtsp" or parsed.username or parsed.password:
                raise ValueError("Only credential-free rtsp sources are supported")
            if not parsed.hostname:
                raise ValueError("RTSP source must include a hostname")
            return source

        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../data/videos")
        )
        candidate = os.path.abspath(source)
        try:
            within_root = os.path.commonpath([root, candidate]) == root
        except ValueError:
            within_root = False
        if not within_root or not os.path.isfile(candidate):
            raise ValueError("Video source must be an existing file under data/videos")
        return candidate

    def run(self):
        self.running = True
        self.status = "OFFLINE"
        self.update_db_status()
        retry_delay = 3.0

        while self.running and not self.stop_event.is_set():
            logger.info(f"[{self.camera_id}] Connecting to source: {self.source}")
            cap = cv2.VideoCapture(self.abs_source)
            if not cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open camera source")
                self.status = "NO_SIGNAL"
                self.update_db_status()
                self.stop_event.wait(retry_delay)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0 or video_fps > 60:
                video_fps = self.target_fps

            self.resolution = f"{width}x{height}"
            self.status = "ONLINE"
            self.update_db_status()
            logger.info(
                f"[{self.camera_id}] Stream online: {self.resolution} @ {video_fps:.1f} FPS"
            )

            frame_duration = 1.0 / video_fps
            start_time = time.time()
            self.frame_count = 0

            while self.running and not self.stop_event.is_set() and cap.isOpened():
                loop_start = time.perf_counter()
                ret, frame = cap.read()

                if not ret:
                    # Video loop logic for local test files
                    if not isinstance(self.abs_source, int) and not str(
                        self.abs_source
                    ).startswith("rtsp://"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tracking_service.reset_camera(self.camera_id)
                        continue
                    else:
                        self.status = "NO_SIGNAL"
                        self.update_db_status()
                        break

                t_capture = (time.perf_counter() - loop_start) * 1000.0

                # Cap resolution to 960x540 to prevent memory pressure on CPU and Windows heap
                if frame is not None:
                    h, w = frame.shape[:2]
                    if w > 960 or h > 540:
                        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)

                with self.frame_lock:
                    self.raw_frame = frame

                # Hand off frame to rolling evidence buffer (lightweight sampled)
                evidence_service.record_frame(self.camera_id, frame)

                # Asynchronously submit latest frame to AI Scheduler (zero blocking)
                ai_scheduler.submit_frame(self.camera_id, frame, t_capture)

                # Advance tracker state on display frame for smooth 30 FPS visualization
                tracker = tracking_service.get_tracker(self.camera_id)
                tracker.predict_tracks()

                # FPS Meter
                self.frame_count += 1
                now = time.time()
                elapsed = now - start_time
                if elapsed >= 2.0:
                    self.fps = round(self.frame_count / elapsed, 1)
                    self.frame_count = 0
                    start_time = now
                self.last_update = now

                # Precise rate-pacing without CPU busy-wait spinlocks
                process_time = time.perf_counter() - loop_start
                sleep_time = frame_duration - process_time
                if sleep_time > 0.001:
                    self.stop_event.wait(sleep_time)

            cap.release()
            if self.running:
                self.status = "OFFLINE"
                self.update_db_status()
                self.stop_event.wait(retry_delay)

        self.status = "OFFLINE"
        self.update_db_status()

    def get_annotated_jpeg(
        self, quality: int = 65, max_width: int = 960
    ) -> Optional[bytes]:
        """
        Generate annotated tactical JPEG on demand with cached rate-limiting.
        Avoids encoding when no client is actively requesting the stream.
        """
        now = time.time()
        # Serve from cache if requested within 40ms (max 25 FPS encode rate to protect memory)
        if self._cached_jpeg and (now - self._cached_jpeg_time) < 0.040:
            return self._cached_jpeg

        with self.frame_lock:
            if self.raw_frame is None:
                return None
            try:
                frame = self.raw_frame.copy()
            except (MemoryError, Exception):
                return self._cached_jpeg

        # Render zones and virtual fences
        frame = detection_service.draw_zones(frame, self.camera_id)

        # Get active smoothed tracks
        tracker = tracking_service.get_tracker(self.camera_id)
        current_tracks = tracker.get_current_tracks()

        # Draw bounding boxes
        if current_tracks:
            frame = detection_service.draw_detections(frame, current_tracks)

        # Tactical HUD overlay
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        metrics = ai_scheduler.get_metrics(self.camera_id)
        ai_fps = metrics.get("detector_fps", 0.0)
        hud_text = f"{self.camera_id} | {timestamp_str} | STREAM: {self.fps:.1f} FPS | AI: {ai_fps:.1f} FPS"

        cv2.putText(
            frame,
            hud_text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 210, 235),
            1,
            cv2.LINE_AA,
        )

        # Scale down if needed for fast network transfer
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(
                frame, (max_width, int(h * scale)), interpolation=cv2.INTER_LINEAR
            )

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpeg_bytes = buffer.tobytes()
        self._cached_jpeg = jpeg_bytes
        self._cached_jpeg_time = now
        return jpeg_bytes

    def update_db_status(self):
        db: Session = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if camera:
                camera.status = self.status
                camera.resolution = self.resolution
                if self.status == "ONLINE":
                    camera.last_seen = datetime.utcnow()
                db.commit()
        except Exception as e:
            logger.error(f"[{self.camera_id}] DB status update error: {e}")
            db.rollback()
        finally:
            db.close()

    def stop(self):
        self.running = False
        self.stop_event.set()


class CameraManager:
    """Singleton Camera Manager controlling ingestion across all surveillance feeds."""

    _instance = None
    _lock = threading.Lock()

    # R-04: Maximum concurrent MJPEG stream clients across all cameras.
    MAX_CONCURRENT_STREAMS: int = 32

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self):
        if self.initialized:
            return
        self.streams: Dict[str, CameraStreamThread] = {}
        self._active_stream_clients: int = 0
        self._stream_clients_lock = threading.Lock()
        self.initialized = True

    def initialize_cameras_in_db(self):
        db: Session = SessionLocal()
        try:
            cameras_conf = get_cameras_config()
            for conf in cameras_conf:
                camera = db.query(Camera).filter(Camera.id == conf["id"]).first()
                if not camera:
                    camera = Camera(
                        id=conf["id"],
                        name=conf["name"],
                        source=conf["source"],
                        location=conf["location"],
                        resolution=conf["resolution"],
                        fps=conf["fps"],
                        is_active=conf["is_active"],
                        status="OFFLINE",
                    )
                    db.add(camera)
                else:
                    camera.name = conf["name"]
                    camera.source = conf["source"]
                    camera.location = conf["location"]
                    camera.resolution = conf["resolution"]
                    camera.fps = conf["fps"]
                    camera.is_active = conf["is_active"]
            db.commit()
        except Exception as e:
            logger.error(f"Failed to initialize cameras in database: {e}")
            db.rollback()
        finally:
            db.close()

    def start_all(self):
        # Start AI scheduler first
        ai_scheduler.start()
        self.initialize_cameras_in_db()
        cameras_conf = get_cameras_config()

        for conf in cameras_conf:
            if not conf.get("is_active", True):
                continue

            camera_id = conf["id"]
            if camera_id in self.streams:
                continue

            try:
                stream = CameraStreamThread(
                    camera_id=camera_id,
                    name=conf["name"],
                    source=conf["source"],
                    target_fps=conf.get("fps", 30),
                )
            except Exception as exc:
                logger.error(f"[{camera_id}] Camera configuration rejected: {exc}")
                continue
            self.streams[camera_id] = stream
            stream.start()
            logger.info(f"Started camera feed thread for: {camera_id}")

    def stop_all(self):
        ai_scheduler.stop()
        for camera_id, stream in self.streams.items():
            stream.stop()
            logger.info(f"Stopped camera feed thread for: {camera_id}")
        for stream in self.streams.values():
            stream.join(timeout=5)
        self.streams.clear()

    def generate_mjpeg_stream(
        self, camera_id: str, idle_timeout: float = 30.0
    ) -> Generator[bytes, None, None]:
        """Generate smooth multipart MJPEG stream for client browser consumption.

        R-04: Enforces an idle timeout (default 30 s) so a stalled client
        does not hold a generator thread forever.  Also respects a global
        MAX_CONCURRENT_STREAMS cap to prevent resource exhaustion.
        """
        with self._stream_clients_lock:
            if self._active_stream_clients >= self.MAX_CONCURRENT_STREAMS:
                logger.warning(
                    "R-04: MAX_CONCURRENT_STREAMS (%d) reached; rejecting new stream for %s",
                    self.MAX_CONCURRENT_STREAMS,
                    camera_id,
                )
                return
            self._active_stream_clients += 1

        idle_since = time.time()
        try:
            while True:
                try:
                    frame_bytes = self.get_jpeg_frame(camera_id)
                except Exception:
                    frame_bytes = None

                if frame_bytes is not None:
                    idle_since = time.time()  # reset idle clock on every live frame
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                else:
                    if time.time() - idle_since > idle_timeout:
                        logger.info(
                            "R-04: MJPEG stream for %s idle for %.0fs — closing.",
                            camera_id,
                            idle_timeout,
                        )
                        break
                time.sleep(0.040)  # ~25 FPS pace for optimal bandwidth & low memory
        finally:
            with self._stream_clients_lock:
                self._active_stream_clients = max(0, self._active_stream_clients - 1)

    def get_jpeg_frame(self, camera_id: str) -> Optional[bytes]:
        stream = self.streams.get(camera_id)
        if stream and stream.status == "ONLINE":
            return stream.get_annotated_jpeg()
        return None

    def get_raw_frame(self, camera_id: str):
        stream = self.streams.get(camera_id)
        if stream and stream.status == "ONLINE":
            with stream.frame_lock:
                return stream.raw_frame
        return None

    def get_camera_status(self, camera_id: str) -> Dict:
        stream = self.streams.get(camera_id)
        metrics = ai_scheduler.get_metrics(camera_id)
        tracker = tracking_service.get_tracker(camera_id)
        current_tracks = tracker.get_current_tracks()

        if stream:
            return {
                "id": camera_id,
                "name": stream.name,
                "status": stream.status,
                "fps": round(stream.fps, 1),
                "resolution": stream.resolution,
                "last_seen": datetime.utcnow() if stream.status == "ONLINE" else None,
                "detections": current_tracks,
                "inference_fps": metrics.get("detector_fps", 0.0),
                "inference_latency_ms": metrics.get("inference_ms", 0.0),
                "tracking_latency_ms": metrics.get("tracking_ms", 0.0),
                "total_pipeline_ms": metrics.get("total_pipeline_ms", 0.0),
                "ai_result_age_seconds": metrics.get("age_seconds", None),
            }
        return {
            "id": camera_id,
            "status": "OFFLINE",
            "fps": 0.0,
            "resolution": "0x0",
            "last_seen": None,
            "detections": [],
            "inference_fps": 0.0,
            "inference_latency_ms": 0.0,
            "ai_result_age_seconds": None,
        }


camera_manager = CameraManager()
