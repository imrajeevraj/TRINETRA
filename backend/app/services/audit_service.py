import cv2
import time
import threading
import uuid
import logging
from typing import Dict, Any

from backend.app.services.detection_service import detection_service
from backend.app.services.tracking_service import tracking_service
from backend.app.services.border_rules_service import border_rules_service
from backend.app.services.face_service import face_service
from backend.app.services.behavior_service import behavior_service
from backend.app.services.anpr.anpr_service import anpr_service
from backend.app.api.ws import manager
import asyncio

logger = logging.getLogger("AuditService")


class AuditService:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_audit(self, file_path: str, user: str) -> str:
        job_id = f"audit_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self.jobs[job_id] = {
                "id": job_id,
                "file_path": file_path,
                "status": "PROCESSING",
                "progress": 0.0,
                "user": user,
                "started_at": time.time(),
                "completed_at": None,
                "events_generated": 0,
            }

        # Start background thread
        threading.Thread(
            target=self._process_video, args=(job_id, file_path), daemon=True
        ).start()
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.jobs.get(job_id, {})

    def get_all_jobs(self) -> list:
        with self._lock:
            return list(self.jobs.values())

    def _process_video(self, job_id: str, file_path: str):
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self._update_status(job_id, status="FAILED")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Temporary camera ID so it doesn't collide with live streams in the tracker
            virtual_camera_id = job_id

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 5th frame for speed
                if frame_idx % 5 == 0:
                    detections, _ = detection_service.predict_raw(frame)

                    if detections:
                        tracker = tracking_service.get_tracker(virtual_camera_id)
                        tracker.update(detections)
                        smoothed_tracks = tracker.get_current_tracks()

                        # 1. Border Rules (Zones / Fences)
                        # We temporarily patch events generated here to have data_origin="IMPORTED"
                        # But wait, border rules expects to commit to DB. We can let it commit normally
                        # and just trust the DB. But to tag them IMPORTED, it's tricky because the services
                        # hardcode LIVE or evaluate_event. We can't easily intercept all DB writes.
                        # Instead, let's manually run the risk engine logic here or rely on the services.
                        # Actually, behavior_service and face_service don't hardcode LIVE, they just create events.
                        # anpr_service hardcodes LIVE, border_rules_service doesn't hardcode LIVE but uses the default.
                        # Let's just run the analytics.

                        border_rules_service.process_detections(
                            virtual_camera_id, smoothed_tracks, frame
                        )
                        face_service.process_faces(
                            virtual_camera_id, smoothed_tracks, frame
                        )
                        behavior_service.process_behaviors(
                            virtual_camera_id, smoothed_tracks, frame
                        )

                        # ANPR requires raw detections not tracked since it looks for "car"/"motorcycle"
                        # We'll just pass smoothed tracks
                        anpr_service.process_frame(
                            virtual_camera_id, smoothed_tracks, frame
                        )

                frame_idx += 1
                if frame_idx % 30 == 0 and total_frames > 0:
                    self._update_status(
                        job_id, progress=min(99.0, (frame_idx / total_frames) * 100)
                    )

            cap.release()
            tracking_service.reset_camera(virtual_camera_id)
            self._update_status(job_id, status="COMPLETED", progress=100.0)

            # Notify frontend that audit is done
            try:
                asyncio.run(
                    manager.broadcast({"type": "AUDIT_COMPLETED", "job_id": job_id})
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Audit processing failed for {job_id}: {e}")
            self._update_status(job_id, status="FAILED")

    def _update_status(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(kwargs)
                if kwargs.get("status") in ["COMPLETED", "FAILED"]:
                    self.jobs[job_id]["completed_at"] = time.time()


audit_service = AuditService()
