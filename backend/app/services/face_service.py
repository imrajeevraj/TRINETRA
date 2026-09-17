import logging
import threading
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

from backend.app.core.database import SessionLocal
from backend.app.models.event import SecurityEvent
from backend.app.services.risk_engine import risk_engine
from backend.app.services.alert_dispatch import alert_dispatch
from backend.app.services.reid_service import reid_service

try:
    from insightface.app import FaceAnalysis

    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger("FaceService")


class FaceService:
    """
    Asynchronous Face Recognition & Watchlist Matching Service.
    Lazy-initializes InsightFace ONNX models in background thread pool.
    """

    def __init__(self):
        self.face_app = None
        self._init_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="Face-Worker"
        )
        self.recent_alerts = {}
        self.ALERT_COOLDOWN = 15.0  # seconds

    def _get_face_app(self):
        if self.face_app is None and INSIGHTFACE_AVAILABLE:
            with self._init_lock:
                if self.face_app is None:
                    try:
                        app = FaceAnalysis(
                            name="buffalo_l", root="app/data/insightface"
                        )
                        try:
                            app.prepare(ctx_id=0, det_size=(320, 320))
                        except Exception:
                            app.prepare(ctx_id=-1, det_size=(320, 320))
                        self.face_app = app
                        logger.info("InsightFace FaceAnalysis lazy-initialized.")
                    except Exception as e:
                        logger.warning(f"Could not lazy-initialize InsightFace: {e}")
        return self.face_app

    def reset_camera(self, camera_id: str):
        if camera_id in self.recent_alerts:
            self.recent_alerts[camera_id] = {}

    def process_detections(
        self, camera_id: str, detections: List[Dict[str, Any]], frame: np.ndarray
    ):
        """Asynchronously dispatch person crops for background face recognition."""
        if not INSIGHTFACE_AVAILABLE or frame is None:
            return

        # Check executor queue depth to prevent lag buildup
        if self.executor._work_queue.qsize() > 4:
            return

        current_time = time.time()
        if camera_id not in self.recent_alerts:
            self.recent_alerts[camera_id] = {}

        # Limit to max 1 face analysis per 5 seconds per camera
        last_cam_time = self.recent_alerts[camera_id].get("_last_cam_face", 0.0)
        if current_time - last_cam_time < 5.0:
            return

        person_crops = []
        for det in detections:
            if det.get("class") != "person":
                continue

            track_id = det.get("track_id")
            if track_id:
                last_alert = self.recent_alerts[camera_id].get(track_id, 0)
                if current_time - last_alert < self.ALERT_COOLDOWN:
                    continue

            x1, y1, x2, y2 = [int(v) for v in det.get("box", [0, 0, 0, 0])]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            person_crops.append((track_id, crop, (x1, y1, x2, y2)))
            if track_id:
                self.recent_alerts[camera_id][track_id] = current_time
            self.recent_alerts[camera_id]["_last_cam_face"] = current_time
            break  # Only 1 crop per cycle

        if person_crops:
            self.executor.submit(self._run_face_analysis_async, camera_id, person_crops)

    def _run_face_analysis_async(self, camera_id: str, person_crops: List[tuple]):
        """Runs in background thread pool."""
        app = self._get_face_app()
        if app is None:
            return
        try:
            for track_id, crop, (x1, y1, x2, y2) in person_crops:
                faces = app.get(crop)
                if not faces:
                    continue

                face = faces[0]
                embedding = face.embedding.tolist()
                person_name, watchlist_status = reid_service.match_embedding(embedding)

                if watchlist_status == "WATCHLIST":
                    event_type = "KNOWN_THREAT"
                elif watchlist_status == "AUTHORIZED":
                    event_type = "KNOWN_FRIENDLY"
                else:
                    continue  # Only raise alert for recognized or watchlist matches

                bx = (x1 + x2) / 2.0
                by = float(y2)

                event_dict = {
                    "event_type": event_type,
                    "camera_id": camera_id,
                    "zone_id": None,
                    "track_id": track_id or "UNKNOWN",
                    "object_type": "PERSON",
                    "direction": None,
                    "x": int(bx),
                    "y": int(by),
                    "confidence": float(face.det_score),
                    "timestamp": datetime.utcnow(),
                }

                if person_name:
                    event_dict["operator_notes"] = f"Identified Subject: {person_name}"

                # Persist event
                db = SessionLocal()
                try:
                    augmented = risk_engine.evaluate_event(event_dict)
                    db_event = SecurityEvent(**augmented)
                    db.add(db_event)
                    db.commit()
                    alert_dispatch.dispatch_event(augmented)
                except Exception as e:
                    logger.error(f"Face event persist error: {e}")
                    db.rollback()
                finally:
                    db.close()
        except Exception as exc:
            logger.debug(f"Async face analysis error: {exc}")


face_service = FaceService()
