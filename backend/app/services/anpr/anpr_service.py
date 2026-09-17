import logging
import numpy as np
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from backend.app.core.database import SessionLocal
from backend.app.models.event import PlateEvent, SecurityEvent
from backend.app.services.anpr.ocr_engine import ocr_engine
from backend.app.services.anpr.normalizer import PlateNormalizer
from backend.app.services.anpr.validator import PlateValidator
from backend.app.services.anpr.watchlist import watchlist_service
from backend.app.services.risk_engine import risk_engine
from backend.app.services.alert_dispatch import alert_dispatch

logger = logging.getLogger("AnprService")


class AnprService:
    def __init__(self):
        self.validator = PlateValidator(required_hits=3, consensus_ratio=0.6)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.validated_tracks = set()
        self._validated_lock = threading.Lock()

    def queue_depth(self) -> int:
        return self.executor._work_queue.qsize()

    def reset_camera(self, camera_id: str):
        with self._validated_lock:
            self.validated_tracks = {
                k for k in self.validated_tracks if k[0] != camera_id
            }

    def process_frame(
        self, frame: np.ndarray, camera_id: str, detections: List[Dict[str, Any]]
    ):
        """
        Process a frame for ANPR.
        Only looks at bounding boxes of tracked vehicles.
        Offloads heavy OCR inference to a background thread pool.
        """
        if frame is None or not detections:
            return

        vehicle_detections = [
            d
            for d in detections
            if d.get("class") in ["car", "motorcycle", "bus", "truck"]
            and d.get("track_id")
        ]

        # Don't queue up thousands of OCR jobs if they pile up
        if self.queue_depth() > 5:
            logger.debug(
                f"[{camera_id}] ANPR queue full, skipping frame to prevent OOM"
            )
            return

        self.executor.submit(
            self._run_ocr_async, frame.copy(), camera_id, vehicle_detections
        )

    def _run_ocr_async(
        self,
        frame: np.ndarray,
        camera_id: str,
        vehicle_detections: List[Dict[str, Any]],
    ):
        import uuid
        import json

        source_frame_id = str(uuid.uuid4())
        for det in vehicle_detections:
            track_id = det["track_id"]

            # Skip OCR if this track has already been successfully validated
            track_key = (camera_id, track_id)
            with self._validated_lock:
                already_validated = track_key in self.validated_tracks
            if already_validated:
                continue

            x1, y1, x2, y2 = map(int, det["box"])

            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            vehicle_crop = frame[y1:y2, x1:x2]

            # Run OCR on the crop
            results = ocr_engine.extract_text(vehicle_crop)

            for res in results:
                raw_text = res["text"]
                conf = res["confidence"]

                normalized = PlateNormalizer.normalize(raw_text)

                if PlateNormalizer.is_valid_format(normalized):
                    logger.debug(
                        f"[{camera_id}] Found plate text '{normalized}' (conf: {conf:.2f}) for track {track_id}"
                    )

                    # Add to validator
                    validation_result = self.validator.add_reading(
                        track_id, normalized, conf
                    )

                    if validation_result:
                        best_plate, avg_conf = validation_result

                        plate_bbox = [
                            int(x1 + res["bbox"][0][0]),
                            int(y1 + res["bbox"][0][1]),
                            int(x1 + res["bbox"][2][0]),
                            int(y1 + res["bbox"][2][1]),
                        ]

                        forensics = {
                            "source_frame_id": source_frame_id,
                            "vehicle_bbox": [x1, y1, x2, y2],
                            "plate_bbox": plate_bbox,
                            "raw_candidates": json.dumps([r["text"] for r in results]),
                            "validation_state": "CONSENSUS_REACHED",
                        }

                        self._handle_validated_plate(
                            camera_id, track_id, best_plate, avg_conf, forensics
                        )

    def _handle_validated_plate(
        self,
        camera_id: str,
        track_id: str,
        plate: str,
        confidence: float,
        forensics: Dict[str, Any] = None,
    ):
        """
        Handle a successfully validated plate:
        1. Check watchlist
        2. Create PlateEvent in DB
        """
        with self._validated_lock:
            self.validated_tracks.add((camera_id, track_id))
        logger.info(
            f"[{camera_id}] Plate validated: {plate} (Track: {track_id}, Conf: {confidence:.2f})"
        )

        match = watchlist_service.check_plate(plate)
        status = "WATCHLIST MATCH" if match else "CLEAN"

        if match:
            logger.warning(
                f"[{camera_id}] WATCHLIST MATCH FOR PLATE {plate} (Track {track_id}) - Details: {match}"
            )

        # Save to DB
        db: Session = None
        try:
            db = SessionLocal()
            event = PlateEvent(
                camera_id=camera_id,
                track_id=track_id,
                plate_number=plate,
                confidence=confidence,
                watchlist_status=status,
                data_origin="LIVE",
                timestamp=datetime.datetime.utcnow(),
                source_frame_id=forensics.get("source_frame_id") if forensics else None,
                vehicle_bbox_x1=forensics["vehicle_bbox"][0] if forensics else None,
                vehicle_bbox_y1=forensics["vehicle_bbox"][1] if forensics else None,
                vehicle_bbox_x2=forensics["vehicle_bbox"][2] if forensics else None,
                vehicle_bbox_y2=forensics["vehicle_bbox"][3] if forensics else None,
                plate_bbox_x1=forensics["plate_bbox"][0] if forensics else None,
                plate_bbox_y1=forensics["plate_bbox"][1] if forensics else None,
                plate_bbox_x2=forensics["plate_bbox"][2] if forensics else None,
                plate_bbox_y2=forensics["plate_bbox"][3] if forensics else None,
                raw_candidates=forensics.get("raw_candidates") if forensics else None,
                validation_state=forensics.get("validation_state")
                if forensics
                else None,
            )
            db.add(event)

            # Create a SecurityEvent if it's a watchlist match (or just generally, so risk engine can score it)
            if match:
                sec_event_dict = {
                    "event_type": "WATCHLIST_MATCH",
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "object_type": "VEHICLE",
                    "confidence": confidence,
                    "timestamp": datetime.datetime.utcnow(),
                    "watchlist_status": status,
                    "operator_notes": f"Plate {plate} matched watchlist. Notes: {match.get('notes', '')}",
                }
                augmented_evt = risk_engine.evaluate_event(sec_event_dict)
                # The value is used by the risk engine but is not persisted on
                # SecurityEvent; keeping it would make SQLAlchemy reject the event.
                augmented_evt.pop("watchlist_status", None)
                db_sec_event = SecurityEvent(**augmented_evt, data_origin="LIVE")
                db.add(db_sec_event)
                alert_dispatch.dispatch_event(augmented_evt)

            db.commit()
        except Exception as e:
            logger.error(f"Failed to save PlateEvent/SecurityEvent to DB: {e}")
            if db:
                db.rollback()
        finally:
            if db:
                db.close()


anpr_service = AnprService()
