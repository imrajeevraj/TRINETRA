import logging
import time
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from backend.app.core.database import SessionLocal
from backend.app.models.event import SecurityEvent
from backend.app.services.risk_engine import risk_engine
from backend.app.services.evidence_service import evidence_service
from backend.app.services.alert_dispatch import alert_dispatch

logger = logging.getLogger("BehaviorService")


class BehaviorService:
    def __init__(self):
        # {camera_id: {track_id: {'first_seen': timestamp, 'last_seen': timestamp, 'positions': [(x,y)], 'alerts': set()}}}
        self.track_states = {}
        self.db_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="Behavior-DB-Worker"
        )

        self.LOITERING_TIME_THRESHOLD = 5.0  # seconds
        self.LOITERING_DISTANCE_THRESHOLD = 50.0  # pixels
        self.CRAWLING_ASPECT_RATIO = 1.2
        self.PLATOON_THRESHOLD = 5
        self.TRACK_TIMEOUT = 10.0  # seconds

    def reset_camera(self, camera_id: str):
        if camera_id in self.track_states:
            self.track_states[camera_id] = {}

    def process_detections(
        self, camera_id: str, detections: List[Dict[str, Any]], frame=None
    ):
        if camera_id not in self.track_states:
            self.track_states[camera_id] = {}

        current_time = time.time()
        new_events = []
        person_count = 0
        current_tracks = set()

        for det in detections:
            obj_type = det.get("class", "UNKNOWN").lower()
            conf = det.get("confidence", 0.0)
            x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
            track_id = det.get("track_id", "UNKNOWN")

            # Contraband and drones
            if obj_type in ["drone", "airplane"]:
                new_events.append(
                    self._create_event_dict(
                        "DRONE_DETECTED",
                        camera_id,
                        None,
                        track_id,
                        "DRONE",
                        None,
                        (x1 + x2) / 2,
                        y2,
                        conf,
                    )
                )
                continue

            if obj_type in ["knife", "weapon", "gun"]:
                new_events.append(
                    self._create_event_dict(
                        "WEAPON_DETECTED",
                        camera_id,
                        None,
                        track_id,
                        "WEAPON",
                        None,
                        (x1 + x2) / 2,
                        y2,
                        conf,
                    )
                )
                continue

            if obj_type in ["backpack", "suitcase", "handbag"]:
                new_events.append(
                    self._create_event_dict(
                        "CONTRABAND_DETECTED",
                        camera_id,
                        None,
                        track_id,
                        obj_type.upper(),
                        None,
                        (x1 + x2) / 2,
                        y2,
                        conf,
                    )
                )
                continue

            if obj_type != "person":
                continue

            person_count += 1

            # Crawling check
            width = x2 - x1
            height = y2 - y1
            if height > 0 and (width / height) > self.CRAWLING_ASPECT_RATIO:
                new_events.append(
                    self._create_event_dict(
                        "CRAWLING_DETECTED",
                        camera_id,
                        None,
                        track_id,
                        "PERSON",
                        None,
                        (x1 + x2) / 2,
                        y2,
                        conf,
                    )
                )

            # Loitering check (requires tracking)
            if track_id:
                current_tracks.add(track_id)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if track_id not in self.track_states[camera_id]:
                    self.track_states[camera_id][track_id] = {
                        "first_seen": current_time,
                        "last_seen": current_time,
                        "positions": [(center_x, center_y)],
                        "alerts": set(),
                    }
                else:
                    state = self.track_states[camera_id][track_id]
                    state["last_seen"] = current_time
                    state["positions"].append((center_x, center_y))

                    if len(state["positions"]) > 50:
                        state["positions"] = state["positions"][-50:]

                    duration = current_time - state["first_seen"]

                    if (
                        duration > self.LOITERING_TIME_THRESHOLD
                        and "loitering" not in state["alerts"]
                    ):
                        xs = [p[0] for p in state["positions"]]
                        ys = [p[1] for p in state["positions"]]
                        max_dist = max(max(xs) - min(xs), max(ys) - min(ys))

                        if max_dist < self.LOITERING_DISTANCE_THRESHOLD:
                            state["alerts"].add("loitering")
                            new_events.append(
                                self._create_event_dict(
                                    "LOITERING_DETECTED",
                                    camera_id,
                                    None,
                                    track_id,
                                    "PERSON",
                                    None,
                                    center_x,
                                    center_y,
                                    conf,
                                )
                            )

        if person_count > self.PLATOON_THRESHOLD:
            new_events.append(
                self._create_event_dict(
                    "PLATOON_DETECTED",
                    camera_id,
                    None,
                    "GROUP",
                    "PERSON",
                    None,
                    0,
                    0,
                    1.0,
                )
            )

        # Cleanup lost tracks
        lost_tracks = []
        for tid, state in self.track_states[camera_id].items():
            if current_time - state["last_seen"] > self.TRACK_TIMEOUT:
                lost_tracks.append(tid)
        for tid in lost_tracks:
            del self.track_states[camera_id][tid]

        # Non-blocking async persistence
        if new_events:
            self.db_executor.submit(
                self._persist_events_async,
                new_events,
                camera_id,
                frame.copy() if frame is not None else None,
            )

    def _persist_events_async(self, events: List[Dict], camera_id: str, frame):
        db = None
        try:
            db = SessionLocal()
            persisted_events = []
            for evt in events:
                augmented_evt = risk_engine.evaluate_event(evt)
                db_event = SecurityEvent(**augmented_evt)
                db.add(db_event)
                persisted_events.append((db_event, augmented_evt))
            db.commit()
            for db_event, augmented_evt in persisted_events:
                snapshot_path, _ = evidence_service.capture_event(
                    camera_id, db_event.id, frame
                )
                if snapshot_path:
                    db_event.snapshot_path = snapshot_path
                alert_dispatch.dispatch_event(augmented_evt)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save behavioral events: {e}")
            if db:
                db.rollback()
        finally:
            if db:
                db.close()

    def _create_event_dict(
        self,
        event_type,
        camera_id,
        zone_id,
        track_id,
        object_type,
        direction,
        x,
        y,
        confidence,
    ):
        return {
            "event_type": event_type,
            "camera_id": camera_id,
            "zone_id": zone_id,
            "track_id": track_id,
            "object_type": object_type,
            "direction": direction,
            "x": int(x),
            "y": int(y),
            "confidence": float(confidence),
            "timestamp": datetime.utcnow(),
        }


behavior_service = BehaviorService()
