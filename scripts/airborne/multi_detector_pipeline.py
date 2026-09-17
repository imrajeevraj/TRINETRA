"""
IBVAP — Multi-Detector Coordinated Pipeline (Phase 9-14)
Coordinates Ground Model v2.0 and Airborne Model v1.1 in a decoupled architecture.
Merges detections through ByteTracker and VirtualFenceEngine, emitting standardized telemetry.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

from scripts.tracking.byte_tracker import ByteTracker
from scripts.fence.virtual_fence_engine import VirtualFenceEngine, VirtualFenceZone

logger = logging.getLogger("MultiDetectorPipeline")

class MultiDetectorPipeline:
    def __init__(
        self,
        ground_model_path: str = "models/current/ibvap_detector.pt",
        airborne_model_path: Optional[str] = None,
        security_item_model_path: Optional[str] = "data/training/runs/ibvap_security_item_v1_exp001/weights/best.pt",
        ground_conf: float = 0.35,
        airborne_conf: float = 0.40,
        security_item_conf: float = 0.35,
        ground_imgsz: int = 768,
        airborne_imgsz: int = 640,
        security_item_imgsz: int = 640,
        device: str = "cuda:0"
    ):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.ground_conf = ground_conf
        self.airborne_conf = airborne_conf
        self.security_item_conf = security_item_conf
        self.ground_imgsz = ground_imgsz
        self.airborne_imgsz = airborne_imgsz
        self.security_item_imgsz = security_item_imgsz

        logger.info(f"Initializing Three-Detector Pipeline on {self.device.upper()}...")
        
        # 1. Ground Detector
        self.ground_model = YOLO(ground_model_path)
        self.ground_model.to(self.device)
        self.ground_names = {0: "person", 1: "vehicle"}

        # 2. Airborne Detector
        self.airborne_model = None
        if airborne_model_path:
            self.airborne_model = YOLO(airborne_model_path)
            self.airborne_model.to(self.device)
            self.airborne_names = {0: "drone", 1: "aircraft"}

        # 3. Security Item Detector (Firearms)
        self.security_item_model = None
        if security_item_model_path and Path(security_item_model_path).exists():
            self.security_item_model = YOLO(security_item_model_path)
            self.security_item_model.to(self.device)
            self.security_item_names = {0: "firearm"}

        # 4. Tracker & Virtual Fence Engine
        self.tracker = ByteTracker(high_thresh=0.35, low_thresh=0.15)
        self.fence_engine = VirtualFenceEngine()

    def add_virtual_fence(self, zone: VirtualFenceZone):
        self.fence_engine.add_zone(zone)

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM-01"
    ) -> Dict[str, Any]:
        """
        Executes decoupled parallel inference across ground and airborne detectors.
        Merges detections, updates tracks, evaluates virtual fences, and outputs telemetry.
        """
        t0 = time.perf_counter()
        raw_detections = []

        # 1. Ground Model Forward Pass
        if self.ground_model is not None and frame is not None:
            g_res = self.ground_model.predict(
                source=frame,
                imgsz=self.ground_imgsz,
                conf=self.ground_conf,
                device=self.device,
                verbose=False
            )[0]
            for b, s, c in zip(g_res.boxes.xyxy.cpu().numpy(), g_res.boxes.conf.cpu().numpy(), g_res.boxes.cls.cpu().numpy().astype(int)):
                if c in self.ground_names:
                    raw_detections.append({
                        "bbox": [float(v) for v in b],
                        "confidence": float(s),
                        "class_id": c,
                        "class_name": self.ground_names[c],
                        "detector_id": "ground",
                        "detector_version": "v2.0"
                    })

        # 2. Airborne Model Forward Pass
        if self.airborne_model is not None and frame is not None:
            a_res = self.airborne_model.predict(
                source=frame,
                imgsz=self.airborne_imgsz,
                conf=self.airborne_conf,
                device=self.device,
                verbose=False
            )[0]
            for b, s, c in zip(a_res.boxes.xyxy.cpu().numpy(), a_res.boxes.conf.cpu().numpy(), a_res.boxes.cls.cpu().numpy().astype(int)):
                if c in self.airborne_names:
                    raw_detections.append({
                        "bbox": [float(v) for v in b],
                        "confidence": float(s),
                        "class_id": c,
                        "class_name": self.airborne_names[c],
                        "detector_id": "airborne",
                        "detector_version": "v1.1",
                        "domain": "AIR"
                    })

        # 3. Security Item Model Forward Pass
        if self.security_item_model is not None and frame is not None:
            s_res = self.security_item_model.predict(
                source=frame,
                imgsz=self.security_item_imgsz,
                conf=self.security_item_conf,
                device=self.device,
                verbose=False
            )[0]
            for b, s, c in zip(s_res.boxes.xyxy.cpu().numpy(), s_res.boxes.conf.cpu().numpy(), s_res.boxes.cls.cpu().numpy().astype(int)):
                if c in self.security_item_names:
                    raw_detections.append({
                        "bbox": [float(v) for v in b],
                        "confidence": float(s),
                        "class_id": c,
                        "class_name": self.security_item_names[c],
                        "detector_id": "security_item",
                        "detector_version": "v1.0",
                        "domain": "SECURITY_ITEM"
                    })

        t1 = time.perf_counter()
        inference_latency_ms = (t1 - t0) * 1000.0

        # 4. Update Multi-Object Tracker
        tracked_objects = self.tracker.update(raw_detections)

        # 5. Evaluate Virtual Fences
        annotated_detections, fence_events = self.fence_engine.process_tracks(
            tracked_objects=tracked_objects,
            camera_id=camera_id,
            detector_version_map={"ground": "v2.0", "airborne": "v1.1", "security_item": "v1.0"}
        )

        t2 = time.perf_counter()
        total_latency_ms = (t2 - t0) * 1000.0

        return {
            "camera_id": camera_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_detections": len(annotated_detections),
            "detections": annotated_detections,
            "fence_events": fence_events,
            "latency_ms": round(total_latency_ms, 2),
            "inference_latency_ms": round(inference_latency_ms, 2),
            "fps": round(1000.0 / max(1.0, total_latency_ms), 1)
        }
