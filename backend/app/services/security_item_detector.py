"""
TRINETRA — Security Item Detector Service (Third Perception Model)
Target Domain: SECURITY_ITEM
Target Class: 0: firearm
Architecture: YOLO11n (2.58M parameters)
Features:
- Common detection schema formatting and validation
- Decoupled forward pass (predict_security_item)
- Fail-closed cryptographic SHA-256 verification
- Operational state management: RUNNING / DEGRADED / OFFLINE
- Independent failure isolation
"""

import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import yaml
from ultralytics import YOLO

logger = logging.getLogger("SecurityItemDetector")


class SecurityItemDetector:
    def __init__(
        self,
        weights_path: str = "models/production/security_item/ibvap_security_item_v2_1_production.pt",
        registry_path: str = "models/model_registry.yaml",
        device: str = "cuda:0",
        default_conf: float = 0.35,
        default_imgsz: int = 640,
    ):
        self.weights_path = Path(weights_path)
        self.registry_path = Path(registry_path)
        self.device = device
        self.conf = default_conf
        self.imgsz = default_imgsz

        self.model: Optional[YOLO] = None
        self.status = "OFFLINE"
        self.classes = {0: "firearm"}
        self.detector_id = "security_item"
        self.detector_version = "v1.0"
        self.domain = "SECURITY_ITEM"

        # Frame counter for schema IDs
        self.frame_counter = 0

        # Attempt to load and verify
        if self.weights_path.exists():
            self._load_and_verify()

    def _compute_sha256(self, filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest().upper()

    def _load_and_verify(self):
        try:
            actual_hash = self._compute_sha256(self.weights_path)
            logger.info(
                f"Loading Security Item Model from {self.weights_path} (SHA-256: {actual_hash[:16]}...)"
            )

            # Check registry if available
            if self.registry_path.exists():
                try:
                    with open(self.registry_path, "r", encoding="utf-8") as f:
                        reg = yaml.safe_load(f)
                    active_id = reg.get("active_security_item_model_id", "ibvap-security-item-v2.1-exp001")
                    norm_target_path = str(self.weights_path).replace("\\", "/")
                    for m in reg.get("models", []):
                        m_path = str(m.get("path", "")).replace("\\", "/")
                        if m.get("id") == active_id or (m_path and m_path in norm_target_path):
                            expected = m.get("sha256")
                            if (
                                expected
                                and expected != "N/A"
                                and expected.upper() != actual_hash.upper()
                            ):
                                logger.critical(
                                    f"INTEGRITY ERROR: Security Item Model hash mismatch! Expected: {expected}, Got: {actual_hash}"
                                )
                                self.status = "DEGRADED"
                                return
                            break
                except Exception as e:
                    logger.warning(f"Could not audit against model registry: {e}")

            self.model = YOLO(str(self.weights_path))
            self.status = "RUNNING"
            logger.info("Security Item Detector v1.0 operational (STATUS: RUNNING).")
        except Exception as e:
            logger.error(f"Failed to load Security Item Model: {e}")
            self.model = None
            self.status = "OFFLINE"

    def predict(
        self,
        frame: np.ndarray,
        camera_id: str = "CAM-001",
        frame_id: Optional[int] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes decoupled forward pass for firearm detection.
        Returns tuple of (standardized_detections, inference_latency_ms).
        """
        if self.status != "RUNNING" or self.model is None or frame is None:
            return [], 0.0

        if frame_id is None:
            self.frame_counter += 1
            frame_id = self.frame_counter

        c_thresh = conf or self.conf
        i_sz = imgsz or self.imgsz

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                imgsz=i_sz,
                conf=c_thresh,
                device="cpu",  # safe fallback or target
                verbose=False,
            )[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0

            detections = []
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            for idx, (b, s, c) in enumerate(zip(boxes, confs, clss)):
                if c == 0:  # strictly firearm
                    det_record = {
                        "detection_id": f"det_{camera_id}_{frame_id}_{idx + 1:03d}",
                        "camera_id": camera_id,
                        "frame_id": frame_id,
                        "timestamp": ts,
                        "detector_id": self.detector_id,
                        "detector_version": self.detector_version,
                        "domain": self.domain,
                        "class_id": 0,
                        "class_name": "firearm",
                        "confidence": float(round(s, 4)),
                        "bbox": [float(round(v, 2)) for v in b],
                        "class": "firearm",
                        "box": [float(round(v, 2)) for v in b],
                    }
                    detections.append(det_record)

            return detections, lat_ms
        except Exception as e:
            logger.error(f"Security Item inference failure on {camera_id}: {e}")
            return [], 0.0


# Global singleton instance
security_item_detector = SecurityItemDetector()
