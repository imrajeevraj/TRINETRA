"""
TRINETRA — Dynamic ROI Crop Inference Engine (Phase VI)
Enables targeted high-resolution crop inference on low-confidence triggers
and small-pedestrian clusters without incurring the computational cost of full-image tiling.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from backend.app.services.sahi_engine import nms_xyxy

logger = logging.getLogger("DynamicROIEngine")


class DynamicROIEngine:
    def __init__(
        self,
        max_rois_per_frame: int = 2,
        trigger_min_conf: float = 0.15,
        trigger_max_conf: float = 0.35,
        small_object_max_h: float = 90.0,
        padding_ratio: float = 0.50,
        min_crop_size: int = 384,
        crop_imgsz: int = 640,
        nms_thresh: float = 0.50,
        device: str = "cuda:0"
    ):
        self.max_rois = max_rois_per_frame
        self.trigger_min_conf = trigger_min_conf
        self.trigger_max_conf = trigger_max_conf
        self.small_h = small_object_max_h
        self.padding_ratio = padding_ratio
        self.min_crop_size = min_crop_size
        self.crop_imgsz = crop_imgsz
        self.nms_thresh = nms_thresh
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

    def extract_roi_boxes(
        self,
        primary_preds: List[Dict[str, Any]],
        img_w: int,
        img_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Identifies up to max_rois high-priority crop candidates."""
        candidates = []
        for p in primary_preds:
            box = p["bbox"]
            conf = p["confidence"]
            bw = box[2] - box[0]
            bh = box[3] - box[1]

            # Condition 1: Low confidence suspicious trigger
            is_low_conf = (self.trigger_min_conf <= conf < self.trigger_max_conf)
            # Condition 2: Small pedestrian target
            is_small = (p.get("class_id", 0) == 0 and bh <= self.small_h)

            if is_low_conf or is_small:
                # Priority: lowest confidence or smallest height
                score = (1.0 - conf) + (1.0 - min(1.0, bh / 150.0))
                candidates.append((score, box))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_rois = []

        for _, box in candidates[:self.max_rois]:
            bx1, by1, bx2, by2 = box
            bw = bx2 - bx1
            bh = by2 - by1

            pad_w = int(bw * self.padding_ratio)
            pad_h = int(bh * self.padding_ratio)

            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0

            crop_w = max(self.min_crop_size, bw + 2 * pad_w)
            crop_h = max(self.min_crop_size, bh + 2 * pad_h)

            x1 = max(0, int(cx - crop_w / 2.0))
            y1 = max(0, int(cy - crop_h / 2.0))
            x2 = min(img_w, x1 + int(crop_w))
            y2 = min(img_h, y1 + int(crop_h))

            # Adjust if clamped
            if (x2 - x1) < self.min_crop_size and img_w >= self.min_crop_size:
                x1 = max(0, x2 - self.min_crop_size)
            if (y2 - y1) < self.min_crop_size and img_h >= self.min_crop_size:
                y1 = max(0, y2 - self.min_crop_size)

            selected_rois.append((x1, y1, x2, y2))

        return selected_rois

    def predict_dynamic_roi(
        self,
        model: YOLO,
        frame: np.ndarray,
        primary_detections: Optional[List[Dict[str, Any]]] = None,
        primary_conf: float = 0.25
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes primary inference, extracts suspicious dynamic ROIs, runs second-stage
        high-resolution crop inference, and merges detections via NMS.
        Fail-safe: Returns primary detections if ROI processing encounters an error.
        """
        if frame is None or model is None:
            return [], 0.0

        t0 = time.perf_counter()
        img_h, img_w = frame.shape[:2]

        all_boxes = []
        all_scores = []
        all_classes = []

        try:
            # 1. Primary Full Frame Inference (if not pre-computed)
            if primary_detections is None:
                ff_res = model.predict(source=frame, imgsz=self.crop_imgsz, conf=self.trigger_min_conf, device=self.device, verbose=False)[0]
                primary_detections = []
                if len(ff_res.boxes) > 0:
                    for b, s, c in zip(ff_res.boxes.xyxy.cpu().numpy(), ff_res.boxes.conf.cpu().numpy(), ff_res.boxes.cls.cpu().numpy()):
                        primary_detections.append({
                            "class_id": int(c),
                            "bbox": b.tolist(),
                            "confidence": float(s)
                        })

            # Add qualifying primary detections (>= primary_conf)
            for p in primary_detections:
                if p["confidence"] >= primary_conf:
                    all_boxes.append(p["bbox"])
                    all_scores.append(p["confidence"])
                    all_classes.append(p["class_id"])

            # 2. Extract Candidate Dynamic ROIs
            rois = self.extract_roi_boxes(primary_detections, img_w, img_h)

            # 3. Predict on extracted crops
            for rx1, ry1, rx2, ry2 in rois:
                crop = frame[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    continue

                c_res = model.predict(source=crop, imgsz=self.crop_imgsz, conf=primary_conf, device=self.device, verbose=False)[0]
                if len(c_res.boxes) > 0:
                    for b, s, c in zip(c_res.boxes.xyxy.cpu().numpy(), c_res.boxes.conf.cpu().numpy(), c_res.boxes.cls.cpu().numpy()):
                        # Remap to full-frame coordinates
                        remap_b = [b[0] + rx1, b[1] + ry1, b[2] + rx1, b[3] + ry1]
                        all_boxes.append(remap_b)
                        all_scores.append(float(s))
                        all_classes.append(int(c))

            # 4. Cross-Crop NMS Merge
            final_detections = []
            if len(all_boxes) > 0:
                boxes_arr = np.array(all_boxes)
                scores_arr = np.array(all_scores)
                classes_arr = np.array(all_classes)

                for cid in np.unique(classes_arr):
                    mask = classes_arr == cid
                    cls_boxes = boxes_arr[mask]
                    cls_scores = scores_arr[mask]
                    keep_idx = nms_xyxy(cls_boxes, cls_scores, iou_thresh=self.nms_thresh)

                    for kid in keep_idx:
                        final_detections.append({
                            "class_id": int(cid),
                            "bbox": cls_boxes[kid].tolist(),
                            "confidence": float(cls_scores[kid])
                        })

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return final_detections, latency_ms

        except Exception as e:
            logger.error(f"Dynamic ROI inference failure ({e}), falling back to primary detections.")
            return primary_detections or [], (time.perf_counter() - t0) * 1000.0
