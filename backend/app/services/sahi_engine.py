"""
TRINETRA — SAHI-Style Tiled Inference Engine (Phase VI)
Enables Slicing Aided Hyper Inference across overlapping frame tiles
for high-sensitivity small-object / distant-pedestrian detection.
"""

import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

logger = logging.getLogger("SAHIEngine")


def calculate_tiles(
    img_w: int,
    img_h: int,
    grid_rows: int = 2,
    grid_cols: int = 2,
    overlap: float = 0.20
) -> List[Tuple[int, int, int, int]]:
    """
    Computes overlapping bounding boxes (x1, y1, x2, y2) for regular tile grid.
    Guarantees full image coverage with deterministic bounds.
    """
    tiles = []
    # Calculate nominal tile dimensions
    tile_w = int(img_w / (grid_cols - (grid_cols - 1) * overlap))
    tile_h = int(img_h / (grid_rows - (grid_rows - 1) * overlap))

    # Clamp
    tile_w = min(img_w, max(128, tile_w))
    tile_h = min(img_h, max(128, tile_h))

    step_x = max(1, int(tile_w * (1.0 - overlap)))
    step_y = max(1, int(tile_h * (1.0 - overlap)))

    y_starts = list(range(0, img_h - tile_h + 1, step_y))
    if not y_starts or (y_starts[-1] + tile_h < img_h):
        y_starts.append(img_h - tile_h)

    x_starts = list(range(0, img_w - tile_w + 1, step_x))
    if not x_starts or (x_starts[-1] + tile_w < img_w):
        x_starts.append(img_w - tile_w)

    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(img_w, x1 + tile_w)
            y2 = min(img_h, y1 + tile_h)
            tiles.append((x1, y1, x2, y2))

    return tiles


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.50) -> List[int]:
    """Standard Greedy NMS on [x1, y1, x2, y2] bounding boxes."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


class SAHIEngine:
    def __init__(
        self,
        grid_rows: int = 2,
        grid_cols: int = 2,
        overlap: float = 0.20,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.50,
        device: str = "cuda:0"
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.overlap = overlap
        self.imgsz = imgsz
        self.conf = conf_thresh
        self.nms_thresh = nms_thresh
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

    def predict_sahi(
        self,
        model: YOLO,
        frame: np.ndarray,
        include_full_frame: bool = True
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes SAHI tiled inference over frame, shifts coordinates, and suppresses duplicates.
        Fail-Safe: If tiling fails, executes standard full-frame fallback.
        """
        if frame is None or model is None:
            return [], 0.0

        t0 = time.perf_counter()
        img_h, img_w = frame.shape[:2]

        all_boxes = []
        all_scores = []
        all_classes = []

        try:
            # 1. Full frame pass if enabled
            if include_full_frame:
                ff_res = model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                if len(ff_res.boxes) > 0:
                    for b, s, c in zip(ff_res.boxes.xyxy.cpu().numpy(), ff_res.boxes.conf.cpu().numpy(), ff_res.boxes.cls.cpu().numpy()):
                        all_boxes.append(b)
                        all_scores.append(float(s))
                        all_classes.append(int(c))

            # 2. Compute tiles
            tiles = calculate_tiles(img_w, img_h, self.grid_rows, self.grid_cols, self.overlap)

            # 3. Predict on each tile
            for tx1, ty1, tx2, ty2 in tiles:
                tile_img = frame[ty1:ty2, tx1:tx2]
                if tile_img.size == 0:
                    continue

                res = model.predict(source=tile_img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                if len(res.boxes) > 0:
                    for b, s, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                        # Shift tile coords back to full-frame coords
                        bx1 = b[0] + tx1
                        by1 = b[1] + ty1
                        bx2 = b[2] + tx1
                        by2 = b[3] + ty1
                        all_boxes.append([bx1, by1, bx2, by2])
                        all_scores.append(float(s))
                        all_classes.append(int(c))

            # 4. Cross-tile Non-Maximum Suppression per class
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
            logger.error(f"SAHI tiled inference failed ({e}), invoking fail-safe full-frame fallback.")
            try:
                t_fallback = time.perf_counter()
                ff_res = model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
                dets = []
                if len(ff_res.boxes) > 0:
                    for b, s, c in zip(ff_res.boxes.xyxy.cpu().numpy(), ff_res.boxes.conf.cpu().numpy(), ff_res.boxes.cls.cpu().numpy()):
                        dets.append({
                            "class_id": int(c),
                            "bbox": b.tolist(),
                            "confidence": float(s)
                        })
                return dets, (time.perf_counter() - t_fallback) * 1000.0
            except Exception as e2:
                logger.error(f"Fallback full-frame inference also failed: {e2}")
                return [], (time.perf_counter() - t0) * 1000.0
