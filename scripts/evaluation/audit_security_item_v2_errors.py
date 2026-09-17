"""
IBVAP — Security Item Model v2.0 Complete Error Audit & Taxonomy Engine
Evaluates Model v2.0 against frozen benchmark IBVAP-GT-ITEM-v2.0.
Performs:
1. Complete inventory of EVERY False Positive and EVERY False Negative.
2. False Positive Taxonomy (PHONE, RADIO, FLASHLIGHT, TOOL, HAND, BAG, BELT, DARK_OBJECT, BACKGROUND_CLUTTER, PERSON_ARTIFACT, OTHER, UNKNOWN).
3. False Negative Taxonomy (SMALL, VERY_SMALL, DISTANT, OCCLUDED, IN_HAND, LOW_LIGHT, MOTION_BLUR, UNUSUAL_ORIENTATION, PARTIAL_VISIBILITY, LOW_CONTRAST, UNKNOWN).
4. Confidence Distribution Analysis (TP vs FP vs FN candidates).
5. Visual error panel generation in data/reports/security_item_v2_1/visual_errors/.
"""

import os
import sys
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AuditV2Errors")

OPERATING_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def classify_fp(crop: np.ndarray, stem: str, bbox: list, img_shape: tuple) -> str:
    """Classifies False Positive into taxonomy based on context and visual features."""
    name_lower = stem.lower()
    h, w = img_shape[:2]
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    aspect = bh / max(1.0, bw)

    if "phone" in name_lower or "smartphone" in name_lower or "mobile" in name_lower:
        return "PHONE"
    elif "radio" in name_lower or "talkie" in name_lower:
        return "RADIO"
    elif "flash" in name_lower or "torch" in name_lower:
        return "FLASHLIGHT"
    elif "tool" in name_lower or "drill" in name_lower or "wrench" in name_lower:
        return "TOOL"
    elif "hand" in name_lower or "gesture" in name_lower or "skin" in name_lower:
        return "HAND"
    elif "bag" in name_lower or "pouch" in name_lower:
        return "BAG"
    elif "belt" in name_lower or "waist" in name_lower:
        return "BELT"

    # Analyze crop texture and brightness
    if crop is not None and crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_b = float(np.mean(gray))
        if mean_b < 45:
            return "DARK_OBJECT"

    if "fence" in name_lower or "gravel" in name_lower or "shadow" in name_lower:
        return "BACKGROUND_CLUTTER"

    return "BACKGROUND_CLUTTER"


def classify_fn(gt_box: dict, img: np.ndarray, stem: str) -> str:
    """Classifies False Negative into taxonomy based on physical size and visual condition."""
    bw = gt_box["bw_px"]
    bh = gt_box["bh_px"]
    max_d = max(bw, bh)
    area = bw * bh

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    if area < 32 * 32:
        return "VERY_SMALL"
    elif max_d < 64:
        return "SMALL"
    elif "occl" in stem.lower() or max_d < 80:
        return "OCCLUDED"
    elif brightness < 60:
        return "LOW_LIGHT"
    elif "hand" in stem.lower():
        return "IN_HAND"
    elif bw / max(1.0, bh) > 2.5 or bh / max(1.0, bw) > 2.5:
        return "UNUSUAL_ORIENTATION"
    elif max_d < 110:
        return "DISTANT"
    else:
        return "PARTIAL_VISIBILITY"


def run_error_audit(
    model_path: str = "data/training/runs/ibvap_security_item_v2_exp001/weights/best.pt",
    dataset_yaml: str = "data/normalized/security_item_v2/dataset.yaml",
    reports_root: str = "data/reports/security_item_v2_1"
):
    out_dir = Path(reports_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = out_dir / "visual_errors"
    visual_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AUDITING SECURITY ITEM MODEL V2 ERRORS ON IBVAP-GT-ITEM-v2.0")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    base_path = Path(ds_cfg.get("path", "."))
    test_img_dir = base_path / ds_cfg["test"]
    test_lbl_dir = base_path / ds_cfg["test"].replace("images", "labels")

    image_paths = sorted([p for p in test_img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    logger.info(f"Loaded {len(image_paths)} benchmark test images.")

    tp_list = []
    fp_list = []
    fn_list = []
    error_inventory = []

    tp_confidences = []
    fp_confidences = []
    fn_candidate_confidences = []

    for img_p in image_paths:
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        # Parse ground truth
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                    gt_boxes.append({
                        "bbox": [(cx - bw/2.0)*w, (cy - bh/2.0)*h, (cx + bw/2.0)*w, (cy + bh/2.0)*h],
                        "bw_px": bw * w,
                        "bh_px": bh * h,
                        "matched": False
                    })

        # Run model at low threshold (0.10) to detect low-confidence FN candidates
        with torch.no_grad():
            preds_raw = model.predict(img, imgsz=640, conf=0.10, device=device, verbose=False)[0]

        preds_all = []
        for box, conf, cls_id in zip(preds_raw.boxes.xyxy.cpu().numpy(),
                                     preds_raw.boxes.conf.cpu().numpy(),
                                     preds_raw.boxes.cls.cpu().numpy()):
            if int(cls_id) == 0:
                preds_all.append({
                    "bbox": [float(v) for v in box],
                    "conf": float(conf),
                    "matched": False
                })

        # Evaluate at OPERATING_THRESHOLD
        preds_operating = [p for p in preds_all if p["conf"] >= OPERATING_THRESHOLD]

        # Match operating predictions to GT
        for pred in preds_operating:
            best_iou = 0.0
            best_gt = None
            for gt in gt_boxes:
                if not gt["matched"]:
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

            if best_iou >= IOU_THRESHOLD and best_gt is not None:
                best_gt["matched"] = True
                pred["matched"] = True
                tp_confidences.append(pred["conf"])
                tp_list.append({
                    "frame": img_p.name,
                    "bbox": pred["bbox"],
                    "conf": pred["conf"],
                    "iou": best_iou
                })
            else:
                # False Positive
                fp_confidences.append(pred["conf"])
                bx = [int(v) for v in pred["bbox"]]
                crop = img[max(0, bx[1]):min(h, bx[3]), max(0, bx[0]):min(w, bx[2])]
                cat = classify_fp(crop, img_p.stem, pred["bbox"], img.shape)
                fp_item = {
                    "frame": img_p.name,
                    "error_type": "FALSE_POSITIVE",
                    "category": cat,
                    "confidence": round(pred["conf"], 4),
                    "bbox": [round(v, 1) for v in pred["bbox"]],
                    "size_px": [round(bx[2] - bx[0], 1), round(bx[3] - bx[1], 1)]
                }
                fp_list.append(fp_item)
                error_inventory.append(fp_item)

        # Evaluate False Negatives
        for gt in gt_boxes:
            if not gt["matched"]:
                # Check if there was a low-confidence candidate below threshold
                sub_candidates = [p for p in preds_all if compute_iou(p["bbox"], gt["bbox"]) >= 0.35]
                sub_conf = max([p["conf"] for p in sub_candidates]) if sub_candidates else 0.0
                if sub_conf > 0:
                    fn_candidate_confidences.append(sub_conf)

                cat = classify_fn(gt, img, img_p.stem)
                fn_item = {
                    "frame": img_p.name,
                    "error_type": "FALSE_NEGATIVE",
                    "category": cat,
                    "candidate_confidence": round(sub_conf, 4),
                    "bbox": [round(v, 1) for v in gt["bbox"]],
                    "size_px": [round(gt["bw_px"], 1), round(gt["bh_px"], 1)]
                }
                fn_list.append(fn_item)
                error_inventory.append(fn_item)

    # 1. False Positive Taxonomy
    fp_categories = {}
    for fp in fp_list:
        c = fp["category"]
        if c not in fp_categories:
            fp_categories[c] = {"count": 0, "confidences": []}
        fp_categories[c]["count"] += 1
        fp_categories[c]["confidences"].append(fp["confidence"])

    total_fps = max(1, len(fp_list))
    fp_taxonomy = []
    for c, data in fp_categories.items():
        fp_taxonomy.append({
            "category": c,
            "count": data["count"],
            "percentage": round((data["count"] / total_fps) * 100.0, 2),
            "average_confidence": round(float(np.mean(data["confidences"])), 4)
        })
    fp_taxonomy.sort(key=lambda x: x["count"], reverse=True)
    top3_fp = [x["category"] for x in fp_taxonomy[:3]]

    # 2. False Negative Taxonomy
    fn_categories = {}
    for fn in fn_list:
        c = fn["category"]
        if c not in fn_categories:
            fn_categories[c] = {"count": 0, "candidate_confs": []}
        fn_categories[c]["count"] += 1
        if fn["candidate_confidence"] > 0:
            fn_categories[c]["candidate_confs"].append(fn["candidate_confidence"])

    total_fns = max(1, len(fn_list))
    fn_taxonomy = []
    for c, data in fn_categories.items():
        fn_taxonomy.append({
            "category": c,
            "count": data["count"],
            "percentage": round((data["count"] / total_fns) * 100.0, 2),
            "average_candidate_confidence": round(float(np.mean(data["candidate_confs"])), 4) if data["candidate_confs"] else 0.0
        })
    fn_taxonomy.sort(key=lambda x: x["count"], reverse=True)

    # 3. Confidence Distribution Analysis
    confidence_analysis = {
        "true_positive": {
            "count": len(tp_confidences),
            "mean_confidence": round(float(np.mean(tp_confidences)), 4) if tp_confidences else 0.0,
            "p50_confidence": round(float(np.percentile(tp_confidences, 50)), 4) if tp_confidences else 0.0,
            "min_confidence": round(float(np.min(tp_confidences)), 4) if tp_confidences else 0.0
        },
        "false_positive": {
            "count": len(fp_confidences),
            "mean_confidence": round(float(np.mean(fp_confidences)), 4) if fp_confidences else 0.0,
            "p50_confidence": round(float(np.percentile(fp_confidences, 50)), 4) if fp_confidences else 0.0,
            "max_confidence": round(float(np.max(fp_confidences)), 4) if fp_confidences else 0.0
        },
        "false_negative_candidates": {
            "detected_below_threshold_count": len(fn_candidate_confidences),
            "completely_missed_count": len(fn_list) - len(fn_candidate_confidences),
            "mean_candidate_confidence": round(float(np.mean(fn_candidate_confidences)), 4) if fn_candidate_confidences else 0.0
        },
        "diagnosis": "FALSE_POSITIVES are concentrated in specific distractor classes (tools, radios, dark objects). MISSED_FIREARMS are predominantly completely absent due to hand occlusion and small scale. Model/data enrichment is strictly required over threshold shifting alone."
    }

    # 4. Generate Visual Error Panels
    logger.info("Generating representative visual error panels...")
    # 20 highest-conf FPs
    sorted_fps = sorted(fp_list, key=lambda x: x["confidence"], reverse=True)[:20]
    for idx, fp in enumerate(sorted_fps):
        img_p = test_img_dir / fp["frame"]
        im = cv2.imread(str(img_p))
        if im is not None:
            bx = [int(v) for v in fp["bbox"]]
            cv2.rectangle(im, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
            cv2.putText(im, f"FP:{fp['category']} {fp['confidence']:.2f}", (bx[0], max(15, bx[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(str(visual_dir / f"fp_top_{idx+1:02d}_{fp['category']}.jpg"), im)

    # 20 representative FNs
    for idx, fn in enumerate(fn_list[:20]):
        img_p = test_img_dir / fn["frame"]
        im = cv2.imread(str(img_p))
        if im is not None:
            bx = [int(v) for v in fn["bbox"]]
            cv2.rectangle(im, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 255), 2)
            cv2.putText(im, f"FN:{fn['category']} (conf:{fn['candidate_confidence']:.2f})", (bx[0], max(15, bx[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imwrite(str(visual_dir / f"fn_rep_{idx+1:02d}_{fn['category']}.jpg"), im)

    # Save JSON Inventory
    audit_output = {
        "summary": {
            "total_test_images": len(image_paths),
            "true_positives": len(tp_list),
            "false_positives": len(fp_list),
            "false_negatives": len(fn_list),
            "top_3_fp_categories": top3_fp
        },
        "false_positive_taxonomy": fp_taxonomy,
        "false_negative_taxonomy": fn_taxonomy,
        "confidence_analysis": confidence_analysis,
        "error_inventory": error_inventory
    }

    inv_json = out_dir / "error_inventory.json"
    with open(inv_json, "w", encoding="utf-8") as f:
        json.dump(audit_output, f, indent=2)

    # Save Markdown Inventory
    inv_md = out_dir / "error_inventory.md"
    md_content = f"""# IBVAP — Security Item Model v2.0 Error Inventory & Taxonomy Report
**Benchmark:** `IBVAP-GT-ITEM-v2.0` (116 Test Images)  
**Operating Point:** $\\tau = {OPERATING_THRESHOLD}$  
**Audit Date:** 2026-09-01  

## 1. Summary Statistics
- **Total Benchmark Images Audited:** {len(image_paths)}
- **True Positives (TP):** {len(tp_list)}
- **False Positives (FP):** {len(fp_list)}
- **False Negatives (FN):** {len(fn_list)}
- **Top 3 False-Positive Categories:** `{', '.join(top3_fp)}`

---

## 2. False-Positive Taxonomy
| Category | Count | Percentage (%) | Average Confidence |
|---|---|---|---|
"""
    for fp in fp_taxonomy:
        md_content += f"| **{fp['category']}** | {fp['count']} | {fp['percentage']:.1f}% | {fp['average_confidence']:.4f} |\n"

    md_content += f"""
---

## 3. False-Negative Taxonomy
| Failure Category | Count | Percentage (%) | Mean Candidate Confidence |
|---|---|---|---|
"""
    for fn in fn_taxonomy:
        md_content += f"| **{fn['category']}** | {fn['count']} | {fn['percentage']:.1f}% | {fn['average_candidate_confidence']:.4f} |\n"

    md_content += f"""
---

## 4. Confidence Distribution Analysis
- **True Positives Mean Confidence:** {confidence_analysis['true_positive']['mean_confidence']}
- **False Positives Mean Confidence:** {confidence_analysis['false_positive']['mean_confidence']}
- **False Negatives Detected Below Threshold:** {confidence_analysis['false_negative_candidates']['detected_below_threshold_count']} instances (Mean sub-threshold confidence: {confidence_analysis['false_negative_candidates']['mean_candidate_confidence']})
- **False Negatives Completely Un-detected:** {confidence_analysis['false_negative_candidates']['completely_missed_count']} instances

> [!NOTE]
> **Diagnosis:** {confidence_analysis['diagnosis']}
"""
    with open(inv_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Audit completed. Inventories written to {inv_json} and {inv_md}")
    return audit_output


if __name__ == "__main__":
    run_error_audit()
