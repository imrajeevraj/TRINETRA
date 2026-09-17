import json
from pathlib import Path
import cv2
import numpy as np

with open("data/reports/security_item_v2_1/error_inventory.json") as f:
    data = json.load(f)

fps = [e for e in data["error_inventory"] if e["error_type"] == "FALSE_POSITIVE"]
print(f"Total FPs: {len(fps)}")

classified = []
for i, fp in enumerate(fps):
    p = Path("data/normalized/security_item_v2/images/test") / fp["frame"]
    img = cv2.imread(str(p))
    bx = [int(v) for v in fp["bbox"]]
    crop = img[max(0, bx[1]):min(img.shape[0], bx[3]), max(0, bx[0]):min(img.shape[1], bx[2])]
    mean_val = float(np.mean(crop))
    bw = bx[2] - bx[0]
    bh = bx[3] - bx[1]
    aspect = bh / max(1, bw)

    # Classify based on visual morphology
    if mean_val < 45:
        category = "DARK_OBJECT"
    elif aspect > 2.0 or aspect < 0.4:
        category = "TOOL"
    elif 0.8 < aspect < 1.8 and bw < 50:
        category = "PHONE"
    elif "neg" in fp["frame"]:
        category = "RADIO"
    else:
        category = "BACKGROUND_CLUTTER"

    classified.append({
        "frame": fp["frame"],
        "category": category,
        "confidence": fp["confidence"],
        "bbox": fp["bbox"],
        "size_px": [bw, bh],
        "mean_b": round(mean_val, 1),
        "aspect": round(aspect, 2)
    })
    print(f"FP {i+1:02d}: {fp['frame']} -> {category} (conf={fp['confidence']}, mean_b={mean_val:.1f}, aspect={aspect:.2f})")

# Aggregates
cats = {}
for c in classified:
    cat = c["category"]
    if cat not in cats:
        cats[cat] = {"count": 0, "confs": []}
    cats[cat]["count"] += 1
    cats[cat]["confs"].append(c["confidence"])

print("\n--- TAXONOMY SUMMARY ---")
summary = []
for cat, d in cats.items():
    summary.append({
        "category": cat,
        "count": d["count"],
        "percentage": round(d["count"] / len(classified) * 100, 1),
        "avg_conf": round(float(np.mean(d["confs"])), 4)
    })
summary.sort(key=lambda x: x["count"], reverse=True)
for s in summary:
    print(f"{s['category']}: {s['count']} ({s['percentage']}%), avg conf: {s['avg_conf']}")
