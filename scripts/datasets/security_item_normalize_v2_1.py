"""
IBVAP — Security Item Dataset v2.1 Normalization Engine
Builds data/normalized/security_item_v2_1/ implementing:
1. Hard Negative Dataset v3 targeting TOP 3 FP classes:
   - BACKGROUND_CLUTTER (fences, poles, shadows)
   - TOOL (wrenches, drills, hardware)
   - PHONE (smartphones in hand)
2. Positive Dataset Enrichment targeting TOP FN classes:
   - SMALL / VERY_SMALL (< 96px, < 32px)
   - OCCLUDED (hand & body occlusion masks)
   - IN_HAND / UNUSUAL_ORIENTATION (high-pitch surveillance tilt)
3. Preserves balanced ratio (~55% positives, ~45% hard negatives).
4. Strictly protects frozen benchmark IBVAP-GT-ITEM-v2.0.
"""

import os
import shutil
import logging
from pathlib import Path
import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NormalizeV2_1")

RAW_ROOT = Path("data/raw/security_items")
V2_ROOT = Path("data/normalized/security_item_v2")
OUTPUT_DIR = Path("data/normalized/security_item_v2_1")


def apply_target_fn_enrichment(img: np.ndarray, boxes: list, mode: str) -> tuple:
    """
    Applies targeted physical transformations addressing specific FN categories:
    - small_scale: scales target down to < 64px
    - occlusion: overlays synthetic hand/finger/clothing patch over weapon
    - perspective_tilt: top-down high-angle camera tilt
    """
    h, w = img.shape[:2]
    out_img = img.copy()
    out_boxes = []

    if mode == "small_scale":
        # Scale image down into a subregion (simulating distant camera)
        scale_f = np.random.uniform(0.4, 0.65)
        new_w, new_h = int(w * scale_f), int(h * scale_f)
        resized = cv2.resize(out_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Place onto a surveillance background canvas
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg_val = np.random.randint(60, 130)
        bg[:] = (bg_val, bg_val + 5, bg_val + 10)

        offset_x = np.random.randint(0, w - new_w)
        offset_y = np.random.randint(0, h - new_h)
        bg[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        out_img = bg

        for cx, cy, bw, bh in boxes:
            new_cx = (cx * new_w + offset_x) / w
            new_cy = (cy * new_h + offset_y) / h
            new_bw = (bw * new_w) / w
            new_bh = (bh * new_h) / h
            out_boxes.append((new_cx, new_cy, new_bw, new_bh))

    elif mode == "occlusion":
        # Simulate hand/clothing partially covering the weapon
        out_boxes = list(boxes)
        for cx, cy, bw, bh in boxes:
            bx1 = int((cx - bw / 2.0) * w)
            by1 = int((cy - bh / 2.0) * h)
            bx2 = int((cx + bw / 2.0) * w)
            by2 = int((cy + bh / 2.0) * h)

            # Hand/skin tone occlusion patch over ~35% of the bbox
            pw = int((bx2 - bx1) * np.random.uniform(0.3, 0.5))
            ph = int((by2 - by1) * np.random.uniform(0.3, 0.5))
            px = np.random.randint(bx1, max(bx1 + 1, bx2 - pw))
            py = np.random.randint(by1, max(by1 + 1, by2 - ph))

            # Skin/glove tones (flesh tone or dark tactical glove)
            colors = [(140, 170, 210), (35, 35, 40), (120, 150, 190)]
            color = colors[np.random.randint(len(colors))]
            cv2.ellipse(out_img, (px + pw // 2, py + ph // 2), (pw // 2, ph // 2),
                        int(np.random.randint(-20, 20)), 0, 360, color, -1)

    elif mode == "perspective_tilt":
        # Affine shear representing high CCTV pitch
        out_boxes = list(boxes)
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        shear = np.random.uniform(0.08, 0.15)
        pts2 = np.float32([[int(w * shear), 0], [w, 0], [0, h]])
        M = cv2.getAffineTransform(pts1, pts2)
        out_img = cv2.warpAffine(out_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    else:
        out_boxes = list(boxes)

    return out_img, out_boxes


def build_hard_negatives_v3(dest_dir: Path, target_count: int = 500):
    """
    Builds Hard Negative Dataset v3 prioritizing:
    1. BACKGROUND_CLUTTER (60%): High-contrast perimeter fence mesh, crossbars, diagonal shadow lines.
    2. TOOL (25%): Elongated metallic hand tools, wrenches, power drills, metallic shafts.
    3. PHONE (15%): Rectangular dark mobile devices held in hand.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    img_dir = dest_dir / "images"
    lbl_dir = dest_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    marker = dest_dir / ".ingest_complete"
    if marker.exists() and len(list(img_dir.glob("*.jpg"))) >= target_count:
        logger.info(f"Hard Negatives v3 already present ({len(list(img_dir.glob('*.jpg')))} images).")
        return

    logger.info(f"Generating Hard Negative Dataset v3 at {dest_dir} (Target: {target_count})...")
    np.random.seed(42)

    for i in range(target_count):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        # Background base
        bg_val = np.random.randint(40, 160)
        img[:] = (bg_val, bg_val + 3, bg_val + 8)
        # Texture noise
        noise = np.random.randint(-12, 12, (640, 640, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cx, cy = np.random.randint(150, 480), np.random.randint(150, 480)

        # 60% Background Clutter (Fence wires, posts, shadows)
        if i % 10 < 6:
            # Diagonal chain-link or security fence grid
            for step in range(-150, 200, 35):
                cv2.line(img, (cx + step - 100, cy - 120), (cx + step + 100, cy + 120), (30, 30, 30), 2)
                cv2.line(img, (cx + step + 100, cy - 120), (cx + step - 100, cy + 120), (30, 30, 30), 2)
            # Vertical post
            cv2.line(img, (cx, cy - 180), (cx, cy + 180), (20, 20, 20), np.random.randint(6, 14))

        # 25% Metallic Tools (wrenches, power drills)
        elif i % 10 < 8:
            tw, th = np.random.randint(20, 38), np.random.randint(85, 140)
            cv2.rectangle(img, (cx, cy), (cx + tw, cy + th), (190, 195, 200), -1)
            cv2.circle(img, (cx + tw // 2, cy + 15), 14, (120, 125, 130), -1)
            # Drill grip
            cv2.rectangle(img, (cx - 25, cy + th // 2), (cx, cy + th // 2 + 25), (45, 45, 50), -1)

        # 15% Handheld Phones
        else:
            pw, ph = np.random.randint(30, 45), np.random.randint(65, 90)
            cv2.rectangle(img, (cx, cy), (cx + pw, cy + ph), (25, 25, 25), -1)
            cv2.rectangle(img, (cx + 3, cy + 5), (cx + pw - 3, cy + ph - 7), (55, 60, 65), -1)
            # Hand holding phone
            cv2.ellipse(img, (cx + pw, cy + ph // 2), (18, 30), 0, 0, 360, (140, 170, 210), -1)

        out_img_name = f"neg_v3_{i:04d}.jpg"
        cv2.imwrite(str(img_dir / out_img_name), img)
        (lbl_dir / f"{out_img_name.replace('.jpg', '.txt')}").write_text("", encoding="utf-8")

    marker.write_text(f"dataset: hard_negatives_v3\ncount: {target_count}\n", encoding="utf-8")
    logger.info(f"Built {target_count} Hard Negatives v3.")


def run_v2_1_normalization():
    logger.info("Initializing Dataset v2.1 Normalization...")

    # Build Hard Negatives v3
    hn3_dir = RAW_ROOT / "hard_negatives_v3"
    build_hard_negatives_v3(hn3_dir, target_count=500)

    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Harvest existing training positives from v2
    v2_train_imgs = list((V2_ROOT / "images" / "train").glob("*.jpg"))
    positives = []
    for p in v2_train_imgs:
        lbl_p = V2_ROOT / "labels" / "train" / f"{p.stem}.txt"
        if lbl_p.exists() and lbl_p.stat().st_size > 0:
            lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
            boxes = []
            for l in lines:
                parts = l.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    boxes.append([float(v) for v in parts[1:5]])
            if boxes:
                positives.append({"img_path": p, "boxes": boxes})

    logger.info(f"Harvested {len(positives)} base training positives.")

    # 2. Enrich Positives (small scale, occlusion, perspective tilt)
    enriched_positives = []
    np.random.seed(42)

    for item in positives:
        im = cv2.imread(str(item["img_path"]))
        if im is None: continue

        # Keep original
        enriched_positives.append({"img": im, "boxes": item["boxes"]})

        # Generate targeted enrichments
        for mode in ["small_scale", "occlusion", "perspective_tilt"]:
            enr_im, enr_boxes = apply_target_fn_enrichment(im, item["boxes"], mode)
            if enr_boxes:
                enriched_positives.append({"img": enr_im, "boxes": enr_boxes})

    logger.info(f"Enriched positives total: {len(enriched_positives)} samples.")

    # 3. Harvest Hard Negatives v3
    hn3_imgs = list((hn3_dir / "images").glob("*.jpg"))
    np.random.shuffle(hn3_imgs)

    # Balanced split: 80% train, 20% val
    n_pos = len(enriched_positives)
    n_neg = min(len(hn3_imgs), int(n_pos * 0.85)) # Maintain ~54% pos, 46% neg

    selected_negatives = hn3_imgs[:n_neg]
    np.random.shuffle(enriched_positives)

    pos_split = int(n_pos * 0.80)
    neg_split = int(n_neg * 0.80)

    train_pos = enriched_positives[:pos_split]
    val_pos = enriched_positives[pos_split:]
    train_neg = selected_negatives[:neg_split]
    val_neg = selected_negatives[neg_split:]

    # Write train split
    idx = 0
    for item in train_pos:
        name = f"secitem_v2_1_train_{idx:05d}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / "images" / "train" / name), item["img"])
        with open(OUTPUT_DIR / "labels" / "train" / f"{name.replace('.jpg', '.txt')}", "w", encoding="utf-8") as lf:
            for cx, cy, bw, bh in item["boxes"]:
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        idx += 1

    for neg_p in train_neg:
        name = f"secitem_v2_1_train_{idx:05d}_neg.jpg"
        shutil.copy2(neg_p, OUTPUT_DIR / "images" / "train" / name)
        (OUTPUT_DIR / "labels" / "train" / f"{name.replace('.jpg', '.txt')}").write_text("", encoding="utf-8")
        idx += 1

    logger.info(f"Train split written: {idx} images ({len(train_pos)} pos, {len(train_neg)} neg).")

    # Write val split
    v_idx = 0
    for item in val_pos:
        name = f"secitem_v2_1_val_{v_idx:05d}.jpg"
        cv2.imwrite(str(OUTPUT_DIR / "images" / "val" / name), item["img"])
        with open(OUTPUT_DIR / "labels" / "val" / f"{name.replace('.jpg', '.txt')}", "w", encoding="utf-8") as lf:
            for cx, cy, bw, bh in item["boxes"]:
                lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        v_idx += 1

    for neg_p in val_neg:
        name = f"secitem_v2_1_val_{v_idx:05d}_neg.jpg"
        shutil.copy2(neg_p, OUTPUT_DIR / "images" / "val" / name)
        (OUTPUT_DIR / "labels" / "val" / f"{name.replace('.jpg', '.txt')}").write_text("", encoding="utf-8")
        v_idx += 1

    logger.info(f"Val split written: {v_idx} images ({len(val_pos)} pos, {len(val_neg)} neg).")

    # Write dataset.yaml pointing to the frozen test benchmark IBVAP-GT-ITEM-v2.0
    frozen_test_path = (V2_ROOT / "images" / "test").resolve().as_posix()
    dataset_yaml_content = f"""# ==============================================================================
# IBVAP — Security Item Model v2.1 Dataset Configuration
# Test split strictly points to frozen benchmark: IBVAP-GT-ITEM-v2.0
# ==============================================================================
path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val
test: {frozen_test_path}

names:
  0: firearm

nc: 1
"""
    with open(OUTPUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(dataset_yaml_content)

    logger.info(f"Dataset v2.1 generated at {OUTPUT_DIR}. Frozen benchmark preserved at {frozen_test_path}.")


if __name__ == "__main__":
    run_v2_1_normalization()
