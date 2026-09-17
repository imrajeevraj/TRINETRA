# IBVAP — Benchmark Methodology & Evaluation Framework

## 1. Intentional Repository Exclusion
In compliance with repository data policies and copyright standards:
- **Raw Benchmark Images:** `benchmark/images/` and `benchmark/rfdetr_dataset/` are **excluded from Git**.
- **Generated Benchmark Artifacts:** Raw bounding box crops, model inference dumps, and intermediate logs (`benchmark/results/`) are excluded.
- **Tracked Assets:** Standardized SHA-256 frame manifests (`benchmark/checksums/`), ground truth annotation labels (`benchmark/labels/`), and evaluation harnesses are tracked in version control.

---

## 2. Ground Truth Dataset & Versioning

- **Dataset Freeze Tag:** `IBVAP-GT-v2.0` (Documented in `docs/benchmark/IBVAP-GT-v2.0_FREEZE.md`)
- **Underlying Data Source:** Real-world aerial and perimeter CCTV surveillance footage derived from the **VIRAT Video Dataset** (Sequences S_000001 through S_000004).
- **Target Classes:**
  - `0: person` — Infiltrators, border guards, civilians
  - `1: vehicle` — Heavy motor vehicles, transport trucks, patrol jeeps
  - `2: animal` — Stray wildlife, grazing herds (evaluated to benchmark false-positive rejection)

---

## 3. Annotation & Validation Workflow

1. **Deterministic Frame Extraction:**
   Frames are sampled at regular temporal intervals across video sequences:
   ```bash
   python scripts/extract_benchmark_frames.py --interval 30 --output benchmark/images/
   ```
2. **Ground Truth Validation:**
   Annotations are verified against bounding box constraints:
   ```bash
   python scripts/validate_ground_truth_annotations.py --labels benchmark/labels/ --manifest benchmark/checksums/labels.sha256
   ```
3. **Dataset Conversion for Object Detectors (RF-DETR / YOLO):**
   ```bash
   python scripts/convert_dataset_for_rfdetr.py --input benchmark/labels/ --output benchmark/rfdetr_dataset/
   ```

---

## 4. Evaluation Metrics & Execution

IBVAP benchmarks model accuracy, real-time edge latency, and false alarm rejection rates:

- **Accuracy:** Mean Average Precision at IoU 0.50 ($mAP_{50}$) and IoU 0.50:0.95 ($mAP_{50-95}$)
- **Inference Latency:** Target $\le 45\text{ms}$ per 1080p frame on edge GPU (RTX 4060 / Jetson Orin)
- **False Incursion Rate:** Percentage of non-threat tracks triggering high-priority border alarms

To execute the standardized evaluation harness:
```bash
python scripts/evaluate_detector.py --model yolo11n.pt --labels benchmark/labels/ --report docs/benchmark/EVALUATION_REPORT.md
```

Full benchmark analysis and evaluation reports are published in `docs/benchmark/RFDETR_EVALUATION_REPORT.md`.
