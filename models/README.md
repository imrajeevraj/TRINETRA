# IBVAP — AI Models & Neural Network Weights Specification
# TRINETRA — AI Models & Neural Network Weights Specification

## 1. Intentional Repository Exclusion
Pretrained deep learning weights (`*.pt`, `*.pth`, `*.onnx`, `*.engine`, `*.safetensors`) are **strictly excluded from this GitHub repository**. 

Weights are binary artifacts that should be downloaded or built locally from authoritative checkpoints to ensure supply-chain integrity.

---

## 2. Supported Architectures in IBVAP
## 2. Supported Architectures in TRINETRA

IBVAP strictly supports two YOLO model families:
TRINETRA strictly supports two YOLO model families:
1. **YOLO11**: Active Production baseline across Ground, Airborne, and Security Item perception domains.
2. **YOLO26**: Candidate architecture for next-generation edge evaluation.

All obsolete model architectures (e.g. YOLOv8) have been retired to `models/archive/obsolete_models/`.

| Model Role | Checkpoint Name | Architecture | Purpose / Taxonomy | Status | Authoritative Path | SHA-256 Verified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ground Production** | `ibvap_ground_v2_production.pt` | Ultralytics YOLO11n (768px) | Person & Vehicle perimeter tracking | **Production Active** | `models/production/ground/` | `7DBF3602...` |
| **Airborne Production** | `ibvap_airborne_v1_production.pt` | Ultralytics YOLO11n (640px) | Drone & Aircraft airspace surveillance | **Production Active** | `models/production/airborne/` | `E1009633...` |
| **Security Item Production** | `ibvap_security_item_v2_1_production.pt` | Ultralytics YOLO11n (640px) | Firearm threat identification | **Production Active** | `models/production/security_item/` | `72464C77...` |
| **Primary Runtime Pointer** | `ibvap_detector.pt` | Ultralytics YOLO11n | Default detector symlink/pointer | **Production Active** | `models/current/ibvap_detector.pt` | `7DBF3602...` |
| **Base Pretrain Checkpoint** | `yolo11n.pt` | Ultralytics YOLO11n (COCO) | Fine-tuning & transfer learning base | **Active Base** | `./yolo11n.pt` | `0EBBC80D...` |
| **YOLO26 Candidate Suite** | `best.pt` (Ground/Air/Item) | Ultralytics YOLO26n | Experimental next-gen evaluation | **Candidate** | `models/candidates/yolo26/` | Locked in registry |
| **Biometric Face Re-ID** | `buffalo_l` | InsightFace (ArcFace ResNet50) | 512-dim facial embedding vector extraction | **Production Biometric** | `app/data/insightface/models/buffalo_l/` | Manifest locked |

---

## 3. Automated Download & Acquisition

### 3.1 Primary YOLO Detectors
Ultralytics automatically downloads official YOLO checkpoints upon first invocation:
```python
from ultralytics import YOLO

# Automatically fetches official Ultralytics weights if not present locally
model = YOLO("yolo11n.pt")
```

To export to ONNX for edge acceleration:
```bash
python backend/scripts/export_onnx.py --model yolo11n.pt --format onnx
```

### 3.2 InsightFace Biometric Model Pack (`buffalo_l`)
InsightFace models are fetched via `insightface.app.FaceAnalysis`:
```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", root="app/data/insightface")
app.prepare(ctx_id=0, det_size=(640, 640))
```
- **License:** InsightFace models are subject to the InsightFace research license (non-commercial research and testing).
- **Official Source:** [InsightFace Model Zoo](https://github.com/deepinsight/insightface)

---

## 4. Cryptographic SHA-256 Verification Procedure

To safeguard against corrupted or tampered model weights in production border deployments, IBVAP validates model checksums against `.env` and `models/model_registry.yaml`:
To safeguard against corrupted or tampered model weights in production border deployments, TRINETRA validates model checksums against `.env` and `models/model_registry.yaml`:

```bash
# In .env:
MODEL_PATH=models/current/ibvap_detector.pt
MODEL_SHA256=7DBF36027768194F61C6EBBC000E1421374624558168F8A9896F727B296277B0
```

To compute the SHA-256 hash of a local model weight file:
```powershell
# Windows PowerShell:
Get-FileHash -Algorithm SHA256 models\current\ibvap_detector.pt

# Linux / macOS:
sha256sum models/current/ibvap_detector.pt
```

The startup routine validates that the calculated hash matches the expected hash before allocating GPU VRAM.

---

## 5. YOLO26 Evaluation Outcome & Production Architecture

In September 2026, IBVAP evaluated candidate **Ultralytics YOLO26** models against the validated **YOLO11** production checkpoints across Ground, Airborne, and Security Item domains under strict promotion gates.
In September 2026, TRINETRA evaluated candidate **Ultralytics YOLO26** models against the validated **YOLO11** production checkpoints across Ground, Airborne, and Security Item domains under strict promotion gates.

### Summary of Findings
- **Ground Detector (`yolo26n`):** 0.6242 mAP50 vs 0.6599 (YOLO11n). Did not beat baseline. **Decision: RETAIN YOLO11n**.
- **Airborne Detector (`yolo26n`):** 0.7846 mAP50 (+0.0006 gain) but 50.7ms P50 latency (5.4x slower than YOLO11n 9.3ms). Failed latency gate (<25ms). **Decision: RETAIN YOLO11n**.
- **Security Item Detector (`yolo26n`):** 0.3054 mAP50 vs 0.8579 (YOLO11n). Severe accuracy drop. **Decision: RETAIN YOLO11n**.
- **Security Item Governance Notice:** `REAL-FIREARM-VIDEO VALIDATION NOT AVAILABLE` remains enforced.
- **Rollback Tool:** `python scripts/yolo26/rollback.py --domain all` provides atomic restoration of production YOLO11 checkpoints. Full report in `YOLO26_IBVAP_MIGRATION_REPORT.md`.
