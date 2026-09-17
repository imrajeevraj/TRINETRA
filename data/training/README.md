# IBVAP — Training Workspace Guidelines
# TRINETRA — Training Workspace Guidelines

## 1. Scope & Isolation
The `data/training/` directory is an **isolated training and evaluation workspace**. 
Assets in this directory are **strictly excluded from version control and production runtime**:

```
data/training/
├── README.md               # (Tracked in Git - architecture guidance)
├── raw/                    # Raw external datasets downloaded from Kaggle/mirrors
├── processed/              # Formatted YOLO annotations and converted label files
├── datasets/               # Structured train/val/test splits (e.g. IBVAP-TRAIN-v1.0)
├── runs/                   # Training run outputs, weights, logs, and confusion matrices
└── downloads/              # Temporary download archives and zip files
```

---

## 2. Benchmark Separation & Anti-Leakage Rules

> [!CAUTION]
> **Strict Benchmark Protection Policy:**
> - The frozen benchmark test dataset (`benchmark/labels/test/` and `benchmark/images/test/` - `IBVAP-GT-v1.0`) must **NEVER** be copied, converted, or used within this training workspace.
> - Hyperparameter tuning and model checkpoints must never touch the frozen benchmark.
> - Only after a training run completes is the candidate model evaluated externally against the frozen benchmark.

---

## 3. Training Workflow

1. **Configure Datasets:** Declare desired datasets in `configs/kaggle_datasets.yaml`.
2. **Download Datasets:** Run `python scripts/kaggle_download.py`.
3. **Inspect & Validate:** Run `python scripts/inspect_kaggle_dataset.py --dataset data/training/raw/<name>`.
4. **Normalize Classes:** Apply `configs/class_mapping.yaml` to ensure `0: person` and `1: vehicle`.
5. **Train Detector:** Run `python scripts/train_ibvap.py --config configs/training.yaml`.
6. **Evaluate Candidate:** Run `python scripts/evaluate_ibvap_model.py --model <run_best.pt>`.
7. **Register Active Model:** Update `models/model_registry.yaml` and promote to `models/current/ibvap_detector.pt` only if recall, false alarms, and latency criteria are met.
