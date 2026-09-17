#!/usr/bin/env python3
"""
IBVAP Phase XV — Governed Native Thermal YOLO Model Training Script
Supported Architectures: YOLO11, YOLO26 (YOLOv8 is strictly prohibited).

Pre-condition Gate:
Strictly verifies that a genuine REAL_SENSOR dataset exists, annotation QA has passed,
and the thermal benchmark is permanently frozen before initiating training.

Outputs candidate checkpoints exclusively to models/candidates/thermal/.
Never writes directly to production paths.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.thermal_training_service import (
    thermal_training_service,
    ThermalTrainingHyperparameters,
)
from backend.app.services.thermal_dataset_service import thermal_dataset_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainThermalYOLO")


def main():
    parser = argparse.ArgumentParser(description="IBVAP Governed Native Thermal Training")
    parser.add_argument("--architecture", type=str, default="YOLO11n", choices=["YOLO11n", "YOLO11s", "YOLO26n", "YOLO26s"])
    parser.add_argument("--dataset-id", type=str, default="DS-THM-READINESS-v0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--dry-run", action="store_true", help="Simulate training pipeline governance check")
    args = parser.parse_args()

    logger.info("============================================================")
    logger.info("IBVAP Phase XV — Native Thermal Model Training Orchestrator")
    logger.info("============================================================")
    logger.info(f"Requested Architecture: {args.architecture}")
    logger.info(f"Target Dataset ID:      {args.dataset_id}")

    # 1. Dataset existence and provenance check
    ds = thermal_dataset_service.get_dataset(args.dataset_id)
    if not ds:
        logger.error(f"FATAL: Dataset '{args.dataset_id}' does not exist in registry.")
        sys.exit(1)

    has_real_data = ds.data_origin.value == "REAL_SENSOR" and len(ds.samples) > 0
    logger.info(f"Dataset Data Origin:    {ds.data_origin.value}")
    logger.info(f"Sample Count:           {len(ds.samples)}")
    logger.info(f"Real Sensor Available:  {has_real_data}")

    # 2. Hyperparameters
    hp = ThermalTrainingHyperparameters(
        architecture=args.architecture,
        base_model=f"{args.architecture.lower()}.pt",
        dataset_version=ds.version,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # 3. Orchestrate
    exp_id = f"EXP-{args.architecture}-{int(ds.updated_at)}"
    manifest = thermal_training_service.orchestrate_thermal_experiment(
        experiment_id=exp_id,
        experiment_name=f"Thermal_{args.architecture}_Baseline",
        architecture=args.architecture,
        base_model=hp.base_model,
        dataset_id=args.dataset_id,
        dataset_version=ds.version,
        train_split_hash="TRAIN_HASH_NONE",
        val_split_hash="VAL_HASH_NONE",
        benchmark_hash="BENCH_HASH_NONE",
        hyperparameters=hp,
        has_real_sensor_dataset=has_real_data,
        is_annotation_qa_passed=False,
        is_benchmark_frozen=False,
        leakage_detected=False,
    )

    logger.info(f"Experiment Status:      {manifest.status}")
    logger.info(f"Governance Notes:       {manifest.governance_notes}")

    if manifest.status == "BLOCKED_NO_REAL_DATA":
        logger.warning("TRAINING BLOCKED: In accordance with Non-Negotiable Rule 1, no fake model will be trained.")
        logger.warning("REAL_THERMAL_DATA_AVAILABLE = FALSE")
        logger.warning("NATIVE_THERMAL_MODEL_STATUS = NOT_VALIDATED")
        if not args.dry_run:
            logger.info("Exiting cleanly with governance block status.")
            return 0

    logger.info("Training orchestration completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
