#!/usr/bin/env python3
"""
TRINETRA — P-01: NVIDIA TensorRT Engine Exporter & Jetson Orin Accelerator
Converts PyTorch (.pt) and ONNX (.onnx) detection/recognition models to
optimized TensorRT (.engine) runtime models with FP16 and INT8 precision calibration.
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TensorRTExporter")


def export_ultralytics_to_tensorrt(
    model_path: str,
    imgsz: int = 640,
    half: bool = True,
    int8: bool = False,
    device: int = 0,
    workspace_gb: int = 4,
) -> str:
    """Export YOLO11/YOLO26 model to TensorRT engine using Ultralytics engine builder."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics package is required. Install with: pip install ultralytics")
        sys.exit(1)

    logger.info("Loading base model: %s", model_path)
    model = YOLO(model_path)

    logger.info(
        "Exporting to TensorRT format (imgsz=%d, FP16=%s, INT8=%s, device=%d, workspace=%d GB)...",
        imgsz, half, int8, device, workspace_gb,
    )
    t0 = time.time()
    exported_engine = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        int8=int8,
        device=device,
        workspace=workspace_gb,
        simplify=True,
    )
    elapsed = time.time() - t0
    logger.info("TensorRT export completed in %.1f seconds: %s", elapsed, exported_engine)
    return str(exported_engine)


def benchmark_engine(engine_path: str, imgsz: int = 640, warmup: int = 10, iterations: int = 50):
    """Benchmark inference latency on target hardware (e.g. Jetson Orin)."""
    import numpy as np
    try:
        from ultralytics import YOLO
        model = YOLO(engine_path)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        logger.info("Warming up TensorRT engine with %d iterations...", warmup)
        for _ in range(warmup):
            model.predict(dummy, imgsz=imgsz, verbose=False)

        logger.info("Benchmarking latency over %d iterations...", iterations)
        timings = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            model.predict(dummy, imgsz=imgsz, verbose=False)
            timings.append((time.perf_counter() - t0) * 1000.0)

        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        fps = 1000.0 / p50
        logger.info("=== TensorRT Engine Benchmark Results ===")
        logger.info("Engine: %s", engine_path)
        logger.info("Median Latency (P50): %.2f ms (%.1f FPS)", p50, fps)
        logger.info("P95 Latency:          %.2f ms", p95)
        logger.info("P99 Latency:          %.2f ms", p99)
    except Exception as e:
        logger.warning("Benchmarking skipped (requires CUDA/TensorRT runtime): %s", e)


def main():
    parser = argparse.ArgumentParser(description="TRINETRA TensorRT Model Exporter")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to input .pt or .onnx model")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", default=False, help="Enable INT8 quantization")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace size in GB")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark after export")
    args = parser.parse_args()

    engine_path = export_ultralytics_to_tensorrt(
        model_path=args.model,
        imgsz=args.imgsz,
        half=args.fp16,
        int8=args.int8,
        device=args.device,
        workspace_gb=args.workspace,
    )

    if args.benchmark and engine_path and os.path.exists(engine_path):
        benchmark_engine(engine_path, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
