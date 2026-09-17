#!/usr/bin/env python3
"""
IBVAP — Export & Edge Deployment Benchmark for YOLO26 vs YOLO11 Models
Exports candidates and baselines to ONNX format, measures file sizes,
and benchmarks inference latency across formats.
Outputs:
  data/reports/yolo26/export_benchmark.json
  data/reports/yolo26/export_benchmark.md
"""

import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportBenchmark")

MODELS = {
    "ground_yolo11": {
        "path": "models/production/ground/ibvap_ground_v2_production.pt",
        "imgsz": 768,
        "type": "baseline",
    },
    "ground_yolo26": {
        "path": "models/candidates/yolo26/ground/best.pt",
        "imgsz": 768,
        "type": "candidate",
    },
    "airborne_yolo11": {
        "path": "models/production/airborne/ibvap_airborne_v1_production.pt",
        "imgsz": 640,
        "type": "baseline",
    },
    "airborne_yolo26": {
        "path": "models/candidates/yolo26/airborne/best.pt",
        "imgsz": 640,
        "type": "candidate",
    },
    "security_item_yolo11": {
        "path": "models/production/security_item/ibvap_security_item_v2_1_production.pt",
        "imgsz": 640,
        "type": "baseline",
    },
    "security_item_yolo26": {
        "path": "models/candidates/yolo26/security_item/best.pt",
        "imgsz": 640,
        "type": "candidate",
    },
}


def measure_pt_latency(model, imgsz: int, device: str = "cuda:0", runs: int = 100) -> dict:
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(20):
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(1000.0 / float(np.percentile(times, 50)), 1),
    }


def export_model(model_key: str, cfg: dict, export_dir: Path) -> dict:
    pt_path = Path(cfg["path"])
    if not pt_path.exists():
        logger.warning(f"Skipping {model_key}: checkpoint {pt_path} does not exist.")
        return {"status": "SKIPPED_NOT_FOUND"}

    pt_size_mb = round(pt_path.stat().st_size / (1024 * 1024), 2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(pt_path))
    model.to(device)

    # Benchmark PyTorch FP16/FP32
    pt_perf = measure_pt_latency(model, cfg["imgsz"], device=device)

    # Export to ONNX
    onnx_file = None
    onnx_size_mb = None
    onnx_status = "PENDING"
    try:
        logger.info(f"Exporting {model_key} to ONNX (imgsz={cfg['imgsz']})...")
        exported_path = model.export(format="onnx", imgsz=cfg["imgsz"], dynamic=False, opset=12, verbose=False)
        if exported_path and Path(exported_path).exists():
            onnx_file = str(Path(exported_path).resolve())
            onnx_size_mb = round(Path(exported_path).stat().st_size / (1024 * 1024), 2)
            onnx_status = "SUCCESS"
    except Exception as e:
        logger.warning(f"ONNX export for {model_key} failed: {e}")
        onnx_status = f"FAILED: {e}"

    return {
        "status": "COMPLETED",
        "pt_path": str(pt_path),
        "pt_size_mb": pt_size_mb,
        "pt_latency": pt_perf,
        "onnx_path": onnx_file,
        "onnx_size_mb": onnx_size_mb,
        "onnx_status": onnx_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Export and Benchmark YOLO26 & YOLO11 Models")
    parser.add_argument("--out-dir", default="data/reports/yolo26")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, cfg in MODELS.items():
        logger.info(f"--- Processing {key} ---")
        results[key] = export_model(key, cfg, out_dir)

    json_path = out_dir / "export_benchmark.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# IBVAP Model Export & Edge Deployment Benchmark",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "| Model Identifier | Type | PT Size (MB) | ONNX Size (MB) | P50 (ms) | FPS | ONNX Status |",
        "|---|---|---|---|---|---|---|",
    ]

    for key, res in results.items():
        if res.get("status") == "COMPLETED":
            pt_sz = res.get("pt_size_mb", "N/A")
            onnx_sz = res.get("onnx_size_mb", "N/A")
            p50 = res.get("pt_latency", {}).get("p50_ms", "N/A")
            fps = res.get("pt_latency", {}).get("fps", "N/A")
            status = res.get("onnx_status", "N/A")
            mtype = MODELS[key]["type"]
            md_lines.append(f"| `{key}` | {mtype} | {pt_sz} | {onnx_sz} | {p50} | {fps} | {status} |")
        else:
            md_lines.append(f"| `{key}` | {MODELS[key]['type']} | N/A | N/A | N/A | N/A | {res.get('status')} |")

    md_path = out_dir / "export_benchmark.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Saved export benchmark results to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
