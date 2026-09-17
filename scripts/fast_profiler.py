import os
import sys
import time
import psutil
import torch
import cv2
import numpy as np

# Unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def main():
    print("=" * 60, flush=True)
    print("IBVAP STEP-BY-STEP STAGE PROFILER", flush=True)
    print("=" * 60, flush=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB", flush=True)
        print("AI DEVICE: CUDA", flush=True)
        print("MODEL DEVICE: cuda:0", flush=True)
        print("INPUT DEVICE: cuda:0", flush=True)
    
    # 1. Model Loading
    model_path = os.path.join(ROOT_DIR, "models", "current", "ibvap_detector.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(ROOT_DIR, "yolo11n.pt")
    
    print(f"\n1. Loading model: {os.path.basename(model_path)}...", flush=True)
    from ultralytics import YOLO
    t0 = time.perf_counter()
    model = YOLO(model_path)
    model.to(device)
    if device.startswith("cuda") and hasattr(model, 'model') and model.model is not None:
        model.model.half()
    print(f"   Model loaded in {(time.perf_counter() - t0)*1000:.1f} ms", flush=True)

    # 2. Benchmark Inference Latency
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    print("\n2. Measuring Inference Latency across Resolutions:", flush=True)
    # Warmup
    for _ in range(3):
        _ = model.predict(source=dummy_frame, imgsz=640, device=device, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    for res in [320, 480, 640]:
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            _ = model.predict(source=dummy_frame, imgsz=res, device=device, verbose=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        mean_t = np.mean(times)
        p95_t = np.percentile(times, 95)
        print(f"   imgsz={res}x{res}: Mean = {mean_t:.2f} ms | P95 = {p95_t:.2f} ms | Throughput = {1000/mean_t:.1f} FPS", flush=True)

    # 3. Video Capture and Read Timing
    print("\n3. Measuring Video Capture & Decode Timing:", flush=True)
    video_file = os.path.join(ROOT_DIR, "data", "videos", "CAM-001.mp4")
    if os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        read_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            read_times.append((time.perf_counter() - t0) * 1000)
        cap.release()
        print(f"   Frame Capture/Decode (1080p MP4): {np.mean(read_times):.2f} ms per frame", flush=True)

    # 4. Tracking and Zone Timing
    print("\n4. Measuring Tracking & Zone Rules Timing:", flush=True)
    from backend.app.services.border_rules_service import border_rules_service
    dummy_dets = [
        {"class": "person", "confidence": 0.88, "box": [400, 300, 500, 600], "track_id": f"CAM-001:P-{i:03d}"}
        for i in range(5)
    ]
    zone_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        border_rules_service.process_detections("CAM-001", dummy_dets)
        zone_times.append((time.perf_counter() - t0) * 1000)
    print(f"   Zone evaluation (5 tracks): {np.mean(zone_times):.2f} ms", flush=True)

    # 5. Drawing & Annotation Timing
    print("\n5. Measuring Visual Drawing & Annotation Timing:", flush=True)
    draw_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        annotated = dummy_frame.copy()
        for det in dummy_dets:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        draw_times.append((time.perf_counter() - t0) * 1000)
    print(f"   Drawing annotations (1080p frame): {np.mean(draw_times):.2f} ms", flush=True)

    # 6. JPEG Encoding Overhead
    print("\n6. Measuring JPEG Encoding Overhead:", flush=True)
    t_1080 = []
    for _ in range(10):
        t0 = time.perf_counter()
        _, _ = cv2.imencode('.jpg', dummy_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        t_1080.append((time.perf_counter() - t0) * 1000)
    print(f"   cv2.imencode (1080p full frame): {np.mean(t_1080):.2f} ms", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("PROFILING COMPLETE", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
