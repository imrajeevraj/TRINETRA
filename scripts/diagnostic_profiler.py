import os
import sys
import time
import psutil
import torch
import cv2
import numpy as np

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_diagnostics():
    print("=" * 60)
    print("IBVAP REAL-TIME PIPELINE COMPREHENSIVE PROFILER")
    print("=" * 60)
    
    # 1. Environment & Hardware Diagnostics
    print("\n[1. ENVIRONMENT & HARDWARE]")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"GPU Device: {device_name} (Compute: {capability[0]}.{capability[1]})")
        print(f"VRAM Total: {vram_total:.1f} MB")
        print("AI DEVICE: CUDA")
        print("MODEL DEVICE: cuda:0")
        print("INPUT DEVICE: cuda:0")
    else:
        print("AI DEVICE: CPU")

    sys_mem = psutil.virtual_memory()
    proc = psutil.Process()
    print(f"System RAM: {sys_mem.used / (1024**2):.1f} / {sys_mem.total / (1024**2):.1f} MB ({sys_mem.percent}%)")
    print(f"Process RSS: {proc.memory_info().rss / (1024**2):.1f} MB")

    # 2. Model Benchmarks
    print("\n[2. MODEL COMPARISON (ISOLATED CUDA INFERENCE)]")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    
    models_to_test = [
        ("yolo11n.pt", "YOLO11 Nano Baseline"),
        ("models/current/ibvap_detector.pt", "IBVAP Ground Model v2.0 (YOLO11)"),
        ("models/candidates/yolo26/ground/best.pt", "IBVAP Ground Model v3.0 Candidate (YOLO26)"),
    ]
    
    dummy_1080p = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    for filename, name in models_to_test:
        path = os.path.join(ROOT_DIR, filename)
        if not os.path.exists(path):
            print(f"  {name} ({filename}): Not found")
            continue
        try:
            m = YOLO(path)
            m.to(device)
            print(f"\n  --- {name} ({filename}) on {device} ---")
            
            # Warmup
            for _ in range(5):
                _ = m(dummy_1080p, imgsz=640, device=device, verbose=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                
            for res in [320, 480, 640]:
                latencies = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    _ = m(dummy_1080p, imgsz=res, device=device, verbose=False)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)
                mean_l = np.mean(latencies)
                p95_l = np.percentile(latencies, 95)
                fps = 1000.0 / mean_l
                print(f"    imgsz={res}x{res} -> Mean Latency: {mean_l:.2f} ms | P95: {p95_l:.2f} ms | Max FPS: {fps:.1f}")
        except Exception as e:
            print(f"    Error benchmarking {name}: {e}")

    # 3. Pipeline Stages Breakdown
    print("\n[3. PIPELINE STAGES BREAKDOWN (1080p Frame)]")
    
    # 3a. Capture / Decode simulation from actual video file
    video_path = os.path.join(ROOT_DIR, "data", "videos", "CAM-001.mp4")
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        times_read = []
        for _ in range(30):
            t0 = time.perf_counter()
            ret, f = cap.read()
            times_read.append((time.perf_counter() - t0) * 1000)
        cap.release()
        print(f"  VideoCapture.read (1080p MP4): {np.mean(times_read):.2f} ms")
    
    # 3b. Resize
    times_resize_320 = []
    times_resize_640 = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = cv2.resize(dummy_1080p, (320, 320))
        times_resize_320.append((time.perf_counter() - t0) * 1000)
        
        t0 = time.perf_counter()
        _ = cv2.resize(dummy_1080p, (640, 640))
        times_resize_640.append((time.perf_counter() - t0) * 1000)
    print(f"  cv2.resize (1080p -> 320x320): {np.mean(times_resize_320):.2f} ms")
    print(f"  cv2.resize (1080p -> 640x640): {np.mean(times_resize_640):.2f} ms")

    # 3c. JPEG encode (CPU vs downscaled JPEG)
    times_jpeg_1080 = []
    times_jpeg_720 = []
    for _ in range(20):
        t0 = time.perf_counter()
        _, _ = cv2.imencode('.jpg', dummy_1080p, [cv2.IMWRITE_JPEG_QUALITY, 50])
        times_jpeg_1080.append((time.perf_counter() - t0) * 1000)
        
        small_f = cv2.resize(dummy_1080p, (960, 540))
        t0 = time.perf_counter()
        _, _ = cv2.imencode('.jpg', small_f, [cv2.IMWRITE_JPEG_QUALITY, 50])
        times_jpeg_720.append((time.perf_counter() - t0) * 1000)
    print(f"  cv2.imencode 1080p (CPU): {np.mean(times_jpeg_1080):.2f} ms")
    print(f"  cv2.imencode 540p (CPU):  {np.mean(times_jpeg_720):.2f} ms (4x faster)")

    # 3d. Zone check
    from backend.app.services.border_rules_service import border_rules_service
    times_zone = []
    dummy_dets = [
        {"class": "person", "confidence": 0.85, "box": [500, 400, 600, 700], "track_id": f"P-{i:03d}"}
        for i in range(10)
    ]
    for _ in range(50):
        t0 = time.perf_counter()
        border_rules_service.process_detections("CAM-001", dummy_dets)
        times_zone.append((time.perf_counter() - t0) * 1000)
    print(f"  Zone & Rules Processing (10 tracks): {np.mean(times_zone):.2f} ms")

    print("\n" + "=" * 60)
    print("PROFILING RUN FINISHED")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
