import os
import sys
import time
import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

def test_engines():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    from ultralytics import YOLO
    
    model = YOLO("yolo11n.pt")
    model.to(device)
    
    dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=device)
    
    # 1. Standard PyTorch FP32 vs FP16
    print("\n--- 1. PyTorch Eager Mode ---", flush=True)
    net = model.model.eval()
    
    # FP32
    with torch.no_grad():
        for _ in range(10): _ = net(dummy_input)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = net(dummy_input)
            torch.cuda.synchronize()
        fp32_t = (time.perf_counter() - t0) / 30 * 1000
    print(f"PyTorch FP32 (640x640): {fp32_t:.2f} ms ({1000/fp32_t:.1f} FPS)", flush=True)

    # FP16
    net_half = net.half()
    dummy_half = dummy_input.half()
    with torch.no_grad():
        for _ in range(10): _ = net_half(dummy_half)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = net_half(dummy_half)
            torch.cuda.synchronize()
        fp16_t = (time.perf_counter() - t0) / 30 * 1000
    print(f"PyTorch FP16 (640x640): {fp16_t:.2f} ms ({1000/fp16_t:.1f} FPS)", flush=True)

    # FP16 @ 320x320
    dummy_half_320 = torch.zeros((1, 3, 320, 320), dtype=torch.half, device=device)
    with torch.no_grad():
        for _ in range(10): _ = net_half(dummy_half_320)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            _ = net_half(dummy_half_320)
            torch.cuda.synchronize()
        fp16_320_t = (time.perf_counter() - t0) / 30 * 1000
    print(f"PyTorch FP16 (320x320): {fp16_320_t:.2f} ms ({1000/fp16_320_t:.1f} FPS)", flush=True)

    # 2. Check ONNX Runtime with CUDA / TensorRT
    print("\n--- 2. Checking ONNX Runtime Providers ---", flush=True)
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime Version: {ort.__version__}", flush=True)
        print(f"Available Providers: {ort.get_available_providers()}", flush=True)
    except ImportError as e:
        print(f"ONNX Runtime not installed: {e}", flush=True)

if __name__ == "__main__":
    test_engines()
