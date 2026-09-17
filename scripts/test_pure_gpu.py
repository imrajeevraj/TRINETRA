import os
import sys
import time
import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

def main():
    print("Testing pure PyTorch forward pass on CUDA...", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.to(device)
    
    # Extract the underlying PyTorch nn.Module
    net = model.model
    net.eval()
    net.half()
    
    print("Underlying PyTorch model extracted and converted to FP16.", flush=True)
    
    # Create input directly on GPU in FP16
    tensor_input = torch.zeros((1, 3, 640, 640), dtype=torch.half, device=device)
    
    # Warmup
    print("Warming up GPU...", flush=True)
    with torch.no_grad():
        for _ in range(10):
            _ = net(tensor_input)
    torch.cuda.synchronize()
    
    # Pure GPU Forward Pass Timing
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = net(tensor_input)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
            
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    print(f"Pure GPU Forward Pass (1x3x640x640 FP16): {mean_lat:.2f} ms | P95: {p95_lat:.2f} ms | Peak Throughput: {1000/mean_lat:.1f} FPS!", flush=True)
    
    # Test with Batch=4 (for 4 cameras simultaneously)
    tensor_batch4 = torch.zeros((4, 3, 640, 640), dtype=torch.half, device=device)
    latencies_b4 = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = net(tensor_batch4)
            torch.cuda.synchronize()
            latencies_b4.append((time.perf_counter() - t0) * 1000)
            
    mean_b4 = np.mean(latencies_b4)
    print(f"Batch=4 GPU Forward Pass (4 cams batched FP16): {mean_b4:.2f} ms total ({mean_b4/4:.2f} ms per camera!) -> {4000/mean_b4:.1f} FPS across all 4 cameras!", flush=True)

if __name__ == "__main__":
    main()
