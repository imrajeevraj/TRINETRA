"""
TRINETRA — Phase 17: Lightweight Metrics Service
Prometheus-compatible metrics exporter. No heavy observability stack.
Exposes counters, gauges, histograms for all key pipeline telemetry.
"""

import threading
import logging
from collections import defaultdict
from typing import Dict, List, Optional
import torch

logger = logging.getLogger("MetricsService")


class Counter:
    def __init__(self, name: str, help_text: str, labels: Optional[List[str]] = None):
        self.name = name
        self.help = help_text
        self._vals: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels):
        key = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        with self._lock:
            self._vals[key] += amount

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for labels, val in self._vals.items():
                if labels:
                    lines.append(f"{self.name}{{{labels}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return "\n".join(lines)


class Gauge:
    def __init__(self, name: str, help_text: str, labels: Optional[List[str]] = None):
        self.name = name
        self.help = help_text
        self._vals: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels):
        key = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        with self._lock:
            self._vals[key] = value

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for labels, val in self._vals.items():
                if labels:
                    lines.append(f"{self.name}{{{labels}}} {val}")
                else:
                    lines.append(f"{self.name} {val}")
        return "\n".join(lines)


class Histogram:
    BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, float("inf")]

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._sum = 0.0
        self._count = 0
        self._buckets: Dict[float, int] = {b: 0 for b in self.BUCKETS}
        self._lock = threading.Lock()

    def observe(self, value: float):
        with self._lock:
            self._sum += value
            self._count += 1
            for b in self.BUCKETS:
                if value <= b:
                    self._buckets[b] += 1

    def collect(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for b, cnt in self._buckets.items():
                le = "+Inf" if b == float("inf") else str(b)
                lines.append(f'{self.name}_bucket{{le="{le}"}} {cnt}')
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


class IBVAPMetrics:
    """
    Central metrics registry for the IBVAP pipeline.
    Thread-safe. Prometheus text format export via to_prometheus_text().
    """

    def __init__(self):
        # Camera
        self.camera_frames_total = Counter(
            "ibvap_camera_frames_total",
            "Total frames ingested per camera",
            ["camera_id"],
        )
        self.camera_dropped_frames_total = Counter(
            "ibvap_camera_dropped_frames_total",
            "Total dropped frames per camera (Latest-Frame overwritten)",
            ["camera_id"],
        )
        self.camera_reconnect_total = Counter(
            "ibvap_camera_reconnect_total",
            "Total camera reconnect attempts",
            ["camera_id", "result"],
        )

        # AI Inference
        self.ai_inference_total = Counter(
            "ibvap_ai_inference_total",
            "Total inference calls per detector",
            ["detector", "camera_id"],
        )
        self.ai_inference_latency_ms = Histogram(
            "ibvap_ai_inference_latency_ms", "Inference latency in milliseconds"
        )
        self.ai_queue_depth = Gauge(
            "ibvap_ai_queue_depth",
            "Current AI queue depth per camera (bounded = 1 by design)",
            ["camera_id"],
        )
        self.ai_result_age_ms = Gauge(
            "ibvap_ai_result_age_ms",
            "Age of the last AI result in milliseconds",
            ["camera_id"],
        )

        # Detector health
        self.detector_health = Gauge(
            "ibvap_detector_health",
            "Detector health: 1=RUNNING, 0=OFFLINE",
            ["detector"],
        )

        # GPU
        self.gpu_memory_used_mb = Gauge(
            "ibvap_gpu_memory_used_mb", "GPU VRAM used (MB)"
        )

        # Evidence
        self.evidence_write_failures = Counter(
            "ibvap_evidence_write_failures_total", "Total evidence write failures"
        )

        # Database
        self.database_errors = Counter(
            "ibvap_database_errors_total", "Total database errors"
        )

        # WebSocket
        self.websocket_connections = Gauge(
            "ibvap_websocket_connections", "Current active WebSocket connections"
        )

        self._all_metrics = [
            self.camera_frames_total,
            self.camera_dropped_frames_total,
            self.camera_reconnect_total,
            self.ai_inference_total,
            self.ai_inference_latency_ms,
            self.ai_queue_depth,
            self.ai_result_age_ms,
            self.detector_health,
            self.gpu_memory_used_mb,
            self.evidence_write_failures,
            self.database_errors,
            self.websocket_connections,
        ]

        # Start background system metrics collector
        self._stop = threading.Event()
        self._collector_thread = threading.Thread(
            target=self._collect_system_metrics, daemon=True
        )
        self._collector_thread.start()

    def _collect_system_metrics(self):
        """Background thread: update GPU VRAM and detector health every 5s."""
        from backend.app.services.detection_service import detection_service

        while not self._stop.is_set():
            try:
                if torch.cuda.is_available():
                    vram = torch.cuda.memory_reserved(0) / (1024**2)
                    self.gpu_memory_used_mb.set(round(vram, 1))

                self.detector_health.set(
                    1 if detection_service.ground_status == "RUNNING" else 0,
                    detector="ground",
                )
                self.detector_health.set(
                    1 if detection_service.airborne_status == "RUNNING" else 0,
                    detector="airborne",
                )
                self.detector_health.set(
                    1
                    if detection_service.security_item_status in ("RUNNING", "ONLINE")
                    else 0,
                    detector="security_item",
                )
            except Exception as e:
                logger.debug(f"Metrics collector error: {e}")
            self._stop.wait(5.0)

    def to_prometheus_text(self) -> str:
        """Render all metrics in Prometheus exposition format."""
        parts = []
        for m in self._all_metrics:
            parts.append(m.collect())
        return "\n\n".join(parts) + "\n"

    def stop(self):
        self._stop.set()


# Module-level singleton
ibvap_metrics = IBVAPMetrics()
