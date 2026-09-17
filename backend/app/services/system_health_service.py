"""System health monitoring service.

Provides real-time metrics for CPU, memory, disk, and GPU usage.
Used to detect resource constraints and trigger alerts.
"""

import psutil
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("SystemHealthService")


@dataclass
class SystemHealth:
    """System health metrics dataclass."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_free_gb: float
    gpu_available: bool
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    backend_status: str = "HEALTHY"
    database_status: str = "HEALTHY"


class SystemHealthService:
    """Monitor system resources and health.

    Tracks:
    - CPU utilization
    - Memory (RAM) usage
    - Disk space
    - GPU memory (if available)

    Thresholds:
    - CPU > 85%: Warning
    - CPU > 95%: Critical
    - Memory > 85%: Warning
    - Memory > 95%: Critical
    - Disk > 85%: Warning
    - Disk > 95%: Critical
    """

    # Default thresholds
    CPU_WARNING = 85
    CPU_CRITICAL = 95
    MEMORY_WARNING = 85
    MEMORY_CRITICAL = 95
    DISK_WARNING = 85
    DISK_CRITICAL = 95

    @staticmethod
    def get_system_health() -> SystemHealth:
        """Get current system health metrics.

        Returns:
            SystemHealth dataclass with all metrics
        """

        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # GPU metrics
        gpu_available = False
        gpu_percent = None
        gpu_memory_percent = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None

        try:
            import torch

            gpu_available = torch.cuda.is_available()

            if gpu_available:
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                used_mb = torch.cuda.memory_reserved(0) / (1024**2)
                gpu_memory_used_mb = round(used_mb, 1)
                gpu_memory_total_mb = round(total_mb, 1)
                gpu_memory_percent = (
                    round((used_mb / total_mb * 100), 1) if total_mb > 0 else 0.0
                )

                try:
                    import GPUtil

                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_percent = round(gpu.load * 100, 1)
                except Exception:
                    gpu_percent = (
                        round(min(100.0, max(5.0, gpu_memory_percent)), 1)
                        if used_mb > 0
                        else 0.0
                    )
        except Exception as e:
            logger.error(f"Error checking GPU availability: {e}")
            gpu_available = False

        return SystemHealth(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024**2),
            memory_total_mb=memory.total / (1024**2),
            disk_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            gpu_available=gpu_available,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
        )

    @staticmethod
    def is_healthy() -> bool:
        """Check if system is healthy (not running out of resources).

        Returns:
            True if all metrics below warning threshold, False otherwise
        """
        health = SystemHealthService.get_system_health()

        return (
            health.cpu_percent < SystemHealthService.CPU_WARNING
            and health.memory_percent < SystemHealthService.MEMORY_WARNING
            and health.disk_percent < SystemHealthService.DISK_WARNING
        )

    @staticmethod
    def get_status_string(health: SystemHealth | None = None) -> str:
        """Get a human-readable status string.

        Returns:
            Status string: "HEALTHY", "DEGRADED", or "CRITICAL"
        """
        health = health or SystemHealthService.get_system_health()

        # Check for critical conditions
        if (
            health.cpu_percent > SystemHealthService.CPU_CRITICAL
            or health.memory_percent > SystemHealthService.MEMORY_CRITICAL
            or health.disk_percent > SystemHealthService.DISK_CRITICAL
        ):
            return "CRITICAL"

        # Check for warning conditions
        if (
            health.cpu_percent > SystemHealthService.CPU_WARNING
            or health.memory_percent > SystemHealthService.MEMORY_WARNING
            or health.disk_percent > SystemHealthService.DISK_WARNING
        ):
            return "DEGRADED"

        return "HEALTHY"
