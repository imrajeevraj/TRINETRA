"""
TRINETRA Phase XV — Thermal Edge Model Deployment Packaging Service
Builds and cryptographically validates deployable edge packages for thermal AI candidates:
model artifact, SHA-256 digest, architecture, dataset lineage, benchmark version,
calibration dependency, sensor compatibility, preprocessing config, and atomic rollback metadata.

Strictly rejects packages upon SHA mismatch, incompatible architecture, missing calibration, or invalid state.
"""

from __future__ import annotations
import os
import time
import json
import yaml
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("ThermalDeploymentPackageService")

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_ROOT = REPO_ROOT / "data" / "deployment_packages"


class ThermalPackageValidationError(Exception):
    """Raised when an edge package fails validation."""
    pass


@dataclass
class ThermalPackageManifest:
    package_id: str
    model_name: str
    architecture: str  # "YOLO11", "YOLO26"
    model_sha256: str
    dataset_version: str
    benchmark_version: str
    calibration_dependency: Optional[str]
    sensor_compatibility: List[str]
    preprocessing_config: Dict[str, Any]
    runtime_version: str
    rollback_target_sha256: Optional[str]
    package_status: str  # "VALIDATED", "REJECTED", "DEPLOYED"
    created_at: float = field(default_factory=time.time)
    rejection_reason: Optional[str] = None


class ThermalDeploymentPackageService:
    """
    Manages edge model packaging for thermal pipelines with cryptographic verification.
    """

    SUPPORTED_ARCHITECTURES = {"YOLO11", "YOLO11n", "YOLO26", "YOLO26n"}
    SUPPORTED_SENSORS = {"SNS-CAM005-LWIR", "SNS-CAM006-LWIR", "SNS-CAM007-LWIR", "SNS-GENERIC-LWIR"}

    def __init__(self, packages_root: Optional[Path] = None):
        self.packages_root = packages_root or PACKAGES_ROOT
        self.packages_root.mkdir(parents=True, exist_ok=True)
        self._packages: Dict[str, ThermalPackageManifest] = {}

    def build_package(
        self,
        package_id: str,
        model_name: str,
        architecture: str,
        model_bytes: bytes,
        dataset_version: str,
        benchmark_version: str,
        sensor_compatibility: List[str],
        calibration_dependency: Optional[str] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        runtime_version: str = "ONNX_RUNTIME_1.17",
        rollback_target_sha256: Optional[str] = None,
    ) -> ThermalPackageManifest:
        # 1. Architecture check
        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ThermalPackageValidationError(f"INCOMPATIBLE_ARCHITECTURE: '{architecture}' not supported.")

        # 2. Sensor compatibility check
        for sensor in sensor_compatibility:
            if sensor not in self.SUPPORTED_SENSORS:
                raise ThermalPackageValidationError(f"INCOMPATIBLE_SENSOR: '{sensor}' is not an approved LWIR sensor.")

        # 3. Calibration dependency check
        if not calibration_dependency:
            raise ThermalPackageValidationError("MISSING_CALIBRATION: Thermal edge package requires a valid calibration dependency.")

        # 4. Compute model SHA
        computed_sha = hashlib.sha256(model_bytes).hexdigest().upper()

        pkg_dir = self.packages_root / package_id
        pkg_dir.mkdir(parents=True, exist_ok=True)

        model_file = pkg_dir / "model.pt"
        with open(model_file, "wb") as f:
            f.write(model_bytes)

        manifest = ThermalPackageManifest(
            package_id=package_id,
            model_name=model_name,
            architecture=architecture,
            model_sha256=computed_sha,
            dataset_version=dataset_version,
            benchmark_version=benchmark_version,
            calibration_dependency=calibration_dependency,
            sensor_compatibility=sensor_compatibility,
            preprocessing_config=preprocessing_config or {"input_size": 640, "color_mode": "MONO8"},
            runtime_version=runtime_version,
            rollback_target_sha256=rollback_target_sha256,
            package_status="VALIDATED",
        )

        # Write manifest files
        manifest_data = {
            "package_id": manifest.package_id,
            "model_name": manifest.model_name,
            "architecture": manifest.architecture,
            "model_sha256": manifest.model_sha256,
            "dataset_version": manifest.dataset_version,
            "benchmark_version": manifest.benchmark_version,
            "calibration_dependency": manifest.calibration_dependency,
            "sensor_compatibility": manifest.sensor_compatibility,
            "preprocessing_config": manifest.preprocessing_config,
            "runtime_version": manifest.runtime_version,
            "rollback_target_sha256": manifest.rollback_target_sha256,
            "created_at": manifest.created_at,
        }

        with open(pkg_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest_data, f)

        with open(pkg_dir / "sha256.txt", "w") as f:
            f.write(f"{computed_sha}  model.pt\n")

        self._packages[package_id] = manifest
        logger.info(f"Built thermal edge package {package_id} [SHA: {computed_sha[:12]}...]")
        return manifest

    def verify_package(self, package_id: str) -> Tuple[bool, str]:
        manifest = self._packages.get(package_id)
        if not manifest:
            return False, "PACKAGE_NOT_FOUND"

        pkg_dir = self.packages_root / package_id
        model_file = pkg_dir / "model.pt"
        if not model_file.exists():
            return False, "MODEL_FILE_MISSING"

        with open(model_file, "rb") as f:
            disk_sha = hashlib.sha256(f.read()).hexdigest().upper()

        if disk_sha != manifest.model_sha256:
            return False, f"SHA_MISMATCH: Manifest {manifest.model_sha256} != Disk {disk_sha}"

        if manifest.architecture not in self.SUPPORTED_ARCHITECTURES:
            return False, f"INCOMPATIBLE_ARCHITECTURE: {manifest.architecture}"

        if not manifest.calibration_dependency:
            return False, "MISSING_CALIBRATION"

        return True, "PACKAGE_VERIFIED_SUCCESSFULLY"

    def get_package(self, package_id: str) -> Optional[ThermalPackageManifest]:
        return self._packages.get(package_id)

    def list_packages(self) -> List[ThermalPackageManifest]:
        return list(self._packages.values())

    def reset(self):
        self._packages.clear()


thermal_deployment_package_service = ThermalDeploymentPackageService()
