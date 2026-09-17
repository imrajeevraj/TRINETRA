"""
TRINETRA — Terrain-Aware Trajectory Speed Correction Engine (Phase X)
Provides an extensible terrain provider abstraction (NullTerrainProvider, MockTerrainProvider)
adjusting transit ETAs based on gradient slope and surface friction.
Strictly labeled SIMULATED / DISABLED. No fabricated DEM coordinates.
"""

from __future__ import annotations
import math
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("TerrainSpeedCorrection")


class TerrainProvider:
    """Abstract interface for Digital Elevation Model (DEM) and surface friction telemetry."""
    @property
    def mode(self) -> str:
        raise NotImplementedError

    def get_elevation(self, x: float, y: float) -> Optional[float]:
        raise NotImplementedError

    def get_slope_deg(self, x1: float, y1: float, x2: float, y2: float) -> float:
        raise NotImplementedError

    def get_surface_type(self, x: float, y: float) -> str:
        raise NotImplementedError


class NullTerrainProvider(TerrainProvider):
    """Default provider when terrain correction is disabled."""
    @property
    def mode(self) -> str:
        return "DISABLED"

    def get_elevation(self, x: float, y: float) -> Optional[float]:
        return None

    def get_slope_deg(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return 0.0

    def get_surface_type(self, x: float, y: float) -> str:
        return "UNKNOWN"


class MockTerrainProvider(TerrainProvider):
    """
    Simulated terrain model with synthetic elevation contours and friction surfaces.
    Used for SIH demonstrations and benchmarking without external GIS dependencies.
    """
    @property
    def mode(self) -> str:
        return "SIMULATED"

    def get_elevation(self, x: float, y: float) -> Optional[float]:
        # Synthetic undulating border ridge elevation
        return round(150.0 + 25.0 * math.sin(x / 100.0) + 15.0 * math.cos(y / 80.0), 1)

    def get_slope_deg(self, x1: float, y1: float, x2: float, y2: float) -> float:
        z1 = self.get_elevation(x1, y1) or 150.0
        z2 = self.get_elevation(x2, y2) or 150.0
        dist_2d = max(1.0, math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        dz = z2 - z1
        slope_rad = math.atan2(dz, dist_2d)
        return round(math.degrees(slope_rad), 1)

    def get_surface_type(self, x: float, y: float) -> str:
        # Sector-based surface classification
        if y > 200.0:
            return "ROCKY"
        elif x > 150.0:
            return "SAND"
        return "GRAVEL"


class TerrainSpeedCorrectionEngine:
    def __init__(self, provider: Optional[TerrainProvider] = None):
        self.provider = provider or NullTerrainProvider()

    def set_provider(self, provider: TerrainProvider) -> None:
        self.provider = provider
        logger.info(f"Switched Terrain Provider to {provider.mode}")

    def compute_terrain_factor(self, slope_deg: float, surface_type: str = "UNKNOWN") -> float:
        """
        Computes walking speed multiplier:
        - Steep Incline (>15 deg): 0.70x (slows down target)
        - Moderate Incline (5-15 deg): 0.85x
        - Flat (-5 to +5 deg): 1.00x
        - Moderate Decline (-15 to -5 deg): 1.08x
        - Steep Decline (<-15 deg): 0.90x (cautious descent)
        """
        # Slope factor
        if slope_deg > 15.0:
            s_factor = 0.70
        elif slope_deg >= 5.0:
            s_factor = 0.85
        elif slope_deg > -5.0:
            s_factor = 1.00
        elif slope_deg >= -15.0:
            s_factor = 1.08
        else:
            s_factor = 0.90

        # Surface friction factor
        surf = surface_type.upper()
        if surf == "MUD":
            f_factor = 0.75
        elif surf == "SAND":
            f_factor = 0.82
        elif surf == "ROCKY":
            f_factor = 0.88
        elif surf == "GRAVEL":
            f_factor = 0.94
        else:
            f_factor = 1.00

        total_factor = round(s_factor * f_factor, 2)
        return max(0.40, min(1.30, total_factor))

    def compute_adjusted_eta(
        self,
        base_eta_sec: float,
        p1: Tuple[float, float] = (0.0, 0.0),
        p2: Tuple[float, float] = (100.0, 100.0),
        override_surface: Optional[str] = None
    ) -> Tuple[float, float, str]:
        """
        Adjusts expected ETA based on terrain slope and friction.
        Returns: (adjusted_eta_sec, terrain_factor, terrain_mode)
        """
        if self.provider.mode == "DISABLED":
            return base_eta_sec, 1.0, "DISABLED"

        x1, y1 = p1
        x2, y2 = p2
        slope = self.provider.get_slope_deg(x1, y1, x2, y2)
        surf = override_surface or self.provider.get_surface_type(x2, y2)

        factor = self.compute_terrain_factor(slope, surf)
        # Slower speed (factor < 1.0) means longer ETA
        adj_eta = round(base_eta_sec / factor, 1)

        logger.debug(f"Terrain adjustment: Base={base_eta_sec}s -> Adj={adj_eta}s (Slope={slope}°, Surf={surf}, Factor={factor})")
        return adj_eta, factor, self.provider.mode


terrain_speed_engine = TerrainSpeedCorrectionEngine()
