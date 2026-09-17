"""
TRINETRA — Perimeter Exit Vector Forecast Engine (Phase IX)
Estimates likely perimeter boundary exit directions based on trajectory heading,
velocity, and zone boundaries.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional

from backend.app.services.predictive_track_engine import predictive_track_engine

logger = logging.getLogger("PerimeterForecastEngine")


class PerimeterForecastEngine:
    def __init__(self):
        pass

    def estimate_exit_vector(
        self,
        track_id: str,
        current_zone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates likely perimeter boundary breach/exit vector.
        Returns cardinal direction (NORTH, SOUTH, EAST, WEST) or UNKNOWN.
        """
        heading_deg, cardinal_dir = predictive_track_engine.estimate_heading(track_id)
        vx, vy, speed = predictive_track_engine.estimate_velocity(track_id)

        # Safety: if stationary or missing track, return UNKNOWN
        if speed < 0.5 or cardinal_dir in ["UNKNOWN", "STATIONARY"]:
            return {
                "exit_direction": "UNKNOWN",
                "confidence": 0.20,
                "speed": speed,
                "reason": "Target speed insufficient for perimeter trajectory vectoring",
                "status": "ESTIMATED"
            }

        # Determine primary directional vector
        if "NORTH" in cardinal_dir:
            primary_exit = "NORTH"
        elif "SOUTH" in cardinal_dir:
            primary_exit = "SOUTH"
        elif "EAST" in cardinal_dir:
            primary_exit = "EAST"
        elif "WEST" in cardinal_dir:
            primary_exit = "WEST"
        else:
            primary_exit = "UNKNOWN"

        conf = 0.85 if speed > 1.5 else 0.70

        return {
            "exit_direction": primary_exit,
            "cardinal_heading": cardinal_dir,
            "heading_degrees": heading_deg,
            "speed_units_per_sec": speed,
            "confidence": conf,
            "reason": f"Active trajectory vector ({speed:.1f} u/s, {heading_deg:.1f}°) heading {cardinal_dir}",
            "status": "PREDICTED"
        }


perimeter_forecast_engine = PerimeterForecastEngine()
