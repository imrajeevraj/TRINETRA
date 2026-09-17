import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from backend.app.core.config import config_manager

logger = logging.getLogger("RiskEngine")


class RiskResult:
    """Structured result from risk evaluation."""

    def __init__(
        self,
        risk_score: int,
        severity: str,
        reasons: List[Dict[str, Any]],
        available: bool = True,
        error: Optional[str] = None,
    ):
        self.risk_score = risk_score
        self.severity = severity
        self.reasons = reasons  # List of {"reason": str, "points": int}
        self.available = available
        self.error = error

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "severity": self.severity,
            "risk_reasons": json.dumps([r["reason"] for r in self.reasons]),
        }


class RiskEngine:
    """
    Transparent, configurable, rule-based risk scoring engine.

    Transforms raw border-security events into an interpretable risk score and severity.
    Every score contribution produces an explainable reason. No black-box ML.

    Pipeline:
        SecurityEvent → RiskEngine.evaluate_event() → augmented event dict

    The score is deterministic from configuration and context:
        - Object type weight (person/vehicle)
        - Event type weight (zone_entry/zone_exit/fence_crossing)
        - Direction weight (inward/outward/unknown)
        - Night-time context weight
        - Zone-specific weight overrides

    Score is bounded 0–100. Severity is mapped from configurable thresholds.
    """

    def __init__(
        self,
        config_path: str = "configs/risk.yaml",
        zones_config_path: str = "configs/zones.yaml",
    ):
        self.config_path = config_path
        self.zones_config_path = zones_config_path
        self.config: Dict = {}
        self.zones_config: Dict = {}
        self._config_valid = False
        self._config_errors: List[str] = []
        self._use_config_manager = True
        self.load_config()

    def load_config(self):
        """Load and validate risk configuration from YAML files."""
        self._config_valid = False
        self._config_errors = []

        # Use config_manager for dynamic loading
        risk_config = config_manager.get_config("risk")
        self.config = risk_config.get("risk", {}) if risk_config else {}

        # Try to load zones config
        zones_config = config_manager.get_config("zones")
        self.zones_config = zones_config if zones_config else {}

        # Validate configuration
        self._validate_config()

    def _get_weights(self) -> Dict[str, float]:
        """Return weights from the validated configuration snapshot."""
        weights = self.config.get("weights", {})

        # Provide defaults if not specified
        return (
            weights
            if weights
            else {
                "person": 20,
                "vehicle": 15,
                "zone_entry": 30,
                "fence": 25,
                "night": 10,
                "watchlist": 40,
            }
        )

    def _validate_config(self):
        """Validate all configuration values for correctness."""
        errors = []

        # Validate weights
        weights = self.config.get("weights", {})
        if not weights:
            errors.append("No weights configured in risk.yaml")
        else:
            for key, value in weights.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"Weight '{key}' is not a number: {value}")
                elif value < 0:
                    errors.append(f"Weight '{key}' is negative: {value}")

        # Validate severity thresholds
        severity = self.config.get("severity", {})
        if not severity:
            errors.append("No severity thresholds configured")
        else:
            for level, limits in severity.items():
                if not isinstance(limits, dict):
                    errors.append(f"Severity '{level}' is not a dict")
                    continue
                min_val = limits.get("min")
                max_val = limits.get("max")
                if min_val is None or max_val is None:
                    errors.append(f"Severity '{level}' missing min or max")
                elif not isinstance(min_val, (int, float)) or not isinstance(
                    max_val, (int, float)
                ):
                    errors.append(f"Severity '{level}' min/max are not numbers")
                elif min_val > max_val:
                    errors.append(
                        f"Severity '{level}' min ({min_val}) > max ({max_val})"
                    )
                elif min_val < 0 or max_val > 100:
                    errors.append(f"Severity '{level}' out of 0–100 range")

            ranges = sorted(
                (limits.get("min"), limits.get("max"))
                for limits in severity.values()
                if isinstance(limits, dict)
                and isinstance(limits.get("min"), (int, float))
                and isinstance(limits.get("max"), (int, float))
            )
            if ranges and (ranges[0][0] != 0 or ranges[-1][1] != 100):
                errors.append("Severity thresholds must cover the full 0–100 range")
            for previous, current in zip(ranges, ranges[1:]):
                if current[0] <= previous[1]:
                    errors.append("Severity thresholds overlap")
                elif current[0] != previous[1] + 1:
                    errors.append("Severity thresholds contain a gap")

        # Validate night config
        night = self.config.get("night", {})
        if night.get("enabled", False):
            start_str = night.get("start", "")
            end_str = night.get("end", "")
            for label, val in [("start", start_str), ("end", end_str)]:
                if not val or not isinstance(val, str):
                    errors.append(f"Night {label} is missing or not a string")
                else:
                    try:
                        parts = val.split(":")
                        h, m = int(parts[0]), int(parts[1])
                        if not (0 <= h <= 23 and 0 <= m <= 59):
                            errors.append(f"Night {label} '{val}' out of valid range")
                    except (ValueError, IndexError):
                        errors.append(
                            f"Night {label} '{val}' is not valid HH:MM format"
                        )

        # Validate zone-specific weights
        zone_overrides = self.config.get("zones", {})
        if zone_overrides:
            for zone_id, zone_cfg in zone_overrides.items():
                if zone_cfg and isinstance(zone_cfg, dict):
                    rw = zone_cfg.get("risk_weight")
                    if rw is not None:
                        if not isinstance(rw, (int, float)):
                            errors.append(
                                f"Zone '{zone_id}' risk_weight is not a number: {rw}"
                            )
                        elif rw < 0:
                            errors.append(
                                f"Zone '{zone_id}' risk_weight is negative: {rw}"
                            )

        if errors:
            for err in errors:
                logger.error("Risk config validation error: %s", err)
            self._config_errors = errors
            self._config_valid = False
        else:
            self._config_valid = True
            logger.info("Risk configuration validated successfully.")

    def is_night_time(self, current_time: datetime) -> bool:
        """
        Determine if the given timestamp falls within configured night hours.
        Handles midnight-crossing intervals correctly (e.g., 22:00–05:00).
        Uses the configured timezone documentation but evaluates against the
        timestamp as-provided (UTC assumed if naive).
        """
        night_config = self.config.get("night", {})
        if not night_config.get("enabled", False):
            return False

        start_str = night_config.get("start", "22:00")
        end_str = night_config.get("end", "05:00")

        try:
            start_hour, start_minute = map(int, start_str.split(":"))
            end_hour, end_minute = map(int, end_str.split(":"))

            current_minute_of_day = current_time.hour * 60 + current_time.minute
            start_minute_of_day = start_hour * 60 + start_minute
            end_minute_of_day = end_hour * 60 + end_minute

            if start_minute_of_day < end_minute_of_day:
                # Same-day interval (e.g., 01:00–06:00)
                return start_minute_of_day <= current_minute_of_day <= end_minute_of_day
            else:
                # Midnight-crossing interval (e.g., 22:00–05:00)
                return (
                    current_minute_of_day >= start_minute_of_day
                    or current_minute_of_day <= end_minute_of_day
                )
        except Exception as e:
            logger.error("Error parsing night time config: %s", e)
            return False

    def _get_zone_override_weight(self, zone_id: Optional[str]) -> Optional[int]:
        """
        Look up zone-specific risk weight override from risk.yaml.
        Returns the override weight if configured, None otherwise.
        """
        if not zone_id:
            return None

        zone_overrides = self.config.get("zones", {})
        if not zone_overrides:
            return None

        zone_cfg = zone_overrides.get(zone_id)
        if zone_cfg and isinstance(zone_cfg, dict):
            rw = zone_cfg.get("risk_weight")
            if rw is not None and isinstance(rw, (int, float)):
                return int(rw)

        return None

    def evaluate_event(self, event_dict: dict) -> dict:
        """
        Receives a security event dict and augments it with risk_score, severity, and risk_reasons.

        The score is calculated once at event creation time (deterministic).
        Every score contribution produces an explainable reason.
        Score is bounded 0–100.

        If configuration is invalid or missing, the event is marked with
        risk_score=0, severity="UNAVAILABLE" rather than a misleading score.

        Args:
            event_dict: Security event dictionary with keys like event_type, object_type, direction, etc.

        Returns:
            The same dict augmented with risk_score, severity, risk_reasons.
        """
        # Read and validate one coherent configuration snapshot per event.
        # Test instances intentionally bypass this behavior by constructing an
        # engine with ``__new__`` and supplying a deterministic config.
        if getattr(self, "_use_config_manager", False):
            self.load_config()

        if not self.config or not self._config_valid:
            # Configuration is missing or invalid — fail safely
            error_msg = (
                "; ".join(self._config_errors)
                if self._config_errors
                else "No risk configuration loaded"
            )
            logger.warning("Risk assessment unavailable: %s", error_msg)
            event_dict["risk_score"] = 0
            event_dict["severity"] = "UNAVAILABLE"
            event_dict["risk_reasons"] = json.dumps(
                ["Risk assessment unavailable: configuration error"]
            )
            return event_dict

        # Get current weights (may be hot-reloaded from file)
        weights = self._get_weights()
        total_score = 0
        reasons = []  # List of {"reason": str, "points": int}

        # 1. Object Type Weight
        obj_type = event_dict.get("object_type", "UNKNOWN")
        if obj_type:
            obj_type = obj_type.upper()

        if obj_type == "PERSON":
            score = weights.get("person", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Person", "points": score})
        elif obj_type in ["VEHICLE", "CAR", "TRUCK", "BUS", "MOTORCYCLE"]:
            score = weights.get("vehicle", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Vehicle", "points": score})
        # UNKNOWN object type: 0 points, no reason (do not guess)

        # 2. Event Type Weight (with zone-specific override)
        event_type = event_dict.get("event_type", "")
        zone_id = event_dict.get("zone_id")
        zone_override = self._get_zone_override_weight(zone_id)

        if event_type == "ZONE_ENTRY":
            if zone_override is not None:
                total_score += zone_override
                reasons.append(
                    {
                        "reason": "Restricted Zone (zone-specific)",
                        "points": zone_override,
                    }
                )
            else:
                score = weights.get("restricted_zone_entry", 0)
                if score > 0:
                    total_score += score
                    reasons.append({"reason": "Restricted Zone", "points": score})
        elif event_type == "ZONE_EXIT":
            if zone_override is not None:
                # Zone exit with override — use a fraction or the exit weight, not the override
                score = weights.get("zone_exit", 0)
                if score > 0:
                    total_score += score
                    reasons.append({"reason": "Zone Exit", "points": score})
            else:
                score = weights.get("zone_exit", 0)
                if score > 0:
                    total_score += score
                    reasons.append({"reason": "Zone Exit", "points": score})
        elif event_type == "VIRTUAL_FENCE_CROSSING":
            if zone_override is not None:
                total_score += zone_override
                reasons.append(
                    {
                        "reason": "Virtual Fence Crossing (zone-specific)",
                        "points": zone_override,
                    }
                )
            else:
                score = weights.get("virtual_fence_crossing", 0)
                if score > 0:
                    total_score += score
                    reasons.append(
                        {"reason": "Virtual Fence Crossing", "points": score}
                    )
        elif event_type == "DRONE_DETECTED":
            score = weights.get("drone_detected", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Drone/UAV Detected", "points": score})
        elif event_type == "WEAPON_DETECTED":
            score = weights.get("weapon_detected", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Weapon Detected", "points": score})
        elif event_type == "CONTRABAND_DETECTED":
            score = weights.get("contraband_detected", 0)
            if score > 0:
                total_score += score
                reasons.append(
                    {"reason": "Contraband/Suspicious Item Detected", "points": score}
                )
        elif event_type == "CRAWLING_DETECTED":
            score = weights.get("crawling_detected", 0)
            if score > 0:
                total_score += score
                reasons.append(
                    {"reason": "Crawling Behavior Detected", "points": score}
                )
        elif event_type == "LOITERING_DETECTED":
            score = weights.get("loitering_detected", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Loitering Detected", "points": score})
        elif event_type == "PLATOON_DETECTED":
            score = weights.get("platoon_detected", 0)
            if score > 0:
                total_score += score
                reasons.append(
                    {"reason": "Platoon/Army Grouping Detected", "points": score}
                )
        elif event_type == "UNKNOWN_FACE":
            score = weights.get("unknown_face", 0)
            if score > 0:
                total_score += score
                reasons.append(
                    {"reason": "Unknown/Unverified Face Detected", "points": score}
                )

        # 3. Direction Weight
        direction = event_dict.get("direction")
        if direction:
            direction = direction.upper()

        if direction == "INWARD":
            score = weights.get("inward_direction", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Inward Movement", "points": score})
        elif direction == "OUTWARD":
            score = weights.get("outward_direction", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Outward Movement", "points": score})
        # UNKNOWN or None direction: 0 points (do not guess)

        # 4. Night-Time Context
        timestamp = event_dict.get("timestamp", datetime.utcnow())
        if self.is_night_time(timestamp):
            score = weights.get("night_time", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Night Time", "points": score})

        # 5. Watchlist Context
        if event_dict.get("watchlist_status") == "WATCHLIST MATCH":
            score = weights.get("watchlist_match", 0)
            if score > 0:
                total_score += score
                reasons.append({"reason": "Watchlist Match", "points": score})

        # Cap the score at 0–100
        final_score = max(0, min(total_score, 100))

        # Determine Severity from configurable thresholds
        severity_val = "LOW"
        severity_config = self.config.get("severity", {})
        for sev_level, limits in severity_config.items():
            if isinstance(limits, dict):
                sev_min = limits.get("min", 0)
                sev_max = limits.get("max", 100)
                if sev_min <= final_score <= sev_max:
                    severity_val = sev_level.upper()
                    break

        event_dict["risk_score"] = final_score
        event_dict["severity"] = severity_val
        event_dict["risk_reasons"] = json.dumps([r["reason"] for r in reasons])

        return event_dict


# Global singleton
risk_engine = RiskEngine()
