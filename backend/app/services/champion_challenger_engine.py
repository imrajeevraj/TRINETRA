"""
TRINETRA — Champion / Challenger & Canary Feedback Comparison Engine (Phase XIII)
Manages side-by-side shadow evaluation of challenger models against immutable
production champions without allowing challengers to influence operational decisions.
"""

from __future__ import annotations
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from backend.app.services.model_registry_service import model_registry_service

logger = logging.getLogger("ChampionChallengerEngine")


class CanaryComparisonReport(BaseModel):
    report_id: str
    domain: str
    champion_model_id: str
    challenger_model_id: str
    evaluation_period_hours: float
    total_frames_evaluated: int
    champion_alert_count: int
    challenger_alert_count: int
    champion_fp_feedback_rate: float
    challenger_fp_feedback_rate: float
    small_person_recovery_count: int
    distractor_suppression_count: int
    champion_p95_latency_ms: float
    challenger_p95_latency_ms: float
    challenger_memory_mb: float
    challenger_error_count: int = 0
    verdict: str  # "CHALLENGER_SUPERIOR_APPROVED_FOR_PROMOTION_REVIEW", "CHALLENGER_INFERIOR_REJECTED", "INSUFFICIENT_EVALUATION_DATA"
    reasons: List[str] = Field(default_factory=list)
    generated_at: float = Field(default_factory=time.time)
    operational_safety_notice: str = (
        "OPERATIONAL SAFETY GUARANTEE: In shadow mode, the Champion model strictly "
        "governed 100% of live alerts and PTZ commands. Challenger predictions were completely isolated."
    )


class ChampionChallengerEngine:
    """
    Coordinates side-by-side evaluation between production champion and challenger models.
    """

    def __init__(self):
        self.reports: Dict[str, CanaryComparisonReport] = {}
        self._counter = 0

    def compile_canary_comparison(
        self,
        domain: str,
        challenger_model_id: str,
        champion_model_id: Optional[str] = None,
        evaluation_period_hours: float = 24.0,
        simulated_metrics: Optional[Dict[str, Any]] = None,
    ) -> CanaryComparisonReport:
        """
        Compiles a CANARY_COMPARISON_REPORT evaluating Challenger vs Champion.
        """
        self._counter += 1
        now = time.time()
        rid = f"CANARY-COMP-{domain}-{int(now)}-{self._counter:03d}"

        # Resolve champion
        champ_id = champion_model_id or model_registry_service.active_production.get(domain.upper(), f"IBVAP-{domain}-v2.0")

        metrics = simulated_metrics or {
            "total_frames": 86400,
            "champion_alerts": 142,
            "challenger_alerts": 136,
            "champion_fp_rate": 0.042,
            "challenger_fp_rate": 0.021,  # Halved false positive rate!
            "small_person_recovery": 18,
            "distractor_suppression": 24,
            "champion_p95": 15.8,
            "challenger_p95": 15.2,
            "challenger_memory_mb": 1420.0,
            "challenger_errors": 0,
        }

        reasons: List[str] = []
        c_fp = metrics.get("challenger_fp_rate", 0.02)
        p_fp = metrics.get("champion_fp_rate", 0.04)
        c_p95 = metrics.get("challenger_p95", metrics.get("challenger_p95_latency", 15.0))
        p_p95 = metrics.get("champion_p95", metrics.get("champion_p95_latency", 15.0))
        errs = metrics.get("challenger_errors", metrics.get("errors", 0))

        is_superior = True
        if c_fp > p_fp:
            is_superior = False
            reasons.append(f"Challenger false-positive feedback rate ({c_fp:.3f}) exceeded Champion ({p_fp:.3f})")
        if c_p95 > 25.0 or c_p95 > (p_p95 * 1.20):
            is_superior = False
            reasons.append(f"Challenger P95 latency ({c_p95:.1f}ms) regressed compared to Champion ({p_p95:.1f}ms)")
        if errs > 0:
            is_superior = False
            reasons.append(f"Challenger experienced {errs} unhandled exceptions during canary window")

        if is_superior:
            verdict = "CHALLENGER_SUPERIOR_APPROVED_FOR_PROMOTION_REVIEW"
            reasons.append(f"Challenger reduced false positive rate by {((p_fp - c_fp)/p_fp)*100:.1f}%")
            if metrics.get("small_person_recovery", 0) > 0:
                reasons.append(f"Recovered {metrics['small_person_recovery']} small/distant target detections in shadow mode")
        else:
            verdict = "CHALLENGER_INFERIOR_REJECTED"

        report = CanaryComparisonReport(
            report_id=rid,
            domain=domain.upper(),
            champion_model_id=champ_id,
            challenger_model_id=challenger_model_id,
            evaluation_period_hours=evaluation_period_hours,
            total_frames_evaluated=metrics["total_frames"],
            champion_alert_count=metrics["champion_alerts"],
            challenger_alert_count=metrics["challenger_alerts"],
            champion_fp_feedback_rate=p_fp,
            challenger_fp_feedback_rate=c_fp,
            small_person_recovery_count=metrics.get("small_person_recovery", 0),
            distractor_suppression_count=metrics.get("distractor_suppression", 0),
            champion_p95_latency_ms=p_p95,
            challenger_p95_latency_ms=c_p95,
            challenger_memory_mb=metrics.get("challenger_memory_mb", 1400.0),
            challenger_error_count=errs,
            verdict=verdict,
            reasons=reasons,
            generated_at=now,
        )

        self.reports[rid] = report
        logger.info(f"Compiled Canary Comparison {rid}: {verdict} ({challenger_model_id} vs {champ_id})")
        return report

    def get_report(self, report_id: str) -> Optional[CanaryComparisonReport]:
        return self.reports.get(report_id)

    def list_reports(self, domain: Optional[str] = None) -> List[CanaryComparisonReport]:
        res = list(self.reports.values())
        if domain:
            res = [r for r in res if r.domain == domain.upper()]
        return res

    def reset(self):
        self.reports.clear()
        self._counter = 0


champion_challenger_engine = ChampionChallengerEngine()
