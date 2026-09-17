"""
TRINETRA — Phase 17: /metrics API Endpoint (Prometheus text format)
Exposes ibvap_metrics singleton via GET /metrics.
"""

from fastapi import APIRouter, Response
from backend.app.services.metrics_service import ibvap_metrics

router = APIRouter(tags=["Observability"])


@router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus-compatible metrics endpoint",
    description="Returns all IBVAP pipeline metrics in Prometheus text format.",
)
def get_metrics():
    content = ibvap_metrics.to_prometheus_text()
    return Response(
        content=content, media_type="text/plain; version=0.0.4; charset=utf-8"
    )
