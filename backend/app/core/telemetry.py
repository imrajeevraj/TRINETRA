"""
TRINETRA — O-05: OpenTelemetry Distributed Tracing & Telemetry Instrumentation
Provides span generation across video ingestion, AI inference scheduling,
database queries, and HTTP requests with W3C Trace Context propagation.
"""

import logging
from typing import Optional

logger = logging.getLogger("Telemetry")

_tracer = None


def setup_telemetry(
    service_name: str = "ibvap-backend", otlp_endpoint: Optional[str] = None
):
    """O-05: Initialize OpenTelemetry tracer provider with OTLP / Jaeger exporter."""
    global _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create(
            {"service.name": service_name, "deployment.environment": "production"}
        )
        provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info("OpenTelemetry OTLP exporter enabled -> %s", otlp_endpoint)
            except Exception as e:
                logger.warning("Could not initialize OTLP exporter: %s", e)

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("ibvap.analytics")
        logger.info("OpenTelemetry distributed tracing initialized successfully.")
    except ImportError:
        logger.info(
            "OpenTelemetry SDK not installed; distributed tracing operating in no-op mode."
        )
    except Exception as e:
        logger.warning("OpenTelemetry setup failed: %s", e)


def get_tracer():
    """Get active tracer or fallback no-op tracer."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace

            _tracer = trace.get_tracer("ibvap.analytics")
        except Exception:

            class _NoOpTracer:
                class _NoOpSpan:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def set_attribute(self, *args, **kwargs):
                        pass

                    def record_exception(self, *args, **kwargs):
                        pass

                def start_as_current_span(self, name, *args, **kwargs):
                    return self._NoOpSpan()

            _tracer = _NoOpTracer()
    return _tracer
