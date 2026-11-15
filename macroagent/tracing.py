"""
OpenTelemetry tracing for the agent: one trace per question, spans for the agent run, each model call and
each tool call (model, tokens, latency, tool name, arguments, errors).

Spans go to a JSONL file (results/traces.jsonl). Set OTEL_EXPORTER_OTLP_ENDPOINT (and install
opentelemetry-exporter-otlp) to also stream them to any OTLP backend, e.g. a local Arize Phoenix or Langfuse.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a") as f:
            for s in spans:
                ctx = s.get_span_context()
                f.write(json.dumps({
                    "trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}",
                    "parent_id": f"{s.parent.span_id:016x}" if s.parent else None, "name": s.name,
                    "start_ns": s.start_time, "duration_ms": round((s.end_time - s.start_time) / 1e6, 2),
                    "status": s.status.status_code.name, "attributes": dict(s.attributes or {})}, default=str) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def make_tracer(path: Optional[Path], service: str = "macro-forecast-agent"):
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    if path:
        provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except ImportError:
            pass
    return provider.get_tracer("macroagent"), provider
