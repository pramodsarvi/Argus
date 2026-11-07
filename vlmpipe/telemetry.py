"""Metrics (Prometheus) + tracing (OpenTelemetry). Every call is a no-op if the libs are missing.

    from vlmpipe import telemetry as tm
    tm.start_metrics_server(9100)
    tm.observe_decode(seconds, sampled, kept, visual_tokens)
    tm.observe_bundle(run_dict)            # output of vlmpipe.bundle.run_bundle
    with tm.span("segment", video_id=vid): ...
"""
from __future__ import annotations

import contextlib
from typing import Any, Iterator

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    HAVE_PROM = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    HAVE_PROM = False

try:
    from opentelemetry import trace as _otel_trace
    HAVE_OTEL = True
except ImportError:  # pragma: no cover
    HAVE_OTEL = False

_LAT = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300)
_E2E = (5, 10, 30, 60, 120, 180, 300, 600, 1200)


class _Noop:
    def labels(self, *a: Any, **k: Any) -> "_Noop":
        return self

    def observe(self, *a: Any, **k: Any) -> None: ...
    def inc(self, *a: Any, **k: Any) -> None: ...
    def set(self, *a: Any, **k: Any) -> None: ...


def _mk(kind: str, name: str, doc: str, labels: tuple[str, ...] = (), **kw: Any):
    if not HAVE_PROM:
        return _Noop()
    cls = {"h": Histogram, "c": Counter, "g": Gauge}[kind]
    return cls(name, doc, labels, **kw)


DECODE_SECONDS = _mk("h", "vlmpipe_decode_seconds", "Decode+sample time per segment", ("backend",), buckets=_LAT)
FRAMES_SAMPLED = _mk("c", "vlmpipe_frames_sampled_total", "Frames sampled before dedup")
FRAMES_KEPT = _mk("c", "vlmpipe_frames_kept_total", "Frames kept after dedup/filter")
VISUAL_TOKENS = _mk("h", "vlmpipe_visual_tokens", "Visual prompt tokens per segment",
                    buckets=(256, 1024, 2048, 4096, 8192, 16384, 32768))
BUNDLE_WALL = _mk("h", "vlmpipe_bundle_wall_seconds", "Bundle wall time", ("mode",), buckets=_LAT)
REQ_TTFT = _mk("h", "vlmpipe_request_ttft_seconds", "Per-prompt TTFT", ("prompt_id", "mode"), buckets=_LAT)
REQ_TOTAL = _mk("h", "vlmpipe_request_seconds", "Per-prompt total latency", ("prompt_id", "mode"), buckets=_LAT)
PROMPT_TOKENS = _mk("c", "vlmpipe_prompt_tokens_total", "Prompt tokens sent", ("mode",))
CACHED_TOKENS = _mk("c", "vlmpipe_cached_tokens_total", "Prompt tokens served from prefix cache", ("mode",))
CACHED_RATIO = _mk("g", "vlmpipe_last_bundle_cached_ratio", "Cached/prompt tokens, last bundle", ("mode",))
SCHEMA_FAILURES = _mk("c", "vlmpipe_schema_failures_total", "Invalid/failed outputs", ("prompt_id",))
E2E_SECONDS = _mk("h", "vlmpipe_e2e_seconds", "Upload->results latency per segment", buckets=_E2E)


def start_metrics_server(port: int = 9100) -> bool:
    if not HAVE_PROM:
        return False
    start_http_server(port)
    return True


def observe_decode(seconds: float, sampled: int, kept: int, visual_tokens: int,
                   backend: str = "ffmpeg") -> None:
    DECODE_SECONDS.labels(backend).observe(seconds)
    FRAMES_SAMPLED.inc(sampled)
    FRAMES_KEPT.inc(kept)
    VISUAL_TOKENS.observe(visual_tokens)


def observe_bundle(run: dict) -> float | None:
    """Record a run_bundle() result. Returns cached-token ratio (None if no tokens reported)."""
    mode = run["mode"]
    BUNDLE_WALL.labels(mode).observe(run["wall_s"])
    pt = ct = 0
    for r in run["results"]:
        REQ_TTFT.labels(r["prompt_id"], mode).observe(r["ttft_s"])
        REQ_TOTAL.labels(r["prompt_id"], mode).observe(r["total_s"])
        if not r["ok"]:
            SCHEMA_FAILURES.labels(r["prompt_id"]).inc()
        pt += r["prompt_tokens"]
        ct += r["cached_tokens"]
    PROMPT_TOKENS.labels(mode).inc(pt)
    CACHED_TOKENS.labels(mode).inc(ct)
    if not pt:
        return None
    ratio = ct / pt
    CACHED_RATIO.labels(mode).set(ratio)
    return ratio


def observe_e2e(seconds: float) -> None:
    E2E_SECONDS.observe(seconds)


def tracer():
    return _otel_trace.get_tracer("vlmpipe") if HAVE_OTEL else None


@contextlib.contextmanager
def span(name: str, video_id: str | None = None, **attrs: Any) -> Iterator[Any]:
    t = tracer()
    if t is None:
        yield None
        return
    with t.start_as_current_span(name) as s:
        if video_id is not None:
            s.set_attribute("video_id", video_id)
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s
