import pytest

import vlmpipe.telemetry as tm

RUN = {"mode": "warm", "wall_s": 1.2, "warm": None, "results": [
    {"prompt_id": "ocr", "ok": True, "ttft_s": 0.2, "total_s": 1.0, "prompt_tokens": 1000, "cached_tokens": 900},
    {"prompt_id": "safety", "ok": False, "ttft_s": 0.3, "total_s": 1.1, "prompt_tokens": 1000, "cached_tokens": 700},
]}


def test_bundle_ratio():
    assert tm.observe_bundle(RUN) == pytest.approx(0.8)
    assert tm.observe_bundle({**RUN, "results": []}) is None


@pytest.mark.skipif(not tm.HAVE_PROM, reason="prometheus_client missing")
def test_prom_values():
    from prometheus_client import REGISTRY
    before = REGISTRY.get_sample_value("vlmpipe_schema_failures_total", {"prompt_id": "safety"}) or 0
    tm.observe_bundle(RUN)
    tm.observe_decode(2.0, 10, 6, 3000)
    assert REGISTRY.get_sample_value("vlmpipe_schema_failures_total", {"prompt_id": "safety"}) == before + 1
    assert REGISTRY.get_sample_value("vlmpipe_last_bundle_cached_ratio", {"mode": "warm"}) == pytest.approx(0.8)
    assert REGISTRY.get_sample_value("vlmpipe_frames_kept_total") >= 6


def test_span_without_and_with_otel():
    with tm.span("x", video_id="v1", seg=3) as s:
        if tm.HAVE_OTEL:
            assert s is not None


def test_noop_when_libs_missing(monkeypatch):
    import importlib.util
    import sys
    for m in ("prometheus_client", "opentelemetry"):
        monkeypatch.setitem(sys.modules, m, None)   # makes `import m` raise ImportError
    spec = importlib.util.spec_from_file_location("tm_noop", tm.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not mod.HAVE_PROM and not mod.HAVE_OTEL
    assert mod.start_metrics_server() is False
    assert mod.observe_bundle(RUN) == pytest.approx(0.8)
    mod.observe_decode(1, 1, 1, 1)
    with mod.span("x", video_id="v") as s:
        assert s is None
