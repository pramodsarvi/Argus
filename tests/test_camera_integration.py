"""Cross-layer contract: camera orchestration output -> results layer -> absolute timestamps."""
import asyncio
from types import SimpleNamespace

from vlmpipe.orchestration.camera import chunk_metrics
from vlmpipe.results.publisher import AsyncRecordPublisher, InMemoryPublisher
from vlmpipe.results.reducer import reduce_video  # noqa: F401  (import check)
from vlmpipe.results.consumer import default_schemas
from vlmpipe.promptsets import PromptSetRegistry

T0 = 1_793_000_000.0   # chunk start, UTC epoch


def _analyze_input():
    return SimpleNamespace(
        config=SimpleNamespace(camera_id="dock-07"),
        chunk=SimpleNamespace(chunk_start_ts=T0),
        frames=SimpleNamespace(stats={"frame_times": [0.0, 12.5, 40.0]}),
        degraded=False)


def test_absolute_time_not_double_counted():
    cps = SimpleNamespace(set_id="warehouse_dock", set_version=1)
    m = chunk_metrics(_analyze_input(), cps)
    d = {"video_id": "dock-07", "segment_index": int(T0), "prompt_id": "safety",
         "prompt_version": 1, "model": "m", "status": "ok",
         "output": {"violations": [{"type": "no_helmet", "frame": 2, "description": "x"}]},
         "metrics": m}
    rec = AsyncRecordPublisher.to_record(d)
    assert rec.camera_id == "dock-07" and rec.chunk_start_ts == T0
    assert rec.base_ts == T0                         # chunk start, not 2x
    assert rec.base_ts + rec.frame_times[1] == T0 + 12.5   # frame 2 (1-based) absolute time


def test_publish_keyed_by_camera():
    inner = InMemoryPublisher()
    pub = AsyncRecordPublisher(inner)
    cps = SimpleNamespace(set_id="warehouse_dock", set_version=1)
    d = {"video_id": "dock-07", "segment_index": int(T0), "prompt_id": "safety",
         "prompt_version": 1, "model": "m", "status": "ok", "output": {"violations": []},
         "metrics": chunk_metrics(_analyze_input(), cps)}
    asyncio.run(pub.publish(d))
    (topic, key, _), = inner.messages
    assert key == b"dock-07"


def test_consumer_knows_every_camera_prompt_schema():
    reg = PromptSetRegistry.load()
    schemas = default_schemas()
    cams = ["dock-01", "entrance-3", "line-2", "lobby-main", "random-cam"]
    for cam in cams:
        cps = reg.for_camera(cam)
        for p in cps.registry.prompts:
            assert (p.id, p.version) in schemas, f"{cam}: no schema for {p.id} v{p.version}"
