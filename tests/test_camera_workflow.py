"""CameraWorkflow (entity per camera) + camera activity logic.

Workflow tests run on Temporal's time-skipping server with mocked activities. Activity-logic tests
use a fake `vlmpipe.promptsets` and fake `run_two_tier` (both owned by sibling components)."""
import dataclasses
import sys
import types as pytypes
import uuid
from dataclasses import dataclass, field

import pytest

pytest.importorskip("temporalio")
import pytest_asyncio

from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from vlmpipe.orchestration import camera
from vlmpipe.orchestration.types import (CameraConfig, CameraState, CameraSummary, ChunkAnalyzeInput,
                                         ChunkInput, ChunkOutcome, ChunkSampleInput, FramesOut)
from vlmpipe.orchestration.workflows import CameraWorkflow
from vlmpipe.prompts import Registry

Q = "node-test"
CAM = "cam1"


# ------------------------------------------------------------------ workflow (mocked activities)
class Mocks:
    def __init__(self):
        self.sampled: list[tuple[float, bool]] = []
        self.analyzed: list[ChunkAnalyzeInput] = []
        self.stale: list[tuple[float, float]] = []

    def acts(self):
        @activity.defn(name="sample_chunk")
        async def sample_chunk(s: ChunkSampleInput) -> FramesOut:
            self.sampled.append((s.chunk.chunk_start_ts, s.degraded))
            return FramesOut(["/f/a.png"], {"frame_times": [0.0]})

        @activity.defn(name="analyze_chunk")
        async def analyze_chunk(a: ChunkAnalyzeInput) -> ChunkOutcome:
            self.analyzed.append(a)
            return ChunkOutcome(a.chunk.chunk_start_ts, "complete", [], {},
                                {"chunk": a.chunk.chunk_start_ts}, "set@1")

        @activity.defn(name="mark_chunk_stale")
        async def mark_chunk_stale(s: ChunkSampleInput, lag_s: float) -> int:
            self.stale.append((s.chunk.chunk_start_ts, lag_s))
            return 1
        return [sample_chunk, analyze_chunk, mark_chunk_stale]


@pytest_asyncio.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as e:
        yield e


async def _now(env) -> float:
    return (await env.get_current_time()).timestamp()


def cfg(**kw) -> CameraConfig:
    return CameraConfig(CAM, Q, **kw)


async def _start(env, c: CameraConfig, first: ChunkInput):
    return await env.client.start_workflow(
        CameraWorkflow.run, CameraState(c), id=f"camera-{c.camera_id}-{uuid.uuid4().hex[:6]}",
        task_queue=Q, start_signal="chunk", start_signal_args=[first])


@pytest.mark.asyncio
async def test_out_of_order_signals_processed_in_order_with_dedupe_and_context(env):
    m = Mocks()
    base = await _now(env)
    async with Worker(env.client, task_queue=Q, workflows=[CameraWorkflow], activities=m.acts()):
        h = await _start(env, cfg(max_chunks=3), ChunkInput(CAM, "/c/2.mp4", base + 360))
        await h.signal("chunk", ChunkInput(CAM, "/c/0.mp4", base))
        await h.signal("chunk", ChunkInput(CAM, "/c/0.mp4", base))          # duplicate
        await h.signal("chunk", ChunkInput(CAM, "/c/1.mp4", base + 180))
        summary: CameraSummary = await h.result()
    assert [a.chunk.chunk_start_ts for a in m.analyzed] == [base, base + 180, base + 360]
    assert summary.total_chunks == 3 and [r.status for r in summary.recent] == ["complete"] * 3
    # previous chunk's description carried into the next
    assert m.analyzed[0].previous_description is None
    assert m.analyzed[1].previous_description == {"chunk": base}
    assert m.analyzed[2].previous_description == {"chunk": base + 180}
    assert not any(a.degraded for a in m.analyzed) and not m.stale


@pytest.mark.asyncio
async def test_degraded_and_stale_paths(env):
    m = Mocks()
    base = await _now(env)
    stale_ts = base - 180 - 1000          # lag ~1000 s > 2 x 180
    degraded_ts = base - 180 - 270        # lag ~270 s: > 1 x 180, < 2 x 180
    async with Worker(env.client, task_queue=Q, workflows=[CameraWorkflow], activities=m.acts()):
        h = await _start(env, cfg(max_chunks=3), ChunkInput(CAM, "/c/s.mp4", stale_ts))
        await h.signal("chunk", ChunkInput(CAM, "/c/d.mp4", degraded_ts))
        await h.signal("chunk", ChunkInput(CAM, "/c/f.mp4", base))
        summary = await h.result()
    assert [ts for ts, _ in m.stale] == [stale_ts] and m.stale[0][1] > 360
    assert [(ts, d) for ts, d in m.sampled] == [(degraded_ts, True), (base, False)]
    assert [a.degraded for a in m.analyzed] == [True, False]
    st = {r.chunk_start_ts: r for r in summary.recent}
    assert st[stale_ts].status == "stale"
    assert st[degraded_ts].degraded and st[degraded_ts].status == "complete"


@pytest.mark.asyncio
async def test_continue_as_new_preserves_queue_and_context(env):
    m = Mocks()
    base = await _now(env)
    async with Worker(env.client, task_queue=Q, workflows=[CameraWorkflow], activities=m.acts()):
        h = await _start(env, cfg(max_chunks=5, continue_as_new_after=2),
                         ChunkInput(CAM, "/c/0.mp4", base))
        for i in range(1, 5):
            await h.signal("chunk", ChunkInput(CAM, f"/c/{i}.mp4", base + 180 * i))
        summary = await h.result()                      # follows continue-as-new runs
        desc = await env.client.get_workflow_handle(h.id).describe()
    assert [a.chunk.chunk_start_ts for a in m.analyzed] == [base + 180 * i for i in range(5)]
    assert m.analyzed[2].previous_description == {"chunk": base + 180}   # across the CAN boundary
    assert m.analyzed[4].previous_description == {"chunk": base + 540}
    assert summary.total_chunks == 5 and len(summary.recent) == 5
    assert desc.run_id != h.result_run_id                # at least one continue-as-new happened


@pytest.mark.asyncio
async def test_submit_chunk_signal_with_start(env):
    m = Mocks()
    base = await _now(env)
    c = CameraConfig("camX", "node-test", max_chunks=2)
    async with Worker(env.client, task_queue=Q, workflows=[CameraWorkflow], activities=m.acts()):
        await camera.submit_chunk(env.client, "camX", "/c/a.mp4", base, node="test", config=c)
        await camera.submit_chunk(env.client, "camX", "/c/b.mp4", base + 180, node="test", config=c)
        await camera.submit_chunk(env.client, "camX", "/c/b.mp4", base + 180, node="test", config=c)
        summary = await env.client.get_workflow_handle("camera-camX", result_type=CameraSummary).result()
    assert summary.camera_id == "camX" and summary.total_chunks == 2
    assert [a.chunk.chunk_path for a in m.analyzed] == ["/c/a.mp4", "/c/b.mp4"]


# ------------------------------------------------------------------ activity logic (no Temporal)
@dataclass(frozen=True)
class FP:                                   # prompt with optional `required`/`deadline_s`
    id: str
    version: int
    text: str
    max_tokens: int
    schema: dict = field(default_factory=dict, hash=False, compare=False)
    required: bool = True
    deadline_s: float = 30.0


def _fake_promptsets(monkeypatch, sampling=None, routing=None):
    reg = Registry("SYS", 3, (FP("safety", 2, "q1", 50), FP("summary", 1, "q2", 50, required=False)))
    calls = {"load": 0, "reload": 0}

    class PromptSetRegistry:
        @classmethod
        def load(cls, path):
            calls["load"] += 1
            return cls()

        def reload_if_changed(self):
            calls["reload"] += 1
            return None

        def for_camera(self, camera_id):
            if camera_id != CAM:
                raise KeyError(camera_id)
            return pytypes.SimpleNamespace(camera_id=camera_id, set_id="dock", set_version=7,
                                           registry=reg, routing=routing or {}, sampling=sampling or {})
    monkeypatch.setitem(sys.modules, "vlmpipe.promptsets",
                        pytypes.SimpleNamespace(PromptSetRegistry=PromptSetRegistry))
    monkeypatch.setattr(camera, "_REGISTRIES", {})
    return calls


def _fake_two_tier(monkeypatch, seen: dict):
    async def run_two_tier(client, model, reg, frames, **kw):
        seen["prompts"] = [p.id for p in reg.prompts]
        seen.update(kw)
        res = [{"prompt_id": p.id, "ok": True, "output": {"x": p.id}, "error": None, "ttft_s": 0.1,
                "total_s": 0.2, "prompt_tokens": 10, "cached_tokens": 5, "completion_tokens": 2}
               for p in reg.prompts]
        return {"mode": "two_tier", "results": res, "description": {"scene": "dock"}}
    monkeypatch.setitem(sys.modules, "vlmpipe.twotier", pytypes.SimpleNamespace(run_two_tier=run_two_tier))


def _ain(degraded=False, prev=None, mode="two_tier") -> ChunkAnalyzeInput:
    c = CameraConfig(CAM, Q, analysis_mode=mode)
    return ChunkAnalyzeInput(c, ChunkInput(CAM, "/c/x.mp4", 1_700_000_000.0),
                             FramesOut(["/f/a.png", "/f/b.png"], {"frame_times": [0.0, 7.5]}), prev, degraded)


@pytest.mark.asyncio
async def test_analyze_chunk_two_tier_keys_metrics_context(monkeypatch):
    calls = _fake_promptsets(monkeypatch, routing={"summary": "visual"})
    seen: dict = {}
    _fake_two_tier(monkeypatch, seen)
    pub: list[dict] = []

    async def publish(d):
        pub.append(d)
    out = await camera.analyze_chunk(_ain(prev={"scene": "before"}), publish, lambda d: None,
                                     client_factory=lambda url: object())
    assert out.status == "complete" and out.description == {"scene": "dock"} and out.prompt_set == "dock@7"
    assert seen["previous_description"] == {"scene": "before"}
    assert seen["routing_overrides"] == {"summary": "visual"}
    assert seen["prompts"] == ["safety", "summary"]
    k = pub[0]["idempotency_key"]
    assert k.startswith(f"{CAM}:1700000000.000:safety:v2:ps-dock@7:")
    m = pub[0]["metrics"]
    assert m["camera_id"] == CAM and m["chunk_start_ts"] == 1_700_000_000.0
    assert m["segment_start_s"] == 0.0 and m["frame_times"] == [0.0, 7.5]
    assert m["prompt_set"] == "dock" and m["prompt_set_version"] == 7
    # second call hot-reloads instead of re-loading
    await camera.analyze_chunk(_ain(), publish, lambda d: None, client_factory=lambda url: object())
    assert calls == {"load": 1, "reload": 1}


@pytest.mark.asyncio
async def test_analyze_chunk_degraded_skips_optional(monkeypatch):
    _fake_promptsets(monkeypatch)
    seen: dict = {}
    _fake_two_tier(monkeypatch, seen)

    async def publish(d):
        pass
    out = await camera.analyze_chunk(_ain(degraded=True), publish, lambda d: None,
                                     client_factory=lambda url: object())
    assert seen["prompts"] == ["safety"] and [r.prompt_id for r in out.results] == ["safety"]
    assert out.results[0].metrics["degraded"] is True


@pytest.mark.asyncio
async def test_analyze_chunk_resume_skips_published(monkeypatch):
    _fake_promptsets(monkeypatch)
    seen: dict = {}
    _fake_two_tier(monkeypatch, seen)
    pub: list[dict] = []

    async def publish(d):
        pub.append(d)
    first = await camera.analyze_chunk(_ain(), publish, lambda d: None, client_factory=lambda url: object())
    resumed = [dataclasses.asdict(first.results[0])]          # safety already published
    pub.clear()
    await camera.analyze_chunk(_ain(), publish, lambda d: None, resumed=resumed,
                               client_factory=lambda url: object())
    assert [d["prompt_id"] for d in pub] == ["summary"]


@pytest.mark.asyncio
async def test_mark_chunk_stale_publishes_per_prompt(monkeypatch):
    _fake_promptsets(monkeypatch)
    pub: list[dict] = []

    async def publish(d):
        pub.append(d)
    n = await camera.mark_chunk_stale(ChunkSampleInput(CameraConfig(CAM, Q), ChunkInput(CAM, "/c", 100.0)),
                                      publish, 999.0)
    assert n == 2 and {d["status"] for d in pub} == {"stale"}
    assert pub[0]["metrics"]["lag_s"] == 999.0 and pub[0]["metrics"]["camera_id"] == CAM


def test_sampling_params_defaults_override_and_degrade():
    p = camera.sampling_params(None, False)
    assert p["strategy"] == "scene" and p["max_frames"] == 32 and p["max_pixels"] == 256 * 28 * 28
    assert camera.sampling_params({"max_frames": 24}, False)["max_frames"] == 24
    assert camera.sampling_params(None, True)["max_frames"] == 16
    assert camera.sampling_params({"max_frames": 6}, True)["max_frames"] == 4


def test_cap_description():
    assert camera.cap_description({"a": "x" * 10}, 100) == {"a": "x" * 10}
    assert camera.cap_description({"a": "x" * 1000}, 100) is None
    assert camera.cap_description(None, 100) is None


@pytest.mark.asyncio
async def test_idle_camera_workflow_completes(env):
    m = Mocks()
    base = await _now(env)
    async with Worker(env.client, task_queue=Q, workflows=[CameraWorkflow], activities=m.acts()):
        h = await _start(env, cfg(idle_timeout_s=600), ChunkInput(CAM, "/c/0.mp4", base))
        summary: CameraSummary = await h.result()      # time-skipping fast-forwards the idle wait
    assert summary.ended == "idle" and summary.total_chunks == 1


@pytest.mark.asyncio
async def test_degraded_publishes_skipped_for_optional(monkeypatch):
    _fake_promptsets(monkeypatch)
    _fake_two_tier(monkeypatch, {})
    pub: list[dict] = []

    async def publish(d):
        pub.append(d)
    await camera.analyze_chunk(_ain(degraded=True), publish, lambda d: None,
                               client_factory=lambda url: object())
    st = {d["prompt_id"]: d["status"] for d in pub}
    assert st == {"safety": "ok", "summary": "skipped"}



def test_should_warm_only_for_big_prefix_and_many_prompts():
    from vlmpipe.orchestration.segment_runner import should_warm
    assert not should_warm("auto", 936, 5) and not should_warm("auto", 9000, 2)
    assert should_warm("auto", 3744, 5) and should_warm("always", 1, 1) and not should_warm("never", 10**6, 9)


def test_latency_profile_and_overrides():
    lat = camera.sampling_params(None, False, "latency")
    assert lat["max_frames"] == 8 and lat["max_pixels"] == 128 * 28 * 28
    over = camera.sampling_params({"max_frames": 12}, False, "latency")        # camera set wins
    assert over["max_frames"] == 12 and over["max_pixels"] == 128 * 28 * 28
    assert camera.sampling_params(None, True, "balanced")["max_frames"] == 16     # degraded halves


@pytest.mark.asyncio
async def test_frames_removed_after_success_only(monkeypatch, tmp_path):
    _fake_promptsets(monkeypatch)
    _fake_two_tier(monkeypatch, {})
    d = tmp_path / "chunk"
    d.mkdir()
    paths = []
    for n in "ab":
        (d / f"{n}.png").write_bytes(b"x")
        paths.append(str(d / f"{n}.png"))

    async def publish(x):
        pass
    a = _ain()
    a.frames = FramesOut(paths, {"frame_times": [0.0, 1.0]})
    await camera.analyze_chunk(a, publish, lambda x: None, client_factory=lambda u: object())
    assert not d.exists()                                    # frames + empty dir gone
    d.mkdir()
    (d / "a.png").write_bytes(b"x")
    a.frames = FramesOut([str(d / "a.png")], {"frame_times": [0.0]})
    a.config.cleanup_frames = False
    await camera.analyze_chunk(a, publish, lambda x: None, client_factory=lambda u: object())
    assert (d / "a.png").exists()


@pytest.mark.asyncio
async def test_client_is_pooled_per_loop():
    from vlmpipe.clients import get_client
    assert get_client("http://x/v1") is get_client("http://x/v1")
    assert get_client("http://x/v1") is not get_client("http://y/v1")
