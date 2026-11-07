"""Orchestration tests: workflow on Temporal's time-skipping server with mocked activities, plus
SegmentRunner / analyze_segment activity tests (resume, deadlines, retries, two-tier dispatch)."""
import asyncio
import dataclasses
import sys
import types as pytypes
import uuid

import pytest

pytest.importorskip("temporalio")
import pytest_asyncio

from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from vlmpipe.orchestration import activities as acts_mod
from vlmpipe.orchestration.activities import Activities
from vlmpipe.orchestration.segment_runner import SegmentRunner
from vlmpipe.orchestration.types import (AnalyzeInput, FramesOut, PromptPlan, PromptResult, PromptSpec,
                                         SegmentInput, SegmentOutcome, VideoInput, idempotency_key)
from vlmpipe.orchestration.workflows import VideoAnalysisWorkflow, segment_status

Q = "node-test"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def plan(*specs: PromptSpec) -> PromptPlan:
    return PromptPlan("SYS", 1, list(specs))


def ok(pid: str) -> dict:
    return {"prompt_id": pid, "ok": True, "output": {"pid": pid}, "error": None, "ttft_s": 0.1,
            "total_s": 0.2, "prompt_tokens": 100, "cached_tokens": 90, "completion_tokens": 5}


def analyze_input(p: PromptPlan, seg=0, mode="bundle") -> AnalyzeInput:
    v = VideoInput("/in.mp4", "vid", Q, analysis_mode=mode)
    return AnalyzeInput(SegmentInput(v, seg, f"/seg/{seg}.mp4"), p,
                        FramesOut(["/f/a.png", "/f/b.png"],
                                  {"kept": 2, "segment_start_s": 61.5, "frame_times": [0.0, 1.0]}))


# ------------------------------------------------------------------ SegmentRunner (no Temporal)
class Rec:
    def __init__(self):
        self.published: list[dict] = []
        self.beats: list[list] = []

    async def publish(self, d):
        self.published.append(d)

    def heartbeat(self, d):
        self.beats.append(d)


async def nosleep(_):
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_runner_happy_path_metrics_and_order():
    rec, calls = Rec(), []

    async def call(p):
        calls.append(p.id if p else "_warm")
        return ok(p.id) if p else ok("_warm")

    p = plan(PromptSpec("a", 1, "qa", 10), PromptSpec("b", 2, "qb", 10))
    out = await SegmentRunner(analyze_input(p, seg=3), rec.publish, rec.heartbeat, sleep=nosleep).run_bundle(call)
    assert calls[0] == "_warm" and calls.count("_warm") == 1
    assert out.status == "complete" and [r.prompt_id for r in out.results] == ["a", "b"]
    r = out.results[1]
    assert r.idempotency_key == idempotency_key("vid", 3, "b", 2, MODEL)
    assert r.metrics["segment_start_s"] == 61.5 and r.metrics["frame_times"] == [0.0, 1.0]
    assert r.metrics["cached_tokens"] == 90
    assert len(rec.published) == 2 and len(rec.beats[-1]) == 2


@pytest.mark.asyncio
async def test_runner_transport_retry_then_ok_and_invalid_json_not_retried():
    rec, n = Rec(), {"flaky": 0, "bad": 0}

    async def call(p):
        if p is None:
            return ok("_warm")
        n[p.id] += 1
        if p.id == "flaky" and n["flaky"] < 2:
            return {**ok("flaky"), "ok": False, "error": "APIConnectionError()"}
        if p.id == "bad":
            return {**ok("bad"), "ok": False, "output": "{", "error": "invalid json: x"}
        return ok(p.id)

    p = plan(PromptSpec("flaky", 1, "q", 10), PromptSpec("bad", 1, "q", 10, required=False))
    out = await SegmentRunner(analyze_input(p), rec.publish, rec.heartbeat, sleep=nosleep).run_bundle(call)
    st = {r.prompt_id: r.status for r in out.results}
    assert st == {"flaky": "ok", "bad": "invalid"} and n == {"flaky": 2, "bad": 1}
    assert out.status == "partial"


@pytest.mark.asyncio
async def test_runner_transport_exhausted_is_failed():
    rec = Rec()

    async def call(p):
        return ok("_warm") if p is None else {**ok(p.id), "ok": False, "error": "conn refused"}

    p = plan(PromptSpec("r", 1, "q", 10))
    out = await SegmentRunner(analyze_input(p), rec.publish, rec.heartbeat, max_attempts=3,
                              sleep=nosleep).run_bundle(call)
    assert out.results[0].status == "failed" and out.status == "failed"


@pytest.mark.asyncio
async def test_runner_optional_deadline_is_partial_and_published():
    rec = Rec()

    async def call(p):
        if p is not None and p.id == "opt":
            await asyncio.sleep(5)
        return ok(p.id if p else "_warm")

    p = plan(PromptSpec("req", 1, "q", 10), PromptSpec("opt", 1, "q", 10, required=False, deadline_s=0.05))
    out = await SegmentRunner(analyze_input(p), rec.publish, rec.heartbeat).run_bundle(call)
    assert {r.prompt_id: r.status for r in out.results} == {"req": "ok", "opt": "timed_out"}
    assert out.status == "partial"
    assert {d["prompt_id"] for d in rec.published} == {"req", "opt"}
    assert out.results[1].metrics["segment_start_s"] == 61.5


@pytest.mark.asyncio
async def test_runner_required_failure_others_still_publish():
    rec = Rec()

    async def call(p):
        if p is not None and p.id == "req":
            raise RuntimeError("boom")
        return ok(p.id if p else "_warm")

    p = plan(PromptSpec("req", 1, "q", 10), PromptSpec("other", 1, "q", 10))
    out = await SegmentRunner(analyze_input(p), rec.publish, rec.heartbeat).run_bundle(call)
    assert {r.prompt_id: r.status for r in out.results} == {"req": "failed", "other": "ok"}
    assert out.status == "failed" and len(rec.published) == 2


@pytest.mark.asyncio
async def test_runner_warm_failure_not_fatal():
    rec = Rec()

    async def call(p):
        if p is None:
            raise RuntimeError("warm boom")
        return ok(p.id)

    out = await SegmentRunner(analyze_input(plan(PromptSpec("a", 1, "q", 10))), rec.publish,
                              rec.heartbeat).run_bundle(call)
    assert out.status == "complete"


@pytest.mark.asyncio
async def test_runner_resume_skips_published_prompts():
    p = plan(PromptSpec("a", 1, "q", 10), PromptSpec("b", 1, "q", 10))
    a = analyze_input(p)
    # first attempt: "a" publishes, then publishing "b" crashes the attempt
    rec1, state = Rec(), {"n": 0}

    async def flaky_publish(d):
        if d["prompt_id"] == "b":
            raise ConnectionError("kafka down")
        rec1.published.append(d)

    async def call(p_):
        return ok(p_.id if p_ else "_warm")

    with pytest.raises(ConnectionError):
        await SegmentRunner(a, flaky_publish, rec1.heartbeat).run_bundle(call)
    resume = rec1.beats[-1]
    assert [d["prompt_id"] for d in resume] == ["a"]       # only published prompts recorded

    rec2, calls = Rec(), []

    async def call2(p_):
        calls.append(p_.id if p_ else "_warm")
        return ok(p_.id if p_ else "_warm")

    out = await SegmentRunner(a, rec2.publish, rec2.heartbeat, resumed=resume).run_bundle(call2)
    assert "a" not in calls and "b" in calls
    assert [d["prompt_id"] for d in rec2.published] == ["b"]
    assert out.status == "complete" and [r.prompt_id for r in out.results] == ["a", "b"]


@pytest.mark.asyncio
async def test_runner_fully_resumed_does_no_calls():
    p = plan(PromptSpec("a", 1, "q", 10))
    prev = [dataclasses.asdict(PromptResult(idempotency_key("vid", 0, "a", 1, MODEL), "vid", 0, "a", 1,
                                            MODEL, "ok", {"x": 1}))]
    rec, calls = Rec(), []

    async def call(p_):
        calls.append(p_)
        return ok("x")

    out = await SegmentRunner(analyze_input(p), rec.publish, rec.heartbeat, resumed=prev).run_bundle(call)
    assert calls == [] and rec.published == [] and out.status == "complete"


@pytest.mark.asyncio
async def test_runner_periodic_heartbeat():
    rec = Rec()

    async def call(p):
        await asyncio.sleep(0.1)
        return ok(p.id if p else "_warm")

    await SegmentRunner(analyze_input(plan(PromptSpec("a", 1, "q", 10))), rec.publish, rec.heartbeat,
                        heartbeat_every_s=0.02).run_bundle(call)
    assert len(rec.beats) >= 4          # initial + ticker beats + completion


@pytest.mark.asyncio
async def test_runner_two_tier_maps_results_and_flags_missing():
    rec = Rec()

    async def run():
        return {"mode": "two_tier", "results": [ok("a")]}

    p = plan(PromptSpec("a", 1, "q", 10), PromptSpec("b", 1, "q", 10, required=False))
    out = await SegmentRunner(analyze_input(p, mode="two_tier"), rec.publish, rec.heartbeat).run_two_tier(run)
    st = {r.prompt_id: r.status for r in out.results}
    assert st == {"a": "ok", "b": "failed"} and out.status == "partial"
    assert len(rec.published) == 2


# ------------------------------------------------------------------ analyze_segment activity
class ListPublisher:
    def __init__(self):
        self.items = []

    async def publish(self, d):
        self.items.append(d)


def _env(details=None):
    env = ActivityEnvironment()
    beats = []
    env.on_heartbeat = lambda *d: beats.append(d)
    if details is not None:
        env.info = dataclasses.replace(env.info, heartbeat_details=details)
    return env, beats


@pytest.mark.asyncio
async def test_activity_bundle_mode_and_heartbeat_resume(monkeypatch):
    calls = []

    async def fake_call(client, model, a, p):
        calls.append(p.id if p else "_warm")
        return ok(p.id if p else "_warm")

    monkeypatch.setattr(acts_mod, "_call_prompt", fake_call)
    monkeypatch.setattr(acts_mod, "_client", lambda url: object())
    p = plan(PromptSpec("a", 1, "q", 10), PromptSpec("b", 1, "q", 10))
    done_a = dataclasses.asdict(PromptResult(idempotency_key("vid", 0, "a", 1, MODEL), "vid", 0, "a", 1,
                                             MODEL, "ok", {"pid": "a"}))
    pub = ListPublisher()
    env, beats = _env(details=[[done_a]])
    out = await env.run(Activities(pub).analyze_segment, analyze_input(p))
    assert isinstance(out, SegmentOutcome) and out.status == "complete"
    assert "a" not in calls and "b" in calls
    assert [d["prompt_id"] for d in pub.items] == ["b"]
    assert beats and {d["prompt_id"] for d in beats[-1][0]} == {"a", "b"}


@pytest.mark.asyncio
async def test_activity_two_tier_dispatch(monkeypatch):
    seen = {}

    async def run_two_tier(client, model, reg, frames, **kw):
        seen.update(model=model, ids=[p.id for p in reg.prompts], frames=frames)
        return {"mode": "two_tier", "results": [ok(p.id) for p in reg.prompts]}

    monkeypatch.setitem(sys.modules, "vlmpipe.twotier", pytypes.SimpleNamespace(run_two_tier=run_two_tier))
    monkeypatch.setattr(acts_mod, "_client", lambda url: object())

    async def must_not_call(*a, **k):
        raise AssertionError("bundle path used in two_tier mode")

    monkeypatch.setattr(acts_mod, "_call_prompt", must_not_call)
    pub = ListPublisher()
    env, _ = _env()
    p = plan(PromptSpec("a", 1, "q", 10), PromptSpec("b", 1, "q", 10))
    out = await env.run(Activities(pub).analyze_segment, analyze_input(p, mode="two_tier"))
    assert out.status == "complete" and seen["ids"] == ["a", "b"] and seen["frames"] == ["/f/a.png", "/f/b.png"]
    assert len(pub.items) == 2


@pytest.mark.asyncio
async def test_activity_two_tier_missing_module_non_retryable(monkeypatch):
    monkeypatch.setitem(sys.modules, "vlmpipe.twotier", None)      # import -> ImportError
    env, _ = _env()
    with pytest.raises(ApplicationError) as ei:
        await env.run(Activities(ListPublisher()).analyze_segment,
                      analyze_input(plan(PromptSpec("a", 1, "q", 10)), mode="two_tier"))
    assert ei.value.non_retryable


# ------------------------------------------------------------------ workflow (time-skipping)
class Mocks:
    def __init__(self, p: PromptPlan, n_segments=2, frames=3, analyze_fails=(), decode_s=0.05, analyze_s=0.2):
        self.p, self.n, self.nframes = p, n_segments, frames
        self.analyze_fails, self.decode_s, self.analyze_s = set(analyze_fails), decode_s, analyze_s
        self.events: list[tuple[str, int]] = []
        self.inflight = {"decode": 0, "analyze": 0}
        self.max_inflight = {"decode": 0, "analyze": 0}
        self.analyzed: list[AnalyzeInput] = []

    def _enter(self, k):
        self.inflight[k] += 1
        self.max_inflight[k] = max(self.max_inflight[k], self.inflight[k])

    def acts(self):
        @activity.defn(name="segment_video")
        async def segment_video(v: VideoInput) -> list[str]:
            return [f"/seg/{i}.mp4" for i in range(self.n)]

        @activity.defn(name="load_prompt_plan")
        async def load_prompt_plan(path: str) -> PromptPlan:
            return self.p

        @activity.defn(name="sample_frames")
        async def sample_frames(s: SegmentInput) -> FramesOut:
            self._enter("decode")
            self.events.append(("decode_start", s.segment_index))
            await asyncio.sleep(self.decode_s)
            self.events.append(("decode_end", s.segment_index))
            self.inflight["decode"] -= 1
            return FramesOut([f"/f/{s.segment_index}_{j}.png" for j in range(self.nframes)],
                             {"kept": self.nframes, "segment_start_s": 60.0 * s.segment_index})

        @activity.defn(name="analyze_segment")
        async def analyze_segment(a: AnalyzeInput) -> SegmentOutcome:
            i = a.segment.segment_index
            if i in self.analyze_fails:
                raise ApplicationError("vlm down", non_retryable=True)
            self._enter("analyze")
            self.analyzed.append(a)
            self.events.append(("analyze_start", i))
            await asyncio.sleep(self.analyze_s)
            self.events.append(("analyze_end", i))
            self.inflight["analyze"] -= 1
            v = a.segment.video
            res = [PromptResult(idempotency_key(v.video_id, i, p.id, p.version, v.model), v.video_id, i,
                                p.id, p.version, v.model, "ok", {"pid": p.id}) for p in a.plan.prompts]
            return SegmentOutcome(i, segment_status(a.plan, res), res, a.frames.stats)

        return [segment_video, load_prompt_plan, sample_frames, analyze_segment]


@pytest_asyncio.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as e:
        yield e


async def run(env, mocks: Mocks, **vkw):
    async with Worker(env.client, task_queue=Q, workflows=[VideoAnalysisWorkflow], activities=mocks.acts()):
        v = VideoInput("/in.mp4", f"vid-{uuid.uuid4().hex[:8]}", Q, **vkw)
        return await env.client.execute_workflow(VideoAnalysisWorkflow.run, v, id=v.video_id, task_queue=Q)


@pytest.mark.asyncio
async def test_workflow_happy_path(env):
    m = Mocks(plan(PromptSpec("a", 1, "qa", 10), PromptSpec("b", 2, "qb", 10)))
    out = await run(env, m)
    assert out.status == "complete" and [s.segment_index for s in out.segments] == [0, 1]
    by_seg = {a.segment.segment_index: a for a in m.analyzed}     # analyses run concurrently
    assert len(by_seg) == 2 and by_seg[1].frames.stats["segment_start_s"] == 60.0
    assert by_seg[0].plan.prompts[1].id == "b"


@pytest.mark.asyncio
async def test_workflow_pipelines_decode_ahead_of_analysis(env):
    m = Mocks(plan(PromptSpec("a", 1, "q", 10)), n_segments=4, decode_s=0.05, analyze_s=0.3)
    out = await run(env, m, max_parallel_segments=1, max_parallel_decode=1)
    assert out.status == "complete"
    assert m.max_inflight == {"decode": 1, "analyze": 1}
    ev = m.events
    # segment 1 is decoded while segment 0 is still being analysed
    assert ev.index(("decode_end", 1)) < ev.index(("analyze_end", 0))
    # analyses run in segment order
    assert [i for e, i in ev if e == "analyze_start"] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_workflow_concurrency_bounds(env):
    m = Mocks(plan(PromptSpec("a", 1, "q", 10)), n_segments=6, decode_s=0.05, analyze_s=0.2)
    out = await run(env, m, max_parallel_segments=2, max_parallel_decode=3)
    assert out.status == "complete" and len(out.segments) == 6
    assert m.max_inflight["analyze"] <= 2 and m.max_inflight["decode"] <= 3


@pytest.mark.asyncio
async def test_workflow_segment_analysis_failure_fails_video_not_others(env):
    m = Mocks(plan(PromptSpec("a", 1, "q", 10)), n_segments=3, analyze_fails={1})
    out = await run(env, m)
    assert out.status == "failed"
    assert [s.status for s in out.segments] == ["complete", "failed", "complete"]


@pytest.mark.asyncio
async def test_workflow_empty_frames_skips_analysis(env):
    m = Mocks(plan(PromptSpec("a", 1, "q", 10)), n_segments=1, frames=0)
    out = await run(env, m)
    assert out.segments[0].results == [] and m.analyzed == []


@pytest.mark.asyncio
async def test_workflow_passes_analysis_mode(env):
    m = Mocks(plan(PromptSpec("a", 1, "q", 10)), n_segments=1)
    await run(env, m, analysis_mode="two_tier")
    assert m.analyzed[0].segment.video.analysis_mode == "two_tier"


def test_segment_status_unit():
    p = plan(PromptSpec("r", 1, "q", 1), PromptSpec("o", 1, "q", 1, required=False))
    mk = lambda pid, s: PromptResult("k", "v", 0, pid, 1, "m", s)  # noqa: E731
    assert segment_status(p, [mk("r", "ok"), mk("o", "ok")]) == "complete"
    assert segment_status(p, [mk("r", "ok"), mk("o", "failed")]) == "partial"
    assert segment_status(p, [mk("r", "timed_out"), mk("o", "ok")]) == "failed"
    assert segment_status(p, [mk("o", "ok")]) == "failed"            # required missing
