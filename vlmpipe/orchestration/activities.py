"""Activities wrap the existing vlmpipe modules; they run on the node-pinned task queue."""
import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from temporalio import activity
from temporalio.exceptions import ApplicationError

from .segment_runner import SegmentRunner
from .types import (AnalyzeInput, ChunkAnalyzeInput, ChunkOutcome, ChunkSampleInput, FramesOut,
                    PromptPlan, PromptSpec, SegmentInput, SegmentOutcome, VideoInput)


@runtime_checkable
class Publisher(Protocol):
    async def publish(self, result: dict) -> None: ...


def _default_publisher() -> Publisher:
    try:  # provided by vlmpipe.results (separate component)
        from vlmpipe.results.publisher import Publisher as _P  # noqa: F401
        from vlmpipe.results import publisher as mod
        factory = getattr(mod, "from_env", None) or getattr(mod, "default_publisher", None)
        if factory:
            return factory()
    except ImportError:
        pass

    class _LogPublisher:
        async def publish(self, result: dict) -> None:
            activity.logger.info("result %s %s", result.get("idempotency_key"), result.get("status"))
    return _LogPublisher()


class Activities:
    """Instance-bound so the worker injects the publisher (and tests can swap it)."""

    def __init__(self, publisher: Publisher | None = None, heartbeat_every_s: float = 10.0):
        self._publisher = publisher
        self.heartbeat_every_s = heartbeat_every_s

    @property
    def publisher(self) -> Publisher:
        if self._publisher is None:
            self._publisher = _default_publisher()
        return self._publisher

    @activity.defn(name="segment_video")
    async def segment_video(self, v: VideoInput) -> list[str]:
        from vlmpipe.segment import segment
        out = Path(v.workdir) / v.video_id / "segments"
        segs = await asyncio.to_thread(segment, v.video_uri, out, v.segment_seconds)
        if not segs:
            raise ApplicationError(f"no segments produced for {v.video_uri}", non_retryable=True)
        return [str(s) for s in segs]

    @activity.defn(name="sample_frames")
    async def sample_frames(self, s: SegmentInput) -> FramesOut:
        from vlmpipe.frames import sample_frames
        v = s.video
        out = Path(v.workdir) / v.video_id / f"frames_{s.segment_index:05d}"
        frames, stats = await asyncio.to_thread(
            sample_frames, s.segment_path, out, v.fps, v.max_pixels, v.hwaccel,
            strategy=v.sampling, max_frames=v.max_frames)
        from vlmpipe.segment import segment_times
        st = segment_times(s.segment_path)
        stats["segment_start_s"] = st[0] if st else s.segment_index * v.segment_seconds
        return FramesOut([str(f.path) for f in frames], stats)

    @activity.defn(name="load_prompt_plan")
    async def load_prompt_plan(self, prompts_path: str) -> PromptPlan:
        d = yaml.safe_load(Path(prompts_path).read_text())
        ps = [PromptSpec(id=p["id"], version=p["version"], text=p["text"].strip(),
                         max_tokens=p["max_tokens"], schema=p.get("schema", {}),
                         required=p.get("required", True), deadline_s=p.get("deadline_s", 180.0))
              for p in d["prompts"]]
        return PromptPlan(d["shared_system"]["text"].strip(), d["shared_system"]["version"], ps)

    @activity.defn(name="analyze_segment")
    async def analyze_segment(self, a: AnalyzeInput) -> SegmentOutcome:
        """Warm + fan-out (or two-tier) for one segment, in-process. Heartbeats the list of
        already-published results; a retried attempt resumes and skips those prompts."""
        info = activity.info()
        resumed = list(info.heartbeat_details[0]) if info.heartbeat_details else []
        v = a.segment.video
        runner = SegmentRunner(a, self.publisher.publish, lambda d: activity.heartbeat(d),
                               resumed=resumed, heartbeat_every_s=self.heartbeat_every_s)
        if v.analysis_mode == "two_tier":
            try:
                from vlmpipe.twotier import run_two_tier
            except ImportError as e:
                raise ApplicationError(f"two_tier mode unavailable: {e}", non_retryable=True) from e
            client = _client(v.base_url)
            reg = _registry(a.plan)
            return await runner.run_two_tier(
                lambda: run_two_tier(client, v.model, reg, a.frames.frame_paths,
                                     input_mode=v.input_mode, fps=v.fps))
        if v.analysis_mode != "bundle":
            raise ApplicationError(f"unknown analysis_mode {v.analysis_mode!r}", non_retryable=True)
        client = _client(v.base_url)
        from .segment_runner import should_warm
        warm = should_warm("auto", int(a.frames.stats.get("prompt_visual_tokens", 0) or 0),
                           len(a.plan.prompts))
        return await runner.run_bundle(lambda p: _call_prompt(client, v.model, a, p), warm=warm)

    # ---- camera entity workflow ------------------------------------------------------------
    @activity.defn(name="sample_chunk")
    async def sample_chunk(self, s: ChunkSampleInput) -> FramesOut:
        from . import camera
        try:
            return await camera.sample_chunk(s)
        except (KeyError, FileNotFoundError) as e:   # unknown camera / missing file: don't retry
            raise ApplicationError(f"sample_chunk: {e!r}", non_retryable=True) from e

    @activity.defn(name="analyze_chunk")
    async def analyze_chunk(self, a: ChunkAnalyzeInput) -> ChunkOutcome:
        from . import camera
        info = activity.info()
        resumed = list(info.heartbeat_details[0]) if info.heartbeat_details else []
        try:
            return await camera.analyze_chunk(a, self.publisher.publish, lambda d: activity.heartbeat(d),
                                              resumed=resumed, heartbeat_every_s=self.heartbeat_every_s)
        except (ImportError, KeyError, ValueError) as e:
            raise ApplicationError(f"analyze_chunk: {e!r}", non_retryable=True) from e

    @activity.defn(name="mark_chunk_stale")
    async def mark_chunk_stale(self, s: ChunkSampleInput, lag_s: float) -> int:
        from . import camera
        try:
            return await camera.mark_chunk_stale(s, self.publisher.publish, lag_s)
        except (ImportError, KeyError) as e:
            raise ApplicationError(f"mark_chunk_stale: {e!r}", non_retryable=True) from e

    def all(self) -> list:
        return [self.segment_video, self.sample_frames, self.load_prompt_plan, self.analyze_segment,
                self.sample_chunk, self.analyze_chunk, self.mark_chunk_stale]


def _client(base_url: str):
    from vlmpipe.clients import get_client
    return get_client(base_url)


def _registry(plan: PromptPlan):
    from vlmpipe.prompts import Prompt, Registry
    return Registry(plan.system_text, plan.system_version,
                    tuple(Prompt(p.id, p.version, p.text, p.max_tokens, p.schema) for p in plan.prompts))


async def _call_prompt(client, model: str, a: AnalyzeInput, p: PromptSpec | None) -> dict:
    from vlmpipe.bundle import _call
    from vlmpipe.prompts import build_messages
    v = a.segment.video
    msgs = build_messages(a.plan.system_text, a.frames.frame_paths, p.text if p else None,
                          input_mode=v.input_mode, fps=v.fps)
    r = await _call(client, model, msgs, p.max_tokens if p else 1, p.id if p else "_warm",
                    p.schema if p else None, p.deadline_s if p else 120.0)
    return asdict(r)
