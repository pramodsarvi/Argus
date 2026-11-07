"""Durable control flow. Deterministic: no I/O here, only activity calls.

Granularity (design §10.4): one `analyze_segment` activity per segment (warm + fan-out in-process),
not one activity per prompt. Pipelining (§10.8): sampling of segment N+1 runs while segment N is
being analysed, bounded separately by max_parallel_decode and max_parallel_segments.
"""
import asyncio
import json
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from .segment_runner import segment_status as _segment_status
    from .types import (AnalyzeInput, CameraState, CameraSummary, ChunkAnalyzeInput, ChunkInput,
                        ChunkOutcome, ChunkRecord, ChunkSampleInput, FramesOut, PromptPlan,
                        PromptResult, SegmentInput, SegmentOutcome, VideoInput, VideoOutcome)

IO_RETRY = RetryPolicy(initial_interval=timedelta(seconds=2), backoff_coefficient=2.0,
                       maximum_interval=timedelta(seconds=30), maximum_attempts=3)
ANALYZE_RETRY = RetryPolicy(initial_interval=timedelta(seconds=5), backoff_coefficient=2.0,
                            maximum_interval=timedelta(seconds=60), maximum_attempts=3,
                            non_retryable_error_types=["ValueError"])
HEARTBEAT_TIMEOUT = timedelta(seconds=60)
WARM_BUDGET_S = 120.0


def segment_status(plan: PromptPlan, results: list[PromptResult]) -> str:
    return _segment_status(plan.prompts, results)


def analyze_timeout(plan: PromptPlan) -> timedelta:
    # prompts run concurrently: budget = warm + slowest prompt deadline + slack
    longest = max((p.deadline_s for p in plan.prompts), default=60.0)
    return timedelta(seconds=WARM_BUDGET_S + longest * 1.5 + 60)


@workflow.defn
class VideoAnalysisWorkflow:
    @workflow.run
    async def run(self, v: VideoInput) -> VideoOutcome:
        segs: list[str] = await workflow.execute_activity(
            "segment_video", v, result_type=list[str], task_queue=v.node_queue,
            start_to_close_timeout=timedelta(minutes=15), retry_policy=IO_RETRY)
        plan: PromptPlan = await workflow.execute_activity(
            "load_prompt_plan", v.prompts_path, result_type=PromptPlan, task_queue=v.node_queue,
            start_to_close_timeout=timedelta(seconds=30), retry_policy=IO_RETRY)

        decode_sem = asyncio.Semaphore(max(1, v.max_parallel_decode))
        analyze_sem = asyncio.Semaphore(max(1, v.max_parallel_segments))

        async def segment(i: int, path: str) -> SegmentOutcome:
            s = SegmentInput(v, i, path)
            async with decode_sem:
                frames: FramesOut = await workflow.execute_activity(
                    "sample_frames", s, result_type=FramesOut, task_queue=v.node_queue,
                    activity_id=f"sample-{i:05d}",
                    start_to_close_timeout=timedelta(minutes=10), retry_policy=IO_RETRY)
            if not frames.frame_paths:
                return SegmentOutcome(i, "complete", [], frames.stats)
            async with analyze_sem:
                try:
                    return await workflow.execute_activity(
                        "analyze_segment", AnalyzeInput(s, plan, frames), result_type=SegmentOutcome,
                        task_queue=v.node_queue, activity_id=f"analyze-{i:05d}",
                        start_to_close_timeout=analyze_timeout(plan),
                        heartbeat_timeout=HEARTBEAT_TIMEOUT, retry_policy=ANALYZE_RETRY)
                except ActivityError as e:
                    workflow.logger.error("segment %d analysis failed: %s", i, e.cause or e)
                    return SegmentOutcome(i, "failed", [], frames.stats)

        outcomes = list(await asyncio.gather(*[segment(i, p) for i, p in enumerate(segs)]))
        sts = {o.status for o in outcomes}
        status = "failed" if "failed" in sts else ("partial" if "partial" in sts else "complete")
        return VideoOutcome(v.video_id, status, outcomes)


# ----------------------------------------------------------------------------------------------
# Camera entity workflow
# ----------------------------------------------------------------------------------------------
SEEN_WINDOW = 500          # dedupe window (chunk start timestamps)
RECENT_WINDOW = 50         # chunk records kept for the status query
CHUNK_ANALYZE_TIMEOUT = timedelta(minutes=10)


@workflow.defn
class CameraWorkflow:
    """One long-running workflow per camera (id ``camera-<camera_id>``).

    * ``chunk`` signals enqueue chunks; processing is strictly one-at-a-time in start-ts order,
      duplicates (same start ts) are ignored.
    * Real-time guard: lag = now - chunk end. lag > degrade_after -> fewer frames, required prompts
      only; lag > stale_after -> publish ``stale`` records and skip.
    * The previous chunk's description is carried as context for the next (two-tier mode).
    * continue-as-new every N chunks (or when Temporal suggests it), carrying queue + context.
    """

    @workflow.init
    def __init__(self, state: CameraState) -> None:
        self.s = state
        self.s.pending = sorted(self.s.pending, key=lambda c: c.chunk_start_ts)

    @workflow.signal(name="chunk")
    def chunk(self, c: ChunkInput) -> None:
        s = self.s
        if c.chunk_start_ts in s.seen_ts or any(p.chunk_start_ts == c.chunk_start_ts for p in s.pending):
            return
        s.pending.append(c)
        s.pending.sort(key=lambda x: x.chunk_start_ts)

    @workflow.query(name="status")
    def status(self) -> CameraSummary:
        return CameraSummary(self.s.config.camera_id, self.s.total_chunks, list(self.s.recent))

    @workflow.query(name="pending")
    def pending(self) -> list[float]:
        return [c.chunk_start_ts for c in self.s.pending]

    @workflow.run
    async def run(self, state: CameraState) -> CameraSummary:
        s, cfg = self.s, self.s.config
        done_this_run = 0
        while True:
            if cfg.max_chunks is not None and s.total_chunks >= cfg.max_chunks:
                return CameraSummary(cfg.camera_id, s.total_chunks, list(s.recent))
            if done_this_run >= cfg.continue_as_new_after or workflow.info().is_continue_as_new_suggested():
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(s)
            await workflow.wait_condition(lambda: bool(s.pending))
            if cfg.reorder_window_s > 0:
                await workflow.sleep(cfg.reorder_window_s)      # let near-simultaneous arrivals sort
            c = s.pending.pop(0)
            await self._process(c)
            done_this_run += 1

    async def _process(self, c: ChunkInput) -> None:
        s, cfg = self.s, self.s.config
        dur = c.duration_s or cfg.chunk_seconds
        lag = workflow.now().timestamp() - (c.chunk_start_ts + dur)
        s.seen_ts.append(c.chunk_start_ts)
        del s.seen_ts[:-SEEN_WINDOW]
        s.total_chunks += 1
        q = cfg.node_queue

        if lag > cfg.stale_after_chunks * dur:
            try:
                await workflow.execute_activity(
                    "mark_chunk_stale", args=[ChunkSampleInput(cfg, c), lag], result_type=int,
                    task_queue=q, start_to_close_timeout=timedelta(minutes=2), retry_policy=IO_RETRY)
            except ActivityError as e:
                workflow.logger.error("stale marking failed for %s: %s", c.chunk_start_ts, e.cause or e)
            self._record(c, "stale", False, lag)
            return

        degraded = lag > cfg.degrade_after_chunks * dur
        try:
            frames: FramesOut = await workflow.execute_activity(
                "sample_chunk", ChunkSampleInput(cfg, c, degraded), result_type=FramesOut, task_queue=q,
                activity_id=f"sample-{c.chunk_start_ts:.0f}",
                start_to_close_timeout=timedelta(minutes=5), retry_policy=IO_RETRY)
        except ActivityError as e:
            workflow.logger.error("sampling failed for %s: %s", c.chunk_start_ts, e.cause or e)
            self._record(c, "failed", degraded, lag)
            return
        if not frames.frame_paths:
            self._record(c, "empty", degraded, lag)
            return

        in_order = s.last_ts is None or c.chunk_start_ts > s.last_ts
        prev = s.previous_description if in_order else None
        try:
            out: ChunkOutcome = await workflow.execute_activity(
                "analyze_chunk", ChunkAnalyzeInput(cfg, c, frames, prev, degraded),
                result_type=ChunkOutcome, task_queue=q, activity_id=f"analyze-{c.chunk_start_ts:.0f}",
                start_to_close_timeout=CHUNK_ANALYZE_TIMEOUT, heartbeat_timeout=HEARTBEAT_TIMEOUT,
                retry_policy=ANALYZE_RETRY)
        except ActivityError as e:
            workflow.logger.error("analysis failed for %s: %s", c.chunk_start_ts, e.cause or e)
            self._record(c, "failed", degraded, lag)
            return
        if in_order:
            s.last_ts = c.chunk_start_ts
            d = out.description
            s.previous_description = (d if d is not None and
                                      len(json.dumps(d, default=str)) <= cfg.max_description_bytes else None)
        self._record(c, out.status, degraded, lag)

    def _record(self, c: ChunkInput, status: str, degraded: bool, lag: float) -> None:
        r = self.s.recent
        r.append(ChunkRecord(c.chunk_start_ts, status, degraded, round(lag, 3)))
        del r[:-RECENT_WINDOW]
