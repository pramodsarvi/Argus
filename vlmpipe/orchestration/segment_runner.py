"""Per-segment analysis, run in-process inside ONE Temporal activity (design §10.4).

Pure asyncio + injected callables, so it is unit-testable without Temporal:
  call(prompt | None) -> bundle.Result-like dict   (None = warm request)
  publish(result_dict) -> awaitable
  heartbeat(list_of_done_result_dicts) -> None     (resume state for a retried activity)
"""
import asyncio
import time
from dataclasses import asdict
from typing import Awaitable, Callable

from .types import AnalyzeInput, PromptResult, PromptSpec, SegmentOutcome, idempotency_key

CallFn = Callable[[PromptSpec | None], Awaitable[dict]]
PublishFn = Callable[[dict], Awaitable[None]]
HeartbeatFn = Callable[[list], None]
TwoTierFn = Callable[[], Awaitable[dict]]

RESULT_METRICS = ("ttft_s", "total_s", "prompt_tokens", "cached_tokens", "completion_tokens")


WARM_MIN_VISUAL_TOKENS = 3000
WARM_MIN_PROMPTS = 3


def should_warm(mode: str, visual_tokens: int, n_prompts: int) -> bool:
    """Measured: the extra warm round trip (~0.7 s) only pays for itself with a large shared
    prefix and several prompts; below that, firing all prompts at once is as fast or faster."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    return visual_tokens >= WARM_MIN_VISUAL_TOKENS and n_prompts >= WARM_MIN_PROMPTS


def segment_status(prompts: list[PromptSpec], results: list[PromptResult]) -> str:
    by_id = {r.prompt_id: r for r in results}
    req_ok = all(p.id in by_id and by_id[p.id].status == "ok" for p in prompts if p.required)
    all_ok = len(results) == len(prompts) and all(r.status == "ok" for r in results)
    return "complete" if all_ok else ("partial" if req_ok else "failed")


def _is_transport_error(r: dict) -> bool:
    err = r.get("error")
    return bool(err) and not str(err).startswith("invalid json")


class SegmentRunner:
    def __init__(self, a: AnalyzeInput, publish: PublishFn, heartbeat: HeartbeatFn,
                 resumed: list[dict] | None = None, max_attempts: int = 3,
                 backoff_s: float = 2.0, heartbeat_every_s: float = 10.0,
                 warm_timeout_s: float = 120.0, sleep=asyncio.sleep):
        self.a, self.publish, self.heartbeat = a, publish, heartbeat
        self.max_attempts, self.backoff_s = max_attempts, backoff_s
        self.heartbeat_every_s, self.warm_timeout_s, self.sleep = heartbeat_every_s, warm_timeout_s, sleep
        self.done: dict[str, PromptResult] = {}
        for d in resumed or []:
            r = PromptResult(**d)
            self.done[r.prompt_id] = r
        v, st = a.segment.video, a.frames.stats
        self.timing = {k: st[k] for k in ("segment_start_s", "frame_times") if k in st}
        self._v = v

    # ---- helpers -------------------------------------------------------------------------
    def _base(self, p: PromptSpec) -> dict:
        v, i = self._v, self.a.segment.segment_index
        return dict(idempotency_key=idempotency_key(v.video_id, i, p.id, p.version, v.model),
                    video_id=v.video_id, segment_index=i, prompt_id=p.id,
                    prompt_version=p.version, model=v.model)

    def _from_call(self, p: PromptSpec, r: dict) -> PromptResult:
        err = str(r.get("error") or "")
        status = ("ok" if r.get("ok") else "timed_out" if err.startswith("deadline ")
                  else "invalid" if not _is_transport_error(r) else "failed")
        return PromptResult(**self._base(p), status=status, output=r.get("output"),
                            error=r.get("error"),
                            metrics={**self.timing, **{k: r[k] for k in RESULT_METRICS if k in r}})

    def _beat(self) -> None:
        self.heartbeat([asdict(r) for r in self.done.values()])

    async def _finish(self, r: PromptResult) -> PromptResult:
        await self.publish(asdict(r))          # publish BEFORE recording, so resume never skips
        self.done[r.prompt_id] = r             # an unpublished prompt
        self._beat()
        return r

    async def _ticker(self) -> None:
        while True:
            await self.sleep(self.heartbeat_every_s)
            self._beat()

    # ---- bundle mode ---------------------------------------------------------------------
    async def _one(self, call: CallFn, p: PromptSpec) -> PromptResult:
        deadline = time.monotonic() + p.deadline_s
        last: dict | None = None
        try:
            for attempt in range(self.max_attempts):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                last = await asyncio.wait_for(call(p), timeout=remaining)
                if not _is_transport_error(last):
                    break
                if attempt + 1 < self.max_attempts:
                    await self.sleep(min(self.backoff_s * 2 ** attempt, max(0.0, deadline - time.monotonic())))
            res = self._from_call(p, last)
        except asyncio.TimeoutError:
            res = PromptResult(**self._base(p), status="timed_out",
                               error=f"deadline {p.deadline_s}s exceeded", metrics=dict(self.timing))
        except Exception as e:  # noqa: BLE001 - per-prompt failure must not sink the segment
            res = PromptResult(**self._base(p), status="failed", error=repr(e), metrics=dict(self.timing))
        return await self._finish(res)

    async def run_bundle(self, call: CallFn, warm: bool = True) -> SegmentOutcome:
        todo = [p for p in self.a.plan.prompts if p.id not in self.done]
        ticker = asyncio.ensure_future(self._ticker())
        try:
            self._beat()
            if todo:
                if warm:
                    try:   # warm the shared prefix once; best-effort
                        await asyncio.wait_for(call(None), timeout=self.warm_timeout_s)
                    except Exception:  # noqa: BLE001
                        pass
                await asyncio.gather(*[self._one(call, p) for p in todo])
        finally:
            ticker.cancel()
        return self._outcome()

    # ---- two-tier mode -------------------------------------------------------------------
    async def emit_two_tier(self, r: dict) -> None:
        """Publish one prompt's result the moment the two-tier run produces it (pass as
        run_two_tier(on_result=...)). Idempotent per prompt."""
        p = next((p for p in self.a.plan.prompts if p.id == r.get("prompt_id")), None)
        if p is not None and p.id not in self.done:
            await self._finish(self._from_call(p, r))

    async def run_two_tier(self, run: TwoTierFn) -> SegmentOutcome:
        """`run` should stream results through `emit_two_tier`; anything it returns but did not
        emit is published at the end. On overall timeout only unfinished prompts are timed_out."""
        todo = [p for p in self.a.plan.prompts if p.id not in self.done]
        ticker = asyncio.ensure_future(self._ticker())
        try:
            self._beat()
            if todo:
                timeout = max(p.deadline_s for p in todo) + self.warm_timeout_s
                by_id: dict = {}
                err = None
                try:
                    out = await asyncio.wait_for(run(), timeout=timeout)
                    by_id = {r.get("prompt_id"): r for r in out.get("results", [])}
                except asyncio.TimeoutError:
                    err = "timed_out"
                for p in todo:
                    if p.id in self.done:
                        continue
                    r = by_id.get(p.id)
                    if r is not None:
                        res = self._from_call(p, r)
                    else:
                        res = PromptResult(**self._base(p), status=err or "failed",
                                           error=err or "missing from two-tier output",
                                           metrics=dict(self.timing))
                    await self._finish(res)
        finally:
            ticker.cancel()
        return self._outcome()

    def _outcome(self) -> SegmentOutcome:
        order = [p.id for p in self.a.plan.prompts]
        results = [self.done[i] for i in order if i in self.done]
        return SegmentOutcome(self.a.segment.segment_index,
                              segment_status(self.a.plan.prompts, results), results, self.a.frames.stats)
