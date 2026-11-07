"""Camera orchestration: activity logic for the CameraWorkflow entity + `submit_chunk` for ingest.

One camera produces one ~3-min chunk every ~3 min. Each camera has a long-running CameraWorkflow
(id ``camera-<camera_id>``) on the node queue that holds its files; ingest signals it per chunk.
The workflow itself lives in workflows.py (deterministic code only); everything with I/O is here.

Contracts with sibling components (imported lazily):
  vlmpipe.promptsets.PromptSetRegistry.load(path).for_camera(camera_id) -> CameraPromptSet
      (camera_id, set_id, set_version, registry: vlmpipe.prompts.Registry, routing, sampling)
  vlmpipe.twotier.run_two_tier(..., previous_description=, routing_overrides=)
"""
import asyncio
import dataclasses
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .segment_runner import SegmentRunner, should_warm
from .types import (DEFAULT_CHUNK_SAMPLING, PROFILES, AnalyzeInput, CameraConfig, CameraState, ChunkAnalyzeInput,
                    ChunkInput, ChunkOutcome, ChunkSampleInput, FramesOut, PromptPlan, PromptResult,
                    PromptSpec, SegmentInput, VideoInput, chunk_idempotency_key)

# ---------------------------------------------------------------------------------------------
# Prompt-set resolution (hot reload, one registry per process per path)
# ---------------------------------------------------------------------------------------------
_REGISTRIES: dict[str, Any] = {}


def resolve_prompt_set(path: str, camera_id: str):
    """Return the CameraPromptSet for this camera, reloading the registry file if it changed."""
    from vlmpipe.promptsets import PromptSetRegistry
    reg = _REGISTRIES.get(path)
    if reg is None:
        reg = PromptSetRegistry.load(path)
    else:
        newer = reg.reload_if_changed()
        if newer is not None and hasattr(newer, "for_camera"):   # tolerate return-new or mutate-in-place
            reg = newer
    _REGISTRIES[path] = reg
    return reg.for_camera(camera_id)


def plan_from_registry(reg, degraded: bool = False) -> PromptPlan:
    specs = [PromptSpec(p.id, p.version, p.text, p.max_tokens, dict(p.schema or {}),
                        required=getattr(p, "required", True),
                        deadline_s=getattr(p, "deadline_s", 180.0))
             for p in reg.prompts]
    if degraded:                       # under backlog: required prompts only
        specs = [s for s in specs if s.required]
    return PromptPlan(reg.system_text, reg.system_version, specs)


def sampling_params(cps_sampling: dict | None, degraded: bool, profile: str = "balanced") -> dict:
    p = {**PROFILES.get(profile, DEFAULT_CHUNK_SAMPLING), **(cps_sampling or {})}
    if degraded and p.get("max_frames"):
        p["max_frames"] = max(4, int(p["max_frames"]) // 2)
    return p


def _set_tag(cps) -> str:
    return f"{cps.set_id}@{cps.set_version}"


# ---------------------------------------------------------------------------------------------
# Resource limits (per worker process; sized for shared enterprise VMs, override via env)
#   SAMPLE_CONCURRENCY  parallel ffmpeg samplings (CPU-bound)        default: min(4, cpus // 2)
#   VLM_MAX_INFLIGHT    chunks analysed at once against the replica   default: 4
# Without these, N cameras whose chunks land together start N ffmpegs and N prompt fan-outs at
# once, which thrashes the CPU and the replica's KV/prefix cache.
# ---------------------------------------------------------------------------------------------
_LIMITS: dict[tuple[str, int], asyncio.Semaphore] = {}


def _limit(name: str, default: int) -> asyncio.Semaphore:
    key = (name, id(asyncio.get_running_loop()))      # semaphores must not cross event loops
    sem = _LIMITS.get(key)
    if sem is None:
        if len(_LIMITS) > 16:
            _LIMITS.clear()
        env = os.environ.get(name.upper())
        sem = _LIMITS[key] = asyncio.Semaphore(max(1, int(env) if env else default))
    return sem


def sample_limit() -> asyncio.Semaphore:
    return _limit("sample_concurrency", max(1, min(4, (os.cpu_count() or 2) // 2)))


def vlm_limit() -> asyncio.Semaphore:
    return _limit("vlm_max_inflight", 4)


# ---------------------------------------------------------------------------------------------
# Runner: SegmentRunner semantics (heartbeat resume, publish-per-prompt, required/optional) with
# camera-scoped idempotency keys and metrics.
# ---------------------------------------------------------------------------------------------
class ChunkRunner(SegmentRunner):
    def __init__(self, a: ChunkAnalyzeInput, plan: PromptPlan, cps, publish, heartbeat,
                 resumed: list[dict] | None = None, **kw):
        cfg, c = a.config, a.chunk
        v = VideoInput(c.chunk_path, cfg.camera_id, cfg.node_queue, model=cfg.model,
                       base_url=cfg.base_url, input_mode=cfg.input_mode,
                       analysis_mode=cfg.analysis_mode)
        super().__init__(AnalyzeInput(SegmentInput(v, int(c.chunk_start_ts), c.chunk_path), plan, a.frames),
                         publish, heartbeat, resumed=resumed, **kw)
        self.chunk, self.cfg, self.cps = c, cfg, cps
        self.timing = chunk_metrics(a, cps)

    def _base(self, p: PromptSpec) -> dict:
        return chunk_base(self.cfg, self.chunk, self.cps, p)


def chunk_metrics(a: ChunkAnalyzeInput, cps) -> dict:
    st = a.frames.stats
    return {"camera_id": a.config.camera_id, "chunk_start_ts": a.chunk.chunk_start_ts,
            # The chunk IS the segment: offset 0. Absolute time = chunk_start_ts + frame_time
            # (results.models.ResultRecord.base_ts adds chunk_start_ts + segment_start_s).
            "segment_start_s": 0.0,
            "frame_times": list(st.get("frame_times", [])),
            "prompt_set": cps.set_id, "prompt_set_version": cps.set_version,
            "degraded": a.degraded}


def chunk_base(cfg: CameraConfig, c: ChunkInput, cps, p: PromptSpec) -> dict:
    return dict(idempotency_key=chunk_idempotency_key(cfg.camera_id, c.chunk_start_ts, p.id, p.version,
                                                      cps.set_id, cps.set_version, cfg.model),
                video_id=cfg.camera_id, segment_index=int(c.chunk_start_ts), prompt_id=p.id,
                prompt_version=p.version, model=cfg.model)


# ---------------------------------------------------------------------------------------------
# Activity bodies (wrapped by Activities in activities.py)
# ---------------------------------------------------------------------------------------------
async def sample_chunk(s: ChunkSampleInput) -> FramesOut:
    from vlmpipe.frames import sample_frames
    cfg, c = s.config, s.chunk
    try:
        cps_sampling = resolve_prompt_set(cfg.promptsets_path, cfg.camera_id).sampling
    except ImportError:
        cps_sampling = None
    p = sampling_params(cps_sampling, s.degraded, cfg.profile)
    out = Path(cfg.workdir) / cfg.camera_id / f"{c.chunk_start_ts:.0f}"
    kw = {k: p[k] for k in ("strategy", "scene_threshold", "min_fps", "max_fps", "max_frames") if k in p}
    async with sample_limit():
        frames, stats = await asyncio.to_thread(
            sample_frames, c.chunk_path, out, p.get("fps", 1.0), p["max_pixels"], cfg.hwaccel, **kw)
    stats.update(camera_id=cfg.camera_id, chunk_start_ts=c.chunk_start_ts,
                 segment_start_s=0.0, degraded=s.degraded)
    return FramesOut([str(f.path) for f in frames], stats)


async def analyze_chunk(a: ChunkAnalyzeInput, publish: Callable, heartbeat: Callable,
                        resumed: list[dict] | None = None, heartbeat_every_s: float = 10.0,
                        client_factory: Callable | None = None) -> ChunkOutcome:
    async with vlm_limit():
        out = await _analyze_chunk(a, publish, heartbeat, resumed, heartbeat_every_s, client_factory)
    if a.config.cleanup_frames:      # only on success: a retried attempt needs the frames
        _remove_frames(a.frames.frame_paths)
    return out


def _remove_frames(paths: list[str]) -> None:
    """Frames are only needed until the replica has read them (the retention sweep is the
    fallback). Removes the per-chunk directory if it ends up empty."""
    dirs = set()
    for f in paths:
        try:
            Path(f).unlink()
        except OSError:
            pass
        dirs.add(Path(f).parent)
    for d in dirs:
        try:
            d.rmdir()
        except OSError:
            pass


async def _analyze_chunk(a: ChunkAnalyzeInput, publish: Callable, heartbeat: Callable,
                         resumed, heartbeat_every_s, client_factory) -> ChunkOutcome:
    cfg = a.config
    cps = resolve_prompt_set(cfg.promptsets_path, cfg.camera_id)
    plan = plan_from_registry(cps.registry, a.degraded)
    runner = ChunkRunner(a, plan, cps, publish, heartbeat, resumed=resumed,
                         heartbeat_every_s=heartbeat_every_s)
    client = (client_factory or _client)(cfg.base_url)
    description: dict | None = None
    if a.degraded:
        # Tell the results assembler the dropped optional prompts will never arrive, so it
        # doesn't hold the chunk until its timeout (alerts matter most when we're behind).
        kept_ids = {s.id for s in plan.prompts}
        for p in plan_from_registry(cps.registry).prompts:
            if p.id not in kept_ids:
                await publish(asdict(PromptResult(
                    **chunk_base(cfg, a.chunk, cps, p), status="skipped",
                    error="skipped: degraded mode (optional prompt)", metrics=chunk_metrics(a, cps))))

    if cfg.analysis_mode == "two_tier":
        from vlmpipe.twotier import run_two_tier
        kept = {s.id for s in plan.prompts}
        reg = dataclasses.replace(cps.registry,
                                  prompts=tuple(p for p in cps.registry.prompts if p.id in kept))

        async def run() -> dict:
            nonlocal description
            out = await run_two_tier(client, cfg.model, reg, a.frames.frame_paths,
                                     input_mode=cfg.input_mode,
                                     fps=float(a.frames.stats.get("fps", 1.0) or 1.0),
                                     previous_description=a.previous_description,
                                     routing_overrides=cps.routing or None,
                                     on_result=runner.emit_two_tier,
                                     deadlines={s.id: s.deadline_s for s in plan.prompts})
            d = out.get("description")
            description = d if isinstance(d, dict) else None
            return out
        seg = await runner.run_two_tier(run)
    elif cfg.analysis_mode == "bundle":
        from .activities import _call_prompt
        v = runner.a.segment.video
        ai = AnalyzeInput(SegmentInput(v, int(a.chunk.chunk_start_ts), a.chunk.chunk_path), plan, a.frames)
        warm = should_warm(cfg.warm, int(a.frames.stats.get("prompt_visual_tokens", 0) or 0),
                           len(plan.prompts))
        seg = await runner.run_bundle(lambda p: _call_prompt(client, cfg.model, ai, p), warm=warm)
    else:
        raise ValueError(f"unknown analysis_mode {cfg.analysis_mode!r}")

    return ChunkOutcome(a.chunk.chunk_start_ts, seg.status, seg.results, a.frames.stats,
                        cap_description(description, cfg.max_description_bytes), _set_tag(cps))


def cap_description(d: dict | None, max_bytes: int) -> dict | None:
    """Carried in workflow state/history: drop it rather than bloat history."""
    if d is None:
        return None
    return d if len(json.dumps(d, default=str)) <= max_bytes else None


async def mark_chunk_stale(s: ChunkSampleInput, publish: Callable, lag_s: float) -> int:
    """Publish one `stale` record per prompt so consumers see the gap explicitly."""
    cfg, c = s.config, s.chunk
    cps = resolve_prompt_set(cfg.promptsets_path, cfg.camera_id)
    n = 0
    for p in plan_from_registry(cps.registry).prompts:
        r = PromptResult(**chunk_base(cfg, c, cps, p), status="stale",
                         error=f"chunk lag {lag_s:.0f}s exceeded stale threshold",
                         metrics={"camera_id": cfg.camera_id, "chunk_start_ts": c.chunk_start_ts,
                                  "segment_start_s": 0.0, "frame_times": [],
                                  "prompt_set": cps.set_id, "prompt_set_version": cps.set_version,
                                  "lag_s": lag_s})
        await publish(asdict(r))
        n += 1
    return n


def _client(base_url: str):
    from vlmpipe.clients import get_client
    return get_client(base_url)


# ---------------------------------------------------------------------------------------------
# Ingest entry point
# ---------------------------------------------------------------------------------------------
def default_config(camera_id: str, node: str, **overrides) -> CameraConfig:
    env = {k: v for k, v in {
        "model": os.environ.get("MODEL"),
        "base_url": os.environ.get("VLLM_BASE_URL"),
        "promptsets_path": os.environ.get("PROMPTSETS_PATH"),
        "workdir": os.environ.get("FRAMES_DIR"),
        "analysis_mode": os.environ.get("ANALYSIS_MODE"),
        "profile": os.environ.get("PROFILE"),
    }.items() if v}
    if os.environ.get("CHUNK_SECONDS"):
        env["chunk_seconds"] = float(os.environ["CHUNK_SECONDS"])
    return CameraConfig(camera_id, f"node-{node}", **{**env, **overrides})


async def submit_chunk(client, camera_id: str, chunk_path: str, chunk_start_ts: float, node: str,
                       duration_s: float | None = None, config: CameraConfig | None = None) -> None:
    """Signal-with-start: starts CameraWorkflow `camera-<id>` on `node-<node>` if not running,
    then delivers the chunk. Duplicate chunks (same start ts) are ignored by the workflow."""
    from .workflows import CameraWorkflow
    cfg = config or default_config(camera_id, node)
    await client.start_workflow(
        CameraWorkflow.run, CameraState(cfg), id=f"camera-{camera_id}", task_queue=f"node-{node}",
        start_signal="chunk",
        start_signal_args=[ChunkInput(camera_id, str(chunk_path), float(chunk_start_ts), duration_s)])
