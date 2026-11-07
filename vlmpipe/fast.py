"""Minimum-latency processing of ONE video file.

    python -m vlmpipe.fast clip.mp4 [--frames 8] [--tokens 128] [--only safety ocr]

Recipe (measured, Qwen2.5-VL-7B FP8, L40S, cold prefix cache, 5 prompts): sample exactly N frames
at ~`tokens` visual tokens each, send them as images, fire all prompts concurrently (a warm request
is added only when the shared prefix is big enough to pay for it), schema-in-prompt JSON.
~1.0 s for 8 frames x 128 tokens; ~1.5 s for 16 x 256.

Frames are written under --workdir, which must be inside the server's --allowed-local-media-path
(default $FRAMES_DIR/fast). Prints one JSON document on stdout and a timing line on stderr.
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .bundle import run_bundle
from .frames import sample_frames
from .orchestration.segment_runner import should_warm
from .prompts import Registry
from .segment import probe

DEFAULT_PROMPTS = Path(__file__).resolve().parent.parent / "configs" / "prompts.yaml"


async def process_video(client, model: str, video: str | Path, reg: Registry, workdir: str | Path,
                        frames: int = 8, tokens: int = 128, only: list[str] | None = None,
                        dedup: int = 0, warm: str = "auto", keep: bool = False,
                        hwaccel: str | None = None) -> dict:
    """Sample -> bundle -> results. Returns {"results", "timing", "frames"}."""
    prompts = tuple(p for p in reg.prompts if not only or p.id in only)
    if not prompts:
        raise ValueError(f"no prompts selected (available: {[p.id for p in reg.prompts]})")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(dir=workdir))
    t0 = time.perf_counter()
    try:
        dur = (await asyncio.to_thread(probe, video))["duration"]
        fr, stats = await asyncio.to_thread(
            sample_frames, video, out, max(frames / max(dur, 0.1), 0.01), tokens * 28 * 28, hwaccel,
            dedup, max_frames=frames)
        t1 = time.perf_counter()
        do_warm = should_warm(warm, stats["prompt_visual_tokens"], len(prompts))
        r = await run_bundle(client, model, reg, [f.path for f in fr],
                             mode="warm" if do_warm else "concurrent", prompts=prompts)
        t2 = time.perf_counter()
    finally:
        if not keep:
            shutil.rmtree(out, ignore_errors=True)
    return {"results": {x["prompt_id"]: x["output"] if x["ok"] else {"error": x["error"]}
                        for x in r["results"]},
            "frames": {"n": len(fr), "times_s": [round(f.t, 2) for f in fr],
                       "visual_tokens": stats["prompt_visual_tokens"]},
            "timing": {"decode_s": round(t1 - t0, 3), "model_s": round(t2 - t1, 3),
                       "total_s": round(t2 - t0, 3), "warm": do_warm,
                       "ok": sum(x["ok"] for x in r["results"]), "n": len(r["results"])}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=8, help="frames to sample (default 8)")
    ap.add_argument("--tokens", type=int, default=128, help="~visual tokens per frame (default 128)")
    ap.add_argument("--only", nargs="*", help="prompt ids (default: all in --prompts)")
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--workdir", default=str(Path(os.environ.get("FRAMES_DIR", "/mnt/nvme/frames")) / "fast"))
    ap.add_argument("--dedup", type=int, default=0, help="dHash distance treated as duplicate (0=off)")
    ap.add_argument("--warm", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--keep", action="store_true", help="keep the sampled frames")
    a = ap.parse_args(argv)

    async def run() -> dict:
        from .clients import get_client
        return await process_video(get_client(a.base_url), a.model, a.video, Registry.load(a.prompts),
                                   a.workdir, a.frames, a.tokens, a.only, a.dedup, a.warm, a.keep)
    res = asyncio.run(run())
    t = res["timing"]
    print(f"total {t['total_s']:.2f}s (decode {t['decode_s']:.2f}s + model {t['model_s']:.2f}s), "
          f"{res['frames']['n']} frames, {t['ok']}/{t['n']} prompts ok", file=sys.stderr)
    json.dump(res, sys.stdout, indent=2)
    print()
    return 0 if t["ok"] == t["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
