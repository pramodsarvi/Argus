"""Week-1 go/no-go spike.

  python -m bench.spike --video sample.mp4 --base-url http://localhost:8000/v1 \
      --model Qwen/Qwen2.5-VL-7B-Instruct --modes concurrent warm merged --repeats 3

Each repeat uses a FRESH segment workdir name but the same frames; between modes we
reset the prefix cache (POST /reset_prefix_cache, needs VLLM_SERVER_DEV_MODE=1) so modes
don't contaminate each other. Writes JSONL to --out.
"""
import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from vlmpipe.bundle import run_bundle
from vlmpipe.frames import sample_frames
from vlmpipe.prompts import Registry
from vlmpipe.segment import segment


def reset_cache(base_url: str) -> bool:
    root = base_url.rsplit("/v1", 1)[0]
    try:
        return httpx.post(f"{root}/reset_prefix_cache", timeout=30).status_code == 200
    except httpx.HTTPError:
        return False


def get_two_tier():
    """vlmpipe.twotier is optional (separate component); same return shape as run_bundle."""
    try:
        from vlmpipe.twotier import run_two_tier
        return run_two_tier
    except ImportError:
        return None


async def run_mode(client, model, reg, paths, mode, input_mode, fps, mm_kwargs):
    if mode == "two_tier":
        fn = get_two_tier()
        if fn is None:
            return None
        return await fn(client, model, reg, paths, input_mode=input_mode, fps=fps,
                        mm_processor_kwargs=mm_kwargs)
    return await run_bundle(client, model, reg, paths, mode=mode, input_mode=input_mode,
                            fps=fps, mm_processor_kwargs=mm_kwargs)


def summarize(run: dict) -> dict:
    rs = run["results"]
    pt = sum(r["prompt_tokens"] for r in rs)
    ct = sum(r["cached_tokens"] for r in rs)
    return {"mode": run["mode"], "input_mode": run.get("input_mode", "images"),
            "prompt_tokens": pt, "wall_s": round(run["wall_s"], 3),
            "warm_s": round(run["warm"]["total_s"], 3) if run["warm"] else None,
            "median_ttft_s": round(statistics.median(r["ttft_s"] for r in rs), 3),
            "cache_hit": round(ct / pt, 3) if pt else None,
            "ok": sum(r["ok"] for r in rs), "n": len(rs)}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--modes", nargs="+", default=["concurrent", "warm", "merged"],
                    choices=["concurrent", "warm", "merged", "two_tier"])
    ap.add_argument("--input-modes", nargs="+", default=["images"], choices=["images", "video"])
    ap.add_argument("--video-mm-fps", type=float, default=None,
                    help="send mm_processor_kwargs={'fps': X} in video mode (vLLM-specific)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--segment-seconds", type=int, default=60)
    ap.add_argument("--hwaccel", default=None, help="e.g. cuda")
    ap.add_argument("--workdir", default=None, help="must be under vLLM --allowed-local-media-path")
    ap.add_argument("--out", default="spike_results.jsonl")
    return ap


async def main() -> None:
    a = build_parser().parse_args()

    work = Path(a.workdir or tempfile.mkdtemp(prefix="vlmspike_"))
    reg = Registry.load(a.prompts)
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY")

    t = time.perf_counter()
    seg = segment(a.video, work / "segments", a.segment_seconds)[0]
    t_seg = time.perf_counter() - t
    t = time.perf_counter()
    frames, fstats = sample_frames(seg, work / "frames", a.fps, a.max_pixels, a.hwaccel)
    t_dec = time.perf_counter() - t
    print(f"segment {seg.name}: split {t_seg:.2f}s, decode+sample {t_dec:.2f}s, {fstats}")

    paths = [f.path for f in frames]
    with open(a.out, "a") as fh:
        for input_mode in a.input_modes:
            mm_kwargs = ({"fps": a.video_mm_fps}
                         if input_mode == "video" and a.video_mm_fps else None)
            for mode in a.modes:
                if mode == "two_tier" and get_two_tier() is None:
                    print("WARN: vlmpipe.twotier not available; skipping two_tier")
                    continue
                for i in range(a.repeats):
                    if not reset_cache(a.base_url):
                        print("WARN: could not reset prefix cache; set VLLM_SERVER_DEV_MODE=1")
                    run = await run_mode(client, a.model, reg, paths, mode, input_mode,
                                         a.fps, mm_kwargs)
                    s = summarize(run)
                    print(json.dumps(s))
                    fh.write(json.dumps({"ts": time.time(), "model": a.model, "frames": fstats,
                                         "decode_s": t_dec, "repeat": i, "summary": s,
                                         **run}) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
