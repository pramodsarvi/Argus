"""End-to-end latency sweep for ONE video: decode+sample -> model -> all prompt results.

Every repetition starts with a cold prefix cache (needs vLLM VLLM_SERVER_DEV_MODE=1, as
scripts/serve_vllm.sh sets), so numbers are what a *new* video costs.

  python -m bench.latency --video clip.mp4 --workdir /mnt/nvme/frames/lat \
      --frames 8 16 --max-pixels-k 128 256 --modes concurrent warm --reps 3
"""
import argparse
import asyncio
import json
import statistics as st
import tempfile
import time
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from vlmpipe.bundle import run_bundle
from vlmpipe.frames import sample_frames
from vlmpipe.prompts import Registry
from vlmpipe.segment import probe


def reset(base_url: str) -> None:
    root = base_url.rsplit("/v1", 1)[0]
    for ep in ("reset_prefix_cache", "reset_mm_cache"):
        try:
            httpx.post(f"{root}/{ep}", timeout=30)
        except httpx.HTTPError:
            pass


async def one(client, model, reg, video, workdir, n, mp, mode, base_url, prompts, hwaccel):
    reset(base_url)
    dur = probe(video)["duration"]
    out = Path(tempfile.mkdtemp(dir=workdir))
    t0 = time.perf_counter()
    frames, stats = await asyncio.to_thread(
        sample_frames, video, out, n / dur, mp, hwaccel, 0, max_frames=n)   # dedup off: exact n
    t1 = time.perf_counter()
    r = await run_bundle(client, model, reg, [f.path for f in frames], mode=mode, prompts=prompts)
    t2 = time.perf_counter()
    rs = r["results"]
    return {"decode_s": t1 - t0, "model_s": t2 - t1, "total_s": t2 - t0,
            "frames": len(frames), "visual_tokens": stats["prompt_visual_tokens"],
            "ok": sum(x["ok"] for x in rs), "n": len(rs),
            "ttft_s": st.median(x["ttft_s"] for x in rs),
            "first_done_s": min(x["total_s"] for x in rs),
            "out_tokens": sum(x["completion_tokens"] for x in rs)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--workdir", required=True, help="under vLLM --allowed-local-media-path")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--only", nargs="*", help="prompt ids to run (default all)")
    ap.add_argument("--frames", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--max-pixels-k", type=int, nargs="+", default=[128, 256],
                    help="max_pixels in units of 28*28 (~visual tokens per frame)")
    ap.add_argument("--modes", nargs="+", default=["concurrent", "warm"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hwaccel", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Path(a.workdir).mkdir(parents=True, exist_ok=True)
    reg = Registry.load(a.prompts)
    prompts = tuple(p for p in reg.prompts if not a.only or p.id in a.only)
    client = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY")
    await one(client, a.model, reg, a.video, a.workdir, 4, 128 * 784, "concurrent", a.base_url,
              prompts, a.hwaccel)                                   # warm up the server / CUDA graphs
    rows = []
    print(f"{'frames':>6} {'tok/fr':>6} {'mode':>10} | {'total p50':>9} {'min':>6} {'max':>6} | "
          f"{'decode':>6} {'model':>6} {'ttft':>5} {'ok':>5}")
    for n in a.frames:
        for k in a.max_pixels_k:
            for mode in a.modes:
                runs = [await one(client, a.model, reg, a.video, a.workdir, n, k * 784, mode,
                                  a.base_url, prompts, a.hwaccel) for _ in range(a.reps)]
                tot = [r["total_s"] for r in runs]
                row = {"frames": n, "k": k, "mode": mode, "total_p50": st.median(tot),
                       "total_min": min(tot), "total_max": max(tot),
                       "decode": st.median(r["decode_s"] for r in runs),
                       "model": st.median(r["model_s"] for r in runs),
                       "ttft": st.median(r["ttft_s"] for r in runs),
                       "ok": f"{min(r['ok'] for r in runs)}/{runs[0]['n']}",
                       "visual_tokens": runs[0]["visual_tokens"], "out_tokens": runs[0]["out_tokens"]}
                rows.append(row)
                print(f"{n:>6} {k:>6} {mode:>10} | {row['total_p50']:>8.2f}s {row['total_min']:>5.2f}s "
                      f"{row['total_max']:>5.2f}s | {row['decode']:>5.2f}s {row['model']:>5.2f}s "
                      f"{row['ttft']:>4.2f}s {row['ok']:>5}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
