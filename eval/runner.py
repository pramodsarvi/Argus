"""Run the eval dataset through vLLM and write a report.

Dataset: JSONL rows {"segment_path": ..., "prompt_id": ..., "expected": {...}}
  python -m eval.runner --dataset eval.jsonl --base-url http://localhost:8000/v1 \
      --workdir /mnt/nvme/frames/eval --out report.json
"""
import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .scorers import score


def load_dataset(path: str | Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        for k in ("segment_path", "prompt_id", "expected"):
            if k not in r:
                raise ValueError(f"line {i}: missing {k}")
        rows.append(r)
    return rows


def aggregate(scored: list[dict]) -> dict:
    by = defaultdict(list)
    for s in scored:
        by[s["prompt_id"]].append(s)
    per = {}
    for pid, xs in sorted(by.items()):
        per[pid] = {"n": len(xs), "mean_score": statistics.fmean(x["score"] for x in xs),
                    "invalid_outputs": sum(1 for x in xs if "error" in x)}
    return per


async def run(rows, base_url, model, prompts_path, workdir, mode, fps, max_pixels):
    from openai import AsyncOpenAI
    from vlmpipe.bundle import run_bundle
    from vlmpipe.frames import sample_frames
    from vlmpipe.prompts import Registry

    reg = Registry.load(prompts_path)
    pmap = {p.id: p for p in reg.prompts}
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment_path"]].append(r)
    scored = []
    for si, (seg, rs) in enumerate(by_seg.items()):
        frames, _ = sample_frames(seg, Path(workdir) / f"seg{si:05d}", fps, max_pixels)
        prompts = tuple(pmap[pid] for pid in dict.fromkeys(r["prompt_id"] for r in rs))
        out = await run_bundle(client, model, reg, [f.path for f in frames], mode=mode, prompts=prompts)
        if mode == "merged":
            m = out["results"][0]["output"]
            preds = {p.id: (m.get(p.id) if isinstance(m, dict) else None) for p in prompts}
        else:
            preds = {r["prompt_id"]: r["output"] for r in out["results"]}
        for r in rs:
            s = score(r["prompt_id"], preds.get(r["prompt_id"]), r["expected"])
            scored.append({"segment_path": seg, "prompt_id": r["prompt_id"], **s})
    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", default="warm", choices=["warm", "concurrent", "merged"])
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--out", default="eval_report.json")
    a = ap.parse_args()
    scored = asyncio.run(run(load_dataset(a.dataset), a.base_url, a.model, a.prompts,
                             a.workdir, a.mode, a.fps, a.max_pixels))
    report = {"ts": time.time(), "model": a.model, "mode": a.mode, "max_pixels": a.max_pixels,
              "per_prompt": aggregate(scored), "items": scored}
    Path(a.out).write_text(json.dumps(report, indent=2))
    for pid, m in report["per_prompt"].items():
        print(f"{pid:10s} n={m['n']:4d} score={m['mean_score']:.3f} invalid={m['invalid_outputs']}")


if __name__ == "__main__":
    main()
