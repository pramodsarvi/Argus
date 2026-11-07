"""Fleet sizing (§9.5) from spike results.

GPUs = segments/hour x GPU-seconds/segment / 3600 x (1 + headroom)
GPU-seconds/segment = effective tokens per segment / throughput tokens/s per GPU
effective tokens = uncached prompt tokens + completion tokens (cached prefix is ~free).

Throughput measured from single-bundle spike runs UNDERESTIMATES a saturated GPU; pass
--tokens-per-s from a concurrent load test when you have one.

  python -m bench.capacity spike_results.jsonl --mode warm --video-hours-per-hour 20
"""
import argparse
import math

from .analyze_spike import load


def per_segment(records: list[dict], mode: str) -> dict:
    rs = [r for r in records if r["mode"] == mode]
    if not rs:
        raise ValueError(f"no records for mode {mode}")
    eff, wall = [], []
    for r in rs:
        items = list(r["results"]) + ([r["warm"]] if r.get("warm") else [])
        eff.append(sum(x["prompt_tokens"] - x["cached_tokens"] + x["completion_tokens"] for x in items))
        wall.append(r["wall_s"])
    n = len(rs)
    return {"runs": n, "effective_tokens": sum(eff) / n, "wall_s": sum(wall) / n,
            "tokens_per_s": sum(eff) / sum(wall)}


def gpus_needed(video_hours_per_hour: float, segment_seconds: float, effective_tokens: float,
                tokens_per_s: float, headroom: float = 0.3) -> dict:
    segs_per_hour = video_hours_per_hour * 3600 / segment_seconds
    gpu_s_per_seg = effective_tokens / tokens_per_s
    raw = segs_per_hour * gpu_s_per_seg / 3600
    return {"segments_per_hour": segs_per_hour, "gpu_s_per_segment": gpu_s_per_seg,
            "gpus_raw": raw, "gpus": math.ceil(raw * (1 + headroom))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--mode", default="warm")
    ap.add_argument("--video-hours-per-hour", type=float, required=True)
    ap.add_argument("--segment-seconds", type=float, default=60)
    ap.add_argument("--tokens-per-s", type=float, help="override measured throughput")
    ap.add_argument("--headroom", type=float, default=0.3)
    a = ap.parse_args(argv)
    s = per_segment(load(a.jsonl), a.mode)
    tps = a.tokens_per_s or s["tokens_per_s"]
    g = gpus_needed(a.video_hours_per_hour, a.segment_seconds, s["effective_tokens"], tps, a.headroom)
    print(f"mode={a.mode} runs={s['runs']} eff_tokens/segment={s['effective_tokens']:.0f} "
          f"tokens/s/GPU={tps:.0f}{' (override)' if a.tokens_per_s else ' (single-bundle, conservative)'}")
    print(f"segments/h={g['segments_per_hour']:.0f} GPU-s/segment={g['gpu_s_per_segment']:.2f} "
          f"GPUs raw={g['gpus_raw']:.2f} -> with {a.headroom:.0%} headroom: {g['gpus']}")
    return g


if __name__ == "__main__":
    main()
