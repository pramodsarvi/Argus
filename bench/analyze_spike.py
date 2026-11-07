"""Aggregate spike_results.jsonl by mode -> markdown table.

  python -m bench.analyze_spike spike_results.jsonl
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def aggregate(records: list[dict]) -> dict:
    by = defaultdict(list)
    for r in records:
        by[(r.get("model", "?"), r["mode"])].append(r)
    out = {}
    for key, rs in sorted(by.items()):
        res = [x for r in rs for x in r["results"]]
        ttft = [x["ttft_s"] for x in res]
        pt = sum(x["prompt_tokens"] for x in res)
        ct = sum(x["cached_tokens"] for x in res)
        out[key] = {"runs": len(rs), "requests": len(res),
                    "median_ttft_s": statistics.median(ttft) if ttft else float("nan"),
                    "p95_ttft_s": pct(ttft, 0.95),
                    "median_wall_s": statistics.median(r["wall_s"] for r in rs),
                    "cache_hit": ct / pt if pt else 0.0,
                    "ok_rate": sum(x["ok"] for x in res) / len(res) if res else 0.0}
    return out


def markdown(agg: dict) -> str:
    lines = ["| model | mode | runs | median TTFT s | p95 TTFT s | median wall s | cache hit | ok rate |",
             "|---|---|---|---|---|---|---|---|"]
    for (model, mode), m in agg.items():
        lines.append(f"| {model} | {mode} | {m['runs']} | {m['median_ttft_s']:.3f} | {m['p95_ttft_s']:.3f} "
                     f"| {m['median_wall_s']:.3f} | {m['cache_hit']:.1%} | {m['ok_rate']:.1%} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    a = ap.parse_args(argv)
    print(markdown(aggregate(load(a.jsonl))))


if __name__ == "__main__":
    main()
