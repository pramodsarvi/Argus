"""Regression gate: compare a report against a baseline. Exit 1 on regression.

  python -m eval.gate --report new.json --baseline base.json [--thresholds th.json]
thresholds JSON: {"default": {"max_drop": 0.02}, "ocr": {"max_drop": 0.01, "min_score": 0.8}}
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT = {"max_drop": 0.02}


def check(report: dict, baseline: dict, thresholds: dict | None = None) -> list[str]:
    th = thresholds or {}
    default = {**DEFAULT, **th.get("default", {})}
    fails = []
    new, base = report["per_prompt"], baseline["per_prompt"]
    for pid, b in base.items():
        t = {**default, **th.get(pid, {})}
        if pid not in new:
            fails.append(f"{pid}: missing from report")
            continue
        n, o = new[pid]["mean_score"], b["mean_score"]
        if o - n > t["max_drop"]:
            fails.append(f"{pid}: {o:.3f} -> {n:.3f} (drop {o - n:.3f} > {t['max_drop']})")
        if "min_score" in t and n < t["min_score"]:
            fails.append(f"{pid}: {n:.3f} < min_score {t['min_score']}")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--thresholds")
    a = ap.parse_args(argv)
    th = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None
    fails = check(json.loads(Path(a.report).read_text()), json.loads(Path(a.baseline).read_text()), th)
    for f in fails:
        print("REGRESSION", f)
    if not fails:
        print("PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
