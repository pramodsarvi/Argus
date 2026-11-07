"""Per-video reduce: absolute timestamps, merged outputs, cross-prompt rules.

Frame numbers in model outputs are assumed 1-based positions in the frame list shown to the
model (prompt order). They are mapped via rec.frame_times + rec.segment_start_s, plus
rec.chunk_start_ts for camera records (giving absolute UTC epoch seconds).
"""
from collections import defaultdict
from pathlib import Path

import yaml

from .models import ResultRecord

FRAME_FIELDS = {"frame": "t", "start_frame": "start_t", "end_frame": "end_t"}


def frame_to_abs(rec: ResultRecord, frame: int) -> float | None:
    if not rec.frame_times or not isinstance(frame, int):
        return None
    i = min(max(frame - 1, 0), len(rec.frame_times) - 1)
    return round(rec.base_ts + rec.frame_times[i], 3)   # chunk_start_ts (camera) + local time


def _absolutize(obj, rec: ResultRecord):
    if isinstance(obj, dict):
        out = {k: _absolutize(v, rec) for k, v in obj.items()}
        for f, t in FRAME_FIELDS.items():
            if f in obj:
                out[t] = frame_to_abs(rec, obj[f])
        return out
    if isinstance(obj, list):
        return [_absolutize(x, rec) for x in obj]
    return obj


def get_path(obj, path: str):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


_OPS = {
    "nonempty": lambda v, _: bool(v),
    "empty": lambda v, _: v is not None and not v,
    "eq": lambda v, x: v == x,
    "ne": lambda v, x: v is not None and v != x,
    "gt": lambda v, x: isinstance(v, (int, float)) and v > x,
    "contains": lambda v, x: v is not None and x in v,
}


def load_rules(path: str | Path) -> list[dict]:
    rules = yaml.safe_load(Path(path).read_text())["rules"]
    for r in rules:
        for c in r.get("all", []) + r.get("any", []):
            if c["op"] not in _OPS:
                raise ValueError(f"rule {r['id']}: unknown op {c['op']}")
    return rules


def eval_rule(rule: dict, view: dict) -> bool:
    """view = {prompt_id: output}. A condition on a missing prompt is False."""
    check = lambda c: _OPS[c["op"]](get_path(view, c["path"]), c.get("value"))  # noqa: E731
    ok_all = all(check(c) for c in rule.get("all", []))
    ok_any = any(check(c) for c in rule["any"]) if rule.get("any") else True
    return ok_all and ok_any


def reduce_video(records: list[ResultRecord], rules: list[dict] | None = None) -> dict:
    by_seg: dict[int, dict[str, ResultRecord]] = defaultdict(dict)
    for r in records:
        prev = by_seg[r.segment_idx].get(r.prompt_id)
        if prev is None or (r.prompt_version, r.created_at) > (prev.prompt_version, prev.created_at):
            by_seg[r.segment_idx][r.prompt_id] = r

    merged: dict[str, list] = defaultdict(list)
    summaries, alerts, failures = [], [], []
    for seg in sorted(by_seg):
        view = {}
        for pid, r in sorted(by_seg[seg].items()):
            if not r.ok or not isinstance(r.output, dict):
                failures.append({"segment_idx": seg, "prompt_id": pid, "error": r.error})
                continue
            out = _absolutize(r.output, r)
            view[pid] = out
            if pid == "summary" and "summary" in out:
                summaries.append({"segment_idx": seg, "start_s": r.segment_start_s,
                                  "summary": out["summary"]})
                continue
            lists = {k: v for k, v in out.items() if isinstance(v, list)}
            if lists:
                for k, v in lists.items():
                    merged[f"{pid}.{k}"].extend({**x, "segment_idx": seg} if isinstance(x, dict)
                                                else x for x in v)
            else:
                merged[pid].append({**out, "segment_idx": seg, "start_s": r.segment_start_s})
        for rule in rules or []:
            if eval_rule(rule, view):
                alerts.append({"rule_id": rule["id"], "severity": rule.get("severity", "info"),
                               "segment_idx": seg,
                               "start_s": next(iter(by_seg[seg].values())).segment_start_s,
                               "message": rule.get("message", rule["id"])})
    video_id = records[0].video_id if records else None
    return {"video_id": video_id, "segments": len(by_seg), "merged": dict(merged),
            "summary": " ".join(s["summary"] for s in summaries),
            "segment_summaries": summaries, "alerts": alerts, "failures": failures}
