"""Camera-level temporal rules over consecutive chunks of the same camera.

Rule types (configs/rules.yaml, key `camera_rules`):
  consecutive: condition true in K consecutive chunks       {k: 2}
  window:      condition true in >= N chunks within T secs  {count: 3, window_s: 900}

`condition` uses the same all/any + ops syntax as per-segment rules, evaluated over the
chunk view {prompt_id: validated output}. A condition on a prompt missing from the chunk is
False. Chunks whose status is in `break_on` (default: stale, failed) reset `consecutive`
streaks; for `window` they simply don't count. `max_gap_s` resets streaks when chunks are
missing (e.g. camera offline). `cooldown_s` suppresses repeat alerts per (camera, rule),
measured in chunk time, so replays are deterministic.

Feed chunks in chunk_start_ts order per camera. Out-of-order or duplicate chunks (ts <= last
seen for that camera) are ignored and counted in `skipped`, so redelivery is harmless.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import ResultRecord
from .reducer import _OPS, eval_rule

RULE_TYPES = ("consecutive", "window")


def load_camera_rules(path: str | Path) -> list[dict]:
    d = yaml.safe_load(Path(path).read_text()) or {}
    rules = d.get("camera_rules") or []
    for r in rules:
        if r.get("type") not in RULE_TYPES:
            raise ValueError(f"camera rule {r.get('id')}: type must be one of {RULE_TYPES}")
        cond = r.get("condition") or {}
        if not (cond.get("all") or cond.get("any")):
            raise ValueError(f"camera rule {r['id']}: empty condition")
        for c in cond.get("all", []) + cond.get("any", []):
            if c["op"] not in _OPS:
                raise ValueError(f"camera rule {r['id']}: unknown op {c['op']}")
        if r["type"] == "consecutive" and int(r.get("k", 0)) < 1:
            raise ValueError(f"camera rule {r['id']}: k must be >= 1")
        if r["type"] == "window" and (int(r.get("count", 0)) < 1 or float(r.get("window_s", 0)) <= 0):
            raise ValueError(f"camera rule {r['id']}: count >= 1 and window_s > 0 required")
    return rules


@dataclass
class Chunk:
    camera_id: str
    chunk_start_ts: float
    view: dict                      # {prompt_id: output} of ok records only
    status: str                     # ok | partial | failed | stale


def chunk_status(recs: list[ResultRecord]) -> str:
    st = [r.effective_status() for r in recs]
    if "stale" in st:
        return "stale"
    if not any(s == "ok" for s in st):
        return "failed"
    return "ok" if all(s == "ok" for s in st) else "partial"


def chunks_from_records(records: list[ResultRecord]) -> list[Chunk]:
    """Group camera records into chunks ordered by (chunk_start_ts, camera_id). Latest prompt
    version wins per prompt within a chunk."""
    groups: dict[tuple[str, float], list[ResultRecord]] = defaultdict(list)
    for r in records:
        if r.camera_id is None or r.chunk_start_ts is None:
            continue
        groups[(r.camera_id, float(r.chunk_start_ts))].append(r)
    out = []
    for (cam, ts), recs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        best: dict[str, ResultRecord] = {}
        for r in recs:
            p = best.get(r.prompt_id)
            if p is None or (r.prompt_version, r.created_at) > (p.prompt_version, p.created_at):
                best[r.prompt_id] = r
        view = {pid: r.output for pid, r in best.items()
                if r.effective_status() == "ok" and isinstance(r.output, dict)}
        out.append(Chunk(cam, ts, view, chunk_status(list(best.values()))))
    return out


@dataclass
class _State:
    last_ts: float | None = None
    streak: dict = field(default_factory=lambda: defaultdict(int))          # rule -> n
    hits: dict = field(default_factory=lambda: defaultdict(deque))          # rule -> ts deque
    last_alert: dict = field(default_factory=dict)                          # rule -> ts


def _dump(s: _State) -> dict:
    return {"last_ts": s.last_ts, "streak": dict(s.streak),
            "hits": {k: list(v) for k, v in s.hits.items()}, "last_alert": dict(s.last_alert)}


def _load(d: dict) -> _State:
    s = _State(last_ts=d.get("last_ts"))
    s.streak.update(d.get("streak") or {})
    for k, v in (d.get("hits") or {}).items():
        s.hits[k] = deque(v)
    s.last_alert.update(d.get("last_alert") or {})
    return s


class CameraRuleEngine:
    """state_store (optional): object with load_rule_state(camera_id) -> dict | None and
    save_rule_state(camera_id, dict) (the SQLite/Postgres stores). With it, streaks, cooldowns
    and the last-seen chunk survive restarts. State is saved after each observed chunk, BEFORE
    alerts are delivered: a crash in between loses that alert (at-most-once) rather than
    repeating it. Use one engine/consumer instance per camera set (state is not shared live)."""

    def __init__(self, rules: list[dict], state_store=None) -> None:
        self.rules = rules
        self.store = state_store
        self.state: dict[str, _State] = defaultdict(_State)
        self.skipped = 0

    def _state_for(self, camera_id: str) -> _State:
        if camera_id not in self.state and self.store is not None:
            d = self.store.load_rule_state(camera_id)
            if d is not None:
                self.state[camera_id] = _load(d)
        return self.state[camera_id]

    def observe(self, chunk: Chunk) -> list[dict]:
        s = self._state_for(chunk.camera_id)
        ts = chunk.chunk_start_ts
        if s.last_ts is not None and ts <= s.last_ts:
            self.skipped += 1
            return []
        gap = None if s.last_ts is None else ts - s.last_ts
        s.last_ts = ts
        alerts = []
        for rule in self.rules:
            rid = rule["id"]
            breaks = chunk.status in set(rule.get("break_on", ["stale", "failed"]))
            matched = not breaks and eval_rule(rule["condition"], chunk.view)
            fired = False
            if rule["type"] == "consecutive":
                max_gap = rule.get("max_gap_s")
                if max_gap is not None and gap is not None and gap > float(max_gap):
                    s.streak[rid] = 0
                s.streak[rid] = s.streak[rid] + 1 if matched else 0
                fired = s.streak[rid] >= int(rule["k"])
                detail = {"streak": s.streak[rid]}
            else:  # window
                win = float(rule["window_s"])
                dq = s.hits[rid]
                if matched:
                    dq.append(ts)
                while dq and dq[0] <= ts - win:
                    dq.popleft()
                fired = matched and len(dq) >= int(rule["count"])
                detail = {"count_in_window": len(dq)}
            if not fired:
                continue
            last = s.last_alert.get(rid)
            if last is not None and ts - last < float(rule.get("cooldown_s", 0)):
                continue
            s.last_alert[rid] = ts
            alerts.append({"rule_id": rid, "type": rule["type"],
                           "severity": rule.get("severity", "info"),
                           "camera_id": chunk.camera_id, "chunk_start_ts": ts,
                           "message": rule.get("message", rid), **detail})
        if self.store is not None:
            self.store.save_rule_state(chunk.camera_id, _dump(s))
        return alerts

    def observe_records(self, records: list[ResultRecord]) -> list[dict]:
        alerts = []
        for ch in chunks_from_records(records):
            alerts.extend(self.observe(ch))
        return alerts
