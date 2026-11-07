from pathlib import Path

import httpx
import pytest

from vlmpipe.results.alerts import WebhookDispatcher
from vlmpipe.results.consumer import load_schemas, process_message, validate
from vlmpipe.results.models import ResultRecord
from vlmpipe.results.publisher import InMemoryPublisher
from vlmpipe.results.reducer import eval_rule, load_rules, reduce_video
from vlmpipe.results.store import SQLiteStore

ROOT = Path(__file__).parent.parent
SCHEMAS = load_schemas(ROOT / "configs" / "prompts.yaml")
RULES = load_rules(ROOT / "configs" / "rules.yaml")
M = "qwen2.5-vl-7b-fp8"


def rec(seg, pid, out, ok=True, start=None, **kw):
    return ResultRecord("vid1", seg, pid, 1, M, ok, out,
                        segment_start_s=60.0 * seg if start is None else start,
                        frame_times=[0.0, 1.0, 2.0, 3.0], **kw)


def test_roundtrip_and_key():
    r = rec(0, "anomaly", {"anomaly": False, "description": ""})
    r2 = ResultRecord.from_json(r.to_json())
    assert r2 == r and r2.key() == ("vid1", 0, "anomaly", 1, M)


def test_publisher_keys_by_video():
    pub = InMemoryPublisher()
    pub.publish(rec(0, "ocr", {"texts": []}))
    topic, key, _ = pub.messages[0]
    assert topic == "vlm.results.ocr" and key == b"vid1"


def test_validate():
    assert validate(rec(0, "anomaly", {"anomaly": True, "description": "x"}), SCHEMAS).ok
    bad = validate(rec(0, "anomaly", {"anomaly": "yes"}), SCHEMAS)
    assert not bad.ok and bad.error.startswith("schema")
    unk = validate(ResultRecord("v", 0, "nope", 9, M, True, {}), SCHEMAS)
    assert not unk.ok and "unknown prompt" in unk.error


def test_idempotent_upsert_via_consumer():
    store = SQLiteStore()
    r = rec(0, "summary", {"summary": "first"})
    process_message(r.to_json(), store, SCHEMAS)
    process_message(r.to_json(), store, SCHEMAS)                 # redelivery
    r.output = {"summary": "second"}
    process_message(r.to_json(), store, SCHEMAS)                 # re-run overwrites
    rows = store.get_video("vid1")
    assert len(rows) == 1 and rows[0].output == {"summary": "second"}


def test_reduce_absolute_times_and_merge():
    recs = [
        rec(0, "safety", {"violations": [{"type": "no_helmet", "frame": 2, "description": "d"}]}),
        rec(0, "anomaly", {"anomaly": True, "description": "spill"}),
        rec(0, "summary", {"summary": "A."}),
        rec(1, "safety", {"violations": []}),
        rec(1, "anomaly", {"anomaly": True, "description": "crate"}),
        rec(1, "actions", {"actions": [{"action": "walk", "start_frame": 1, "end_frame": 4}]}),
        rec(1, "summary", {"summary": "B."}),
        rec(1, "ocr", None, ok=False, error="timeout"),
    ]
    out = reduce_video(recs, RULES)
    v = out["merged"]["safety.violations"][0]
    assert v["t"] == 1.0 and v["segment_idx"] == 0           # seg0 start 0 + frame_times[1]
    a = out["merged"]["actions.actions"][0]
    assert (a["start_t"], a["end_t"]) == (60.0, 63.0)
    assert out["summary"] == "A. B."
    assert out["failures"] == [{"segment_idx": 1, "prompt_id": "ocr", "error": "timeout"}]
    fired = {(x["rule_id"], x["segment_idx"]) for x in out["alerts"]}
    assert fired == {("safety_with_anomaly", 0), ("safety_violation", 0), ("anomaly_only", 1)}


def test_reduce_prefers_latest_version():
    old = rec(0, "summary", {"summary": "old"})
    new = ResultRecord("vid1", 0, "summary", 2, M, True, {"summary": "new"})
    assert reduce_video([new, old])["summary"] == "new"


def test_missing_prompt_rule_false():
    assert not eval_rule(RULES[0], {"safety": {"violations": [1]}})


def _dispatcher(statuses, calls):
    it = iter(statuses)

    def handler(req):
        calls.append(req)
        s = next(it)
        if s == "err":
            raise httpx.ConnectError("down")
        return httpx.Response(s)
    return WebhookDispatcher("http://hook", max_retries=3,
                             client=httpx.Client(transport=httpx.MockTransport(handler)),
                             sleep=lambda _: None)


ALERT = {"rule_id": "r", "segment_idx": 0, "severity": "high"}


@pytest.mark.parametrize("statuses,expect,n", [
    ([200], True, 1), (["err", 503, 200], True, 3), ([429, 429, 429, 429], False, 4),
    ([400], False, 1)])
def test_webhook_retry(statuses, expect, n):
    calls = []
    assert _dispatcher(statuses, calls).send("vid1", ALERT) is expect
    assert len(calls) == n
    assert len({c.headers["Idempotency-Key"] for c in calls}) == 1


# ---------------- camera streams ----------------
import asyncio
import json
import sqlite3

from vlmpipe.results.camera_rules import (Chunk, CameraRuleEngine, chunk_status,
                                          chunks_from_records, load_camera_rules)
from vlmpipe.results.consumer import default_schemas
from vlmpipe.results.publisher import AsyncRecordPublisher
from vlmpipe.results.reducer import frame_to_abs

T0 = 1_762_500_000.0          # absolute UTC epoch of the first chunk
CAM_RULES = load_camera_rules(ROOT / "configs" / "rules.yaml")


def cam(pid, out, ts=T0, ok=True, camera="cam-07", status=None, **kw):
    return ResultRecord(f"{camera}@{ts}", 0, pid, 1, M, ok, out, frame_times=[0.0, 5.0, 10.0],
                        camera_id=camera, chunk_start_ts=ts, prompt_set="dock", prompt_set_version=3,
                        status=status, **kw)


def test_camera_key_partition_and_roundtrip():
    r = cam("safety", {"violations": []})
    assert r.key() == ("cam-07", f"{T0:.3f}", 0, "safety", 1, M)
    assert r.partition_key == "cam-07"
    assert ResultRecord.from_json(r.to_json()) == r
    # forward compatible: unknown fields ignored; old JSON without camera fields still loads
    d = json.loads(r.to_json()); d["future_field"] = 1
    assert ResultRecord.from_json(json.dumps(d)).camera_id == "cam-07"
    old = rec(0, "ocr", {"texts": []})
    assert old.key() == ("vid1", 0, "ocr", 1, M) and old.partition_key == "vid1"


def test_bad_status_rejected():
    with pytest.raises(ValueError):
        cam("safety", {}, status="weird")


def test_publisher_keys_by_camera():
    pub = InMemoryPublisher()
    pub.publish(cam("safety", {"violations": []}))
    assert pub.messages[0][1] == b"cam-07"


def test_adapter_maps_camera_fields_top_level_and_metrics():
    base = dict(video_id="cam-07@x", segment_index=0, prompt_id="safety", prompt_version=1,
                model=M, output={"violations": []})
    top = AsyncRecordPublisher.to_record({**base, "status": "ok", "camera_id": "cam-07",
                                          "chunk_start_ts": T0, "prompt_set": "dock",
                                          "prompt_set_version": 3,
                                          "metrics": {"frame_times": [0.0, 5.0]}})
    met = AsyncRecordPublisher.to_record({**base, "status": "timed_out",
                                          "metrics": {"camera_id": "cam-07", "chunk_start_ts": str(T0),
                                                      "prompt_set": "dock", "prompt_set_version": "3",
                                                      "frame_times": [0.0, 5.0]}})
    for r in (top, met):
        assert (r.camera_id, r.chunk_start_ts, r.prompt_set, r.prompt_set_version) == \
               ("cam-07", T0, "dock", 3)
        assert r.frame_times == [0.0, 5.0]
    assert top.ok and top.status == "ok"
    assert not met.ok and met.status == "timed_out" and met.error == "timed_out"
    stale = AsyncRecordPublisher.to_record({**base, "status": "stale", "camera_id": "c"})
    assert stale.status == "stale"
    plain = AsyncRecordPublisher.to_record({**base, "status": "ok"})
    assert plain.camera_id is None and plain.key()[0] == "cam-07@x"


def test_absolute_frame_timestamps():
    r = cam("ocr", {"texts": [{"frame": 2, "text": "EXIT"}]})
    assert frame_to_abs(r, 2) == round(T0 + 5.0, 3)
    out = reduce_video([r])["merged"]["ocr.texts"][0]
    assert out["t"] == round(T0 + 5.0, 3)


def test_store_camera_upsert_idempotent_and_query(tmp_path):
    st = SQLiteStore(str(tmp_path / "r.db"))
    for out in ({"violations": []}, {"violations": [{"type": "no_helmet", "frame": 1, "description": "x"}]}):
        process_message(cam("safety", out).to_json(), st, SCHEMAS)
    process_message(cam("safety", {"violations": []}, ts=T0 + 180).to_json(), st, SCHEMAS)
    process_message(cam("safety", {"violations": []}, camera="cam-08").to_json(), st, SCHEMAS)
    rows = st.get_camera("cam-07")
    assert [r.chunk_start_ts for r in rows] == [T0, T0 + 180]          # redelivery overwrote
    assert rows[0].output["violations"][0]["type"] == "no_helmet"
    assert rows[0].status == "ok" and rows[0].prompt_set_version == 3
    assert [r.chunk_start_ts for r in st.get_camera("cam-07", since=T0 + 1)] == [T0 + 180]
    idx = {r[1] for r in st.conn.execute("PRAGMA index_list(vlm_results)")}
    assert "vlm_results_camera_ts" in idx


def test_sqlite_migrates_old_schema(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE vlm_results (
      video_id TEXT NOT NULL, segment_idx INTEGER NOT NULL, prompt_id TEXT NOT NULL,
      prompt_version INTEGER NOT NULL, model TEXT NOT NULL, ok BOOLEAN NOT NULL, output TEXT,
      segment_start_s DOUBLE PRECISION, frame_times TEXT, error TEXT, created_at DOUBLE PRECISION,
      PRIMARY KEY (video_id, segment_idx, prompt_id, prompt_version, model))""")
    c.execute("INSERT INTO vlm_results VALUES ('vid1',0,'ocr',1,?,1,'{\"texts\": []}',0,'[]',NULL,1.0)", (M,))
    c.commit(); c.close()
    st = SQLiteStore(str(path))
    cols = {r[1] for r in st.conn.execute("PRAGMA table_info(vlm_results)")}
    assert {"rkey", "camera_id", "chunk_start_ts", "prompt_set", "prompt_set_version", "status"} <= cols
    # old row's backfilled rkey matches key_str -> re-upsert of the same record updates, not duplicates
    st.upsert(rec(0, "ocr", {"texts": [{"frame": 1, "text": "NEW"}]}))
    got = st.get_video("vid1")
    assert len(got) == 1 and got[0].output["texts"][0]["text"] == "NEW"
    st.upsert(cam("safety", {"violations": []}))
    assert len(st.get_camera("cam-07")) == 1


def test_schemas_from_promptsets_library(tmp_path):
    (tmp_path / "prompts.yaml").write_text((ROOT / "configs" / "prompts.yaml").read_text())
    (tmp_path / "promptsets.yaml").write_text("""
library:
  - id: forklift_count
    version: 2
    schema: {type: object, properties: {count: {type: integer}}, required: [count]}
  - {text: missing id is skipped}
sets: {dock: {prompts: [forklift_count]}}
""")
    s = load_schemas(tmp_path / "prompts.yaml", tmp_path / "promptsets.yaml", tmp_path / "nope.yaml")
    assert ("forklift_count", 2) in s and ("safety", 1) in s
    good = ResultRecord("v", 0, "forklift_count", 2, M, True, {"count": 3}, camera_id="c", chunk_start_ts=T0)
    bad = ResultRecord("v", 0, "forklift_count", 2, M, True, {"count": "3"}, camera_id="c", chunk_start_ts=T0)
    assert validate(good, s).status == "ok"
    assert validate(bad, s).status == "invalid" and not bad.ok
    unknown = validate(ResultRecord("v", 0, "zzz", 1, M, True, {}), s)
    assert unknown.status == "invalid"
    # mapping-shaped library and garbage files are tolerated
    (tmp_path / "p2.yaml").write_text("library: {gate_open: {version: 1, schema: {type: object}}}")
    (tmp_path / "p3.yaml").write_text("- just a list")
    s2 = load_schemas(tmp_path / "p2.yaml", tmp_path / "p3.yaml")
    assert ("gate_open", 1) in s2
    assert ("safety", 1) in default_schemas(ROOT / "configs")


# ---- camera rule engine ----

V = {"violations": [{"type": "no_helmet", "frame": 1, "description": "x"}]}
NOV = {"violations": []}


def ch(ts, safety=None, anomaly=None, status="ok", camera="cam-07"):
    view = {}
    if safety is not None:
        view["safety"] = safety
    if anomaly is not None:
        view["anomaly"] = {"anomaly": anomaly, "description": ""}
    return Chunk(camera, T0 + ts, view, status)


def _ids(alerts):
    return [a["rule_id"] for a in alerts]


def test_consecutive_fires_on_k_and_cooldown():
    eng = CameraRuleEngine([r for r in CAM_RULES if r["id"] == "persistent_safety_violation"])
    assert eng.observe(ch(0, V)) == []
    a = eng.observe(ch(180, V))
    assert _ids(a) == ["persistent_safety_violation"] and a[0]["streak"] == 2
    assert a[0]["chunk_start_ts"] == T0 + 180 and a[0]["camera_id"] == "cam-07"
    assert eng.observe(ch(360, V)) == []                    # cooldown 900 s
    assert eng.observe(ch(540, V)) == []
    assert eng.observe(ch(720, V)) == []
    assert eng.observe(ch(900, V)) == []                    # 900-180 = 720 < 900
    assert _ids(eng.observe(ch(1080, V))) == ["persistent_safety_violation"]   # 1080-180 >= 900


def test_consecutive_broken_by_clean_stale_failed_and_gap():
    rule = [r for r in CAM_RULES if r["id"] == "persistent_safety_violation"]
    for breaker in (ch(180, NOV), ch(180, V, status="stale"), ch(180, None, status="failed")):
        eng = CameraRuleEngine(rule)
        eng.observe(ch(0, V)); eng.observe(breaker)
        assert eng.observe(ch(360, V)) == []
    eng = CameraRuleEngine(rule)                              # camera offline > max_gap_s
    eng.observe(ch(0, V))
    assert eng.observe(ch(600, V)) == []
    # break_on configurable: stale does not reset if not listed
    eng = CameraRuleEngine([{**rule[0], "break_on": ["failed"]}])
    eng.observe(ch(0, V))
    assert _ids(eng.observe(ch(180, V, status="stale"))) == ["persistent_safety_violation"]


def test_cameras_are_independent():
    eng = CameraRuleEngine([r for r in CAM_RULES if r["id"] == "persistent_safety_violation"])
    eng.observe(ch(0, V, camera="a"))
    assert eng.observe(ch(180, V, camera="b")) == []
    assert _ids(eng.observe(ch(180, V, camera="a"))) == ["persistent_safety_violation"]


def test_window_rule():
    eng = CameraRuleEngine([r for r in CAM_RULES if r["id"] == "recurring_anomaly"])
    assert eng.observe(ch(0, anomaly=True)) == []
    assert eng.observe(ch(180, anomaly=False)) == []
    assert eng.observe(ch(360, anomaly=True)) == []
    a = eng.observe(ch(540, anomaly=True))
    assert _ids(a) == ["recurring_anomaly"] and a[0]["count_in_window"] == 3
    eng = CameraRuleEngine([r for r in CAM_RULES if r["id"] == "recurring_anomaly"])
    for t in (0, 1000, 2000):                                  # spread beyond 1800 s window
        assert eng.observe(ch(t, anomaly=True)) == []


def test_out_of_order_and_duplicate_chunks_ignored():
    eng = CameraRuleEngine([r for r in CAM_RULES if r["id"] == "persistent_safety_violation"])
    eng.observe(ch(180, V))
    assert eng.observe(ch(180, V)) == [] and eng.observe(ch(0, V)) == []
    assert eng.skipped == 2


def test_chunks_from_records_and_engine_end_to_end():
    recs = [cam("safety", V, ts=T0), cam("anomaly", {"anomaly": False, "description": ""}, ts=T0),
            cam("safety", V, ts=T0 + 180),
            cam("anomaly", None, ts=T0 + 180, ok=False, status="timed_out"),
            cam("safety", None, ts=T0 + 360, ok=False, status="stale"),
            cam("safety", V, ts=T0, camera="cam-08")]
    chunks = chunks_from_records(recs + [rec(0, "ocr", {"texts": []})])   # non-camera ignored
    assert [(c.camera_id, c.chunk_start_ts, c.status) for c in chunks] == [
        ("cam-07", T0, "ok"), ("cam-08", T0, "ok"), ("cam-07", T0 + 180, "partial"),
        ("cam-07", T0 + 360, "stale")]
    assert "anomaly" not in chunks[2].view
    eng = CameraRuleEngine(CAM_RULES)
    alerts = eng.observe_records(recs)
    assert [(a["rule_id"], a["chunk_start_ts"]) for a in alerts] == [
        ("persistent_safety_violation", T0 + 180)]


def test_chunk_status():
    assert chunk_status([cam("a", {}, status="ok")]) == "ok"
    assert chunk_status([cam("a", {}, ok=False, status="failed")]) == "failed"
    assert chunk_status([cam("a", {}, ok=False)]) == "failed"


def test_bad_camera_rules_rejected(tmp_path):
    for body in ("camera_rules: [{id: x, type: sometimes, condition: {all: [{path: a, op: empty}]}}]",
                 "camera_rules: [{id: x, type: consecutive, k: 0, condition: {all: [{path: a, op: empty}]}}]",
                 "camera_rules: [{id: x, type: window, count: 2, condition: {all: [{path: a, op: empty}]}}]",
                 "camera_rules: [{id: x, type: consecutive, k: 2, condition: {}}]",
                 "camera_rules: [{id: x, type: consecutive, k: 2, condition: {all: [{path: a, op: zz}]}}]"):
        p = tmp_path / "r.yaml"; p.write_text(body)
        with pytest.raises(ValueError):
            load_camera_rules(p)


def test_rule_state_survives_restart(tmp_path):
    rules = [r for r in CAM_RULES if r["id"] == "persistent_safety_violation"]
    db = str(tmp_path / "r.db")
    s1 = SQLiteStore(db)
    assert CameraRuleEngine(rules, s1).observe(ch(0, V)) == []        # streak 1, then "crash"
    s2 = SQLiteStore(db)
    eng = CameraRuleEngine(rules, s2)                                 # fresh process
    assert eng.observe(ch(0, V)) == [] and eng.skipped == 1           # redelivery ignored
    a = eng.observe(ch(180, V))
    assert _ids(a) == ["persistent_safety_violation"] and a[0]["streak"] == 2
    eng3 = CameraRuleEngine(rules, SQLiteStore(db))                   # restart again: cooldown kept
    assert eng3.observe(ch(360, V)) == []
