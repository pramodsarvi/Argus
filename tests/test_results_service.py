from vlmpipe.results.camera_rules import CameraRuleEngine
from vlmpipe.results.models import ResultRecord
from vlmpipe.results.service import ChunkAssembler, build_default
from vlmpipe.results.alerts import WebhookDispatcher

RULE = [{"id": "persist", "type": "consecutive", "k": 2,
         "condition": {"all": [{"path": "safety.violations", "op": "nonempty"}]}}]


def rec(ts, pid, out, cam="dock-1"):
    return ResultRecord(video_id=cam, segment_idx=int(ts), prompt_id=pid, prompt_version=1,
                        model="m", ok=True, output=out, camera_id=cam, chunk_start_ts=ts,
                        status="ok")


def make(clock=lambda: 0.0):
    fired = []
    asm = ChunkAssembler(lambda cam: {"safety", "summary"}, CameraRuleEngine(RULE),
                         fired.append, chunk_timeout_s=60, clock=clock)
    return asm, fired


def test_flush_only_when_chunk_complete_and_rule_fires_on_second():
    asm, fired = make()
    v = {"violations": [{"type": "no_helmet", "frame": 1, "description": "x"}]}
    assert asm.add(rec(0, "safety", v)) == []
    asm.add(rec(0, "summary", {"summary": "s"}))
    assert fired == []                                   # streak 1
    asm.add(rec(180, "safety", v))
    asm.add(rec(180, "summary", {"summary": "s"}))
    assert [a["rule_id"] for a in fired] == ["persist"]


def test_timeout_flushes_incomplete_chunk():
    t = [0.0]
    asm, fired = make(clock=lambda: t[0])
    v = {"violations": [{"type": "x", "frame": 1, "description": "x"}]}
    asm.add(rec(0, "safety", v))
    t[0] = 61
    asm.sweep()
    assert asm.pending == {}


def test_build_default_and_camera_alert_idem_key():
    asm = build_default()
    assert asm.expected("dock-01")
    k = WebhookDispatcher.idem_key("dock-01", {"rule_id": "r", "chunk_start_ts": 1.0})
    assert len(k) == 32
