import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vlmpipe.prompts import Registry
from vlmpipe.twotier import (TwoTierConfig, answer_from_description, describe_segment,
                             reduce_video, run_two_tier, with_confidence)

ROOT = Path(__file__).parent.parent
REG = Registry.load(ROOT / "configs" / "prompts.yaml")
CFG = TwoTierConfig.load(ROOT / "configs" / "describe.yaml")
FRAMES = ["/x/a.png", "/x/b.png"]

DESC = {"scene": "warehouse", "objects": [{"name": "forklift", "frames": [1, 2]}],
        "people": [{"description": "worker", "ppe": "helmet", "frames": [1]}],
        "actions": [{"action": "walking", "start_frame": 1, "end_frame": 2}],
        "visible_text": [{"frame": 1, "text": "EXIT"}], "notable_states": []}


class FakeClient:
    """AsyncOpenAI-like stub: answers by response_format name; records calls."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **kw):
        name = kw.get("response_format", {}).get("json_schema", {}).get("name", "_none")
        has_images = any(isinstance(m["content"], list) and
                         any(p.get("type") == "image_url" for p in m["content"])
                         for m in kw["messages"])
        self.calls.append({"name": name, "images": has_images, "kw": kw})
        ans = self.answers.get((name, has_images), self.answers.get(name, {}))
        text = ans if isinstance(ans, str) else json.dumps(ans)

        async def gen():
            mid = len(text) // 2
            for piece in (text[:mid], text[mid:]):
                yield NS(choices=[NS(delta=NS(content=piece))], usage=None)
            yield NS(choices=[], usage=NS(prompt_tokens=100, completion_tokens=10,
                                          prompt_tokens_details=NS(cached_tokens=80)))
        return gen()


def answers(conf=0.9):
    return {
        "_describe": DESC,
        ("safety", True): {"violations": []},
        ("ocr", True): {"texts": [{"frame": 1, "text": "EXIT"}]},
        ("actions", False): {"actions": [{"action": "walking", "start_frame": 1, "end_frame": 2}],
                             "confidence": conf},
        ("actions", True): {"actions": [{"action": "walking (visual)", "start_frame": 1, "end_frame": 2}]},
        ("anomaly", False): {"anomaly": False, "description": "none", "confidence": conf},
        ("anomaly", True): {"anomaly": False, "description": "none (visual)"},
        ("summary", False): {"summary": "A worker walks.", "confidence": conf},
        ("summary", True): {"summary": "visual summary"},
        "_reduce": {"summary": "whole video", "cross_segment_events": [
            {"description": "forklift", "segments": [0, 1]}]},
    }


def run(coro):
    return asyncio.run(coro)


def test_config_routing():
    assert CFG.route("ocr").tier == "visual" and CFG.route("safety").tier == "visual"
    for pid in ("actions", "anomaly", "summary"):
        assert CFG.route(pid).tier == "text" and CFG.route(pid).question
    assert CFG.route("unknown").tier == "visual"
    assert 0 < CFG.threshold < 1


def test_with_confidence_does_not_mutate():
    s = REG.prompts[0].schema
    before = json.dumps(s, sort_keys=True)
    out = with_confidence(s)
    assert "confidence" in out["required"] and "confidence" in out["properties"]
    assert json.dumps(s, sort_keys=True) == before


def test_describe_uses_shared_prefix():
    c = FakeClient(answers())
    r = run(describe_segment(c, "m", REG.system_text, FRAMES, CFG))
    assert r.ok and r.output == DESC
    msgs = c.calls[0]["kw"]["messages"]
    assert msgs[0]["content"] == REG.system_text
    assert msgs[1]["content"][-1]["text"] == CFG.describe.text


def test_answer_from_description_text_only():
    c = FakeClient(answers())
    p = next(p for p in REG.prompts if p.id == "anomaly")
    r = run(answer_from_description(c, "t", DESC, p, CFG))
    assert r.ok and r.output["confidence"] == 0.9
    assert not c.calls[0]["images"]
    assert "warehouse" in c.calls[0]["kw"]["messages"][1]["content"]


def test_run_two_tier_high_confidence():
    c = FakeClient(answers(0.9))
    out = run(run_two_tier(c, "m", REG, FRAMES, cfg=CFG))
    assert out["mode"] == "two_tier" and out["description"] == DESC
    assert set(out) >= {"wall_s", "warm", "results", "escalated"}
    assert out["warm"]["prompt_id"] == "_describe"
    tiers = out["tiers"]
    assert tiers == {"safety": "visual", "ocr": "visual", "actions": "text",
                     "anomaly": "text", "summary": "text"}
    assert out["escalated"] == []
    res = {r["prompt_id"]: r for r in out["results"]}
    assert all(r["ok"] for r in res.values())
    assert "confidence" not in res["summary"]["output"]          # stripped to prompt schema
    # visual calls = describe + safety + ocr only
    assert sum(1 for x in c.calls if x["images"]) == 3


def test_run_two_tier_escalates_low_confidence():
    c = FakeClient(answers(0.2))
    out = run(run_two_tier(c, "m", REG, FRAMES, cfg=CFG))
    assert sorted(out["escalated"]) == ["actions", "anomaly", "summary"]
    res = {r["prompt_id"]: r for r in out["results"]}
    assert res["summary"]["output"] == {"summary": "visual summary"}


def test_describe_failure_falls_back_to_visual():
    a = answers()
    a["_describe"] = "not json"
    c = FakeClient(a)
    out = run(run_two_tier(c, "m", REG, FRAMES, cfg=CFG))
    assert out["description"] is None
    assert all(t == "visual" for t in out["tiers"].values())
    assert sorted(out["escalated"]) == ["actions", "anomaly", "summary"]


def test_separate_text_client():
    vis, txt = FakeClient(answers()), FakeClient(answers())
    run(run_two_tier(vis, "vlm", REG, FRAMES, text_client=txt, text_model="small", cfg=CFG))
    assert {x["name"] for x in txt.calls} == {"actions", "anomaly", "summary"}
    assert all(x["kw"]["model"] == "small" for x in txt.calls)
    assert all(x["images"] for x in vis.calls)


def test_reduce_video():
    c = FakeClient(answers())
    segs = [{"segment_idx": 0, "start_s": 0.0, "description": DESC},
            {"segment_idx": 1, "start_s": 60.2, "description": DESC}]
    r = run(reduce_video(c, "t", segs, CFG))
    assert r.ok and r.output["summary"] == "whole video"
    payload = c.calls[0]["kw"]["messages"][1]["content"]
    assert "60.2" in payload and not c.calls[0]["images"]


def test_spike_two_tier_signature_compatible():
    """bench/spike.py calls run_two_tier with these kwargs; guard against drift."""
    import inspect
    from vlmpipe.twotier import run_two_tier
    params = inspect.signature(run_two_tier).parameters
    for k in ("input_mode", "fps", "mm_processor_kwargs"):
        assert k in params


# ---------------- camera streams: previous-chunk context + routing overrides ----------------

def _describe_call(fc):
    return next(c for c in fc.calls if c["name"] == "_describe")


def _user_parts(call):
    return call["kw"]["messages"][1]["content"]


def test_previous_description_goes_after_media_and_keeps_prefix():
    fc_prev, fc_none = FakeClient(answers()), FakeClient(answers())
    prev = {"scene": "dock", "ongoing_events": [{"description": "truck unloading",
                                                 "from_previous": False}]}
    run(run_two_tier(fc_prev, "m", REG, FRAMES, previous_description=prev))
    run(run_two_tier(fc_none, "m", REG, FRAMES))
    with_ctx, without = _user_parts(_describe_call(fc_prev)), _user_parts(_describe_call(fc_none))
    n_media = len(without) - 1
    # identical media prefix (system + frame labels + images); only the trailing question differs
    assert _describe_call(fc_prev)["kw"]["messages"][0] == _describe_call(fc_none)["kw"]["messages"][0]
    assert json.dumps(with_ctx[:n_media]) == json.dumps(without[:n_media])
    last = with_ctx[-1]
    assert last["type"] == "text" and "truck unloading" in last["text"]
    assert "PREVIOUS chunk" in last["text"]
    assert all(p["type"] != "text" or "truck unloading" not in p["text"] for p in with_ctx[:n_media])
    # visual prompts are unaffected by the previous context
    v_prev = [c for c in fc_prev.calls if c["name"] == "safety"][0]
    v_none = [c for c in fc_none.calls if c["name"] == "safety"][0]
    assert v_prev["kw"]["messages"] == v_none["kw"]["messages"]


def test_previous_description_truncated():
    from vlmpipe.twotier import MAX_PREVIOUS_CHARS, describe_question
    q = describe_question(CFG, {"blob": "x" * (MAX_PREVIOUS_CHARS * 2)})
    assert q.endswith("...(truncated)") and len(q) < len(CFG.describe.text) + MAX_PREVIOUS_CHARS + 500
    assert describe_question(CFG, None) == CFG.describe.text


def test_describe_schema_has_continuation_fields():
    props = CFG.describe.schema["properties"]
    assert props["continues_previous"]["type"] == "boolean"
    assert "ongoing_events" in props
    assert {"continues_previous", "ongoing_events"} <= set(CFG.describe.schema["required"])


def test_routing_overrides_and_unknown_prompt_default_visual():
    from vlmpipe.prompts import Prompt
    cam_prompt = Prompt("forklift_count", 1, "How many forklifts are visible?", 64,
                        {"type": "object", "properties": {"count": {"type": "integer"}},
                         "required": ["count"]})
    a = answers()
    a[("forklift_count", True)] = {"count": 2}
    a[("forklift_count", False)] = {"count": 1, "confidence": 0.95}

    fc = FakeClient(a)   # no override -> unknown prompt goes visual (safe default)
    out = run(run_two_tier(fc, "m", REG, FRAMES, prompts=(cam_prompt,)))
    assert out["tiers"] == {"forklift_count": "visual"}
    assert out["results"][0]["output"] == {"count": 2}

    fc = FakeClient(a)   # override -> answered from the description, own prompt text used
    out = run(run_two_tier(fc, "m", REG, FRAMES, prompts=(cam_prompt,),
                           routing_overrides={"forklift_count": "text"}))
    assert out["tiers"] == {"forklift_count": "text"}
    assert out["results"][0]["output"] == {"count": 1}
    txt = [c for c in fc.calls if c["name"] == "forklift_count"][0]
    assert not txt["images"] and "How many forklifts" in txt["kw"]["messages"][1]["content"]

    fc = FakeClient(answers())   # override can force a configured text prompt to visual
    out = run(run_two_tier(fc, "m", REG, FRAMES, routing_overrides={"summary": "visual"}))
    assert out["tiers"]["summary"] == "visual" and out["tiers"]["actions"] == "text"


def test_bad_routing_override_rejected():
    with pytest.raises(ValueError):
        CFG.route("summary", {"summary": "gpu"})
