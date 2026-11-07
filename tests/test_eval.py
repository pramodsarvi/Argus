import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from eval import gate  # noqa: E402
from eval.runner import aggregate, load_dataset  # noqa: E402
from eval.scorers import cer, levenshtein, score  # noqa: E402


def test_levenshtein_cer():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("EXIT  Door", "exit door") == 0.0
    assert cer("", "abc") == 1.0


def test_ocr():
    exp = {"texts": [{"frame": 1, "text": "EXIT"}, {"frame": 2, "text": "Danger"}]}
    assert score("ocr", exp, exp)["score"] == 1.0
    assert score("ocr", {"texts": [{"frame": 1, "text": "EXT"}]}, exp)["score"] < 1.0


def test_safety_frame_tolerance():
    exp = {"violations": [{"type": "no helmet", "frame": 5, "description": ""}]}
    near = {"violations": [{"type": "No Helmet", "frame": 6, "description": "x"}]}
    far = {"violations": [{"type": "no helmet", "frame": 9, "description": "x"}]}
    assert score("safety", near, exp)["f1"] == 1.0
    assert score("safety", far, exp)["f1"] == 0.0
    assert score("safety", {"violations": []}, {"violations": []})["score"] == 1.0


def test_actions_prf():
    exp = {"actions": [{"action": "walk", "start_frame": 0, "end_frame": 3},
                       {"action": "lift", "start_frame": 5, "end_frame": 7}]}
    pred = {"actions": [{"action": "walk", "start_frame": 1, "end_frame": 3},
                        {"action": "run", "start_frame": 5, "end_frame": 7}]}
    s = score("actions", pred, exp)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_anomaly_summary_invalid():
    assert score("anomaly", {"anomaly": True}, {"anomaly": True})["score"] == 1.0
    assert score("summary", {"summary": "A forklift moves a pallet"},
                 {"keywords": ["forklift", "pallet", "truck"]})["score"] == pytest.approx(2 / 3)
    assert score("ocr", "not json", {"texts": []})["score"] == 0.0


def test_summary_judge_hook():
    from eval.scorers import score_summary
    s = score_summary({"summary": "x"}, {"keywords": ["x"]}, judge=lambda p, e: 0.0)
    assert s["score"] == 0.5


def test_dataset_and_aggregate(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"segment_path": "a.mp4", "prompt_id": "ocr", "expected": {}}) + "\n\n")
    assert len(load_dataset(p)) == 1
    p.write_text(json.dumps({"segment_path": "a.mp4"}) + "\n")
    with pytest.raises(ValueError):
        load_dataset(p)
    agg = aggregate([{"prompt_id": "ocr", "score": 1.0}, {"prompt_id": "ocr", "score": 0.5, "error": "x"}])
    assert agg["ocr"] == {"n": 2, "mean_score": 0.75, "invalid_outputs": 1}


def test_gate(tmp_path):
    base = {"per_prompt": {"ocr": {"mean_score": 0.9}, "safety": {"mean_score": 0.8}}}
    ok = {"per_prompt": {"ocr": {"mean_score": 0.89}, "safety": {"mean_score": 0.85}}}
    bad = {"per_prompt": {"ocr": {"mean_score": 0.80}}}
    assert gate.check(ok, base) == []
    fails = gate.check(bad, base)
    assert len(fails) == 2  # ocr drop + safety missing
    assert gate.check(ok, base, {"ocr": {"min_score": 0.95}})
    for name, d in (("b", base), ("ok", ok), ("bad", bad)):
        (tmp_path / f"{name}.json").write_text(json.dumps(d))
    assert gate.main(["--report", str(tmp_path / "ok.json"), "--baseline", str(tmp_path / "b.json")]) == 0
    assert gate.main(["--report", str(tmp_path / "bad.json"), "--baseline", str(tmp_path / "b.json")]) == 1
