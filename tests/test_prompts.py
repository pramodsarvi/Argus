import json
from pathlib import Path

from vlmpipe.prompts import Registry, build_messages, merged_question

REG = Registry.load(Path(__file__).parent.parent / "configs" / "prompts.yaml")


def test_shared_prefix_identical_across_prompts():
    frames = ["/x/a.png", "/x/b.png", "/x/c.png"]
    warm = build_messages(REG.system_text, frames, None)
    for p in REG.prompts:
        m = build_messages(REG.system_text, frames, p.text)
        assert m[0] == warm[0]
        n = len(warm[1]["content"])
        assert n == 2 * len(frames)                      # "Frame N:" label + image per frame
        assert json.dumps(m[1]["content"][:n]) == json.dumps(warm[1]["content"])
        assert m[1]["content"][-1] == {"type": "text", "text": p.text}
        assert len(m[1]["content"]) == n + 1


def test_frame_labels_are_one_based_and_ordered():
    parts = build_messages(REG.system_text, ["/x/a.png", "/x/b.png"], None)[1]["content"]
    assert [x["text"] for x in parts if x["type"] == "text"] == ["Frame 1:", "Frame 2:"]


def test_no_variable_content_in_prefix():
    # guard against someone templating timestamps/ids into the system prompt
    for tok in ("{", "}", "%(", "$"):
        assert tok not in REG.system_text


def test_merged_schema_covers_all():
    q, schema = merged_question(REG.prompts)
    assert set(schema["required"]) == {p.id for p in REG.prompts}
    for p in REG.prompts:
        assert p.id in q
