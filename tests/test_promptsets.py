import os
import shutil
from pathlib import Path

import pytest
import yaml

from vlmpipe.prompts import Registry, build_messages
from vlmpipe.promptsets import PromptSetError, PromptSetRegistry

CFG = Path(__file__).parent.parent / "configs"


@pytest.fixture
def cfg(tmp_path):
    ps, cm = tmp_path / "promptsets.yaml", tmp_path / "cameras.yaml"
    shutil.copy(CFG / "promptsets.yaml", ps)
    shutil.copy(CFG / "cameras.yaml", cm)
    return ps, cm


def _edit(path: Path, fn):
    d = yaml.safe_load(path.read_text())
    fn(d)
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    st = os.stat(path)                                    # ensure mtime moves even on coarse clocks
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_shipped_configs_load():
    reg = PromptSetRegistry.load(CFG / "promptsets.yaml", CFG / "cameras.yaml")
    assert {"general", "warehouse_dock", "retail_entrance", "production_line"} <= set(reg.set_ids)


def test_precedence_explicit_group_default(cfg):
    reg = PromptSetRegistry.load(*cfg)
    assert reg.for_camera("dock-07").sampling["max_frames"] == 48          # explicit
    d = reg.for_camera("dock-02")                                           # group
    assert d.set_id == "warehouse_dock" and d.sampling["max_frames"] == 40
    assert d.sampling["strategy"] == "scene"                                # merged from defaults
    assert reg.for_camera("lobby-main").set_id == "retail_entrance"         # explicit, no group
    g = reg.for_camera("parking-1")                                         # default
    assert g.set_id == "general" and g.sampling["max_frames"] == 32


def test_glob_groups_first_match_wins(cfg):
    ps, cm = cfg
    _edit(cm, lambda d: d["groups"].insert(0, {"match": "dock-1*", "set": "general"}))
    reg = PromptSetRegistry.load(ps, cm)
    assert reg.for_camera("dock-12").set_id == "general"
    assert reg.for_camera("dock-22").set_id == "warehouse_dock"
    assert reg.for_camera("line-3").sampling["min_fps"] == 0.33


def test_routing_and_options(cfg):
    reg = PromptSetRegistry.load(*cfg)
    r = reg.for_camera("entrance-1")
    assert r.routing["queue_length"] == "visual"          # set override beats library tier
    assert r.routing["loitering"] == "text"
    assert r.options["summary"]["required"] is False
    line = reg.for_camera("line-1")
    assert "line_stopped" in [p.id for p in line.registry.prompts]   # inline prompt


def test_registry_compatible_with_build_messages(cfg):
    reg = PromptSetRegistry.load(*cfg)
    a, b = reg.for_camera("dock-1"), reg.for_camera("entrance-1")
    assert isinstance(a.registry, Registry)
    frames = ["/f/1.png", "/f/2.png"]
    ma = build_messages(a.registry.system_text, frames, a.registry.prompts[0].text)
    mb = build_messages(b.registry.system_text, frames, b.registry.prompts[0].text)
    assert ma[0] == mb[0]                                 # same global shared system across sets
    assert ma[1]["content"][:-1] == mb[1]["content"][:-1]


def test_no_mapping_no_default_raises(cfg):
    ps, cm = cfg
    _edit(cm, lambda d: d["defaults"].pop("set"))
    reg = PromptSetRegistry.load(ps, cm)
    with pytest.raises(KeyError):
        reg.for_camera("parking-1")
    assert reg.for_camera("dock-1").set_id == "warehouse_dock"


def test_hot_reload_and_invalid_edit_keeps_old(cfg):
    ps, cm = cfg
    reg = PromptSetRegistry.load(ps, cm)
    assert reg.reload_if_changed() is False
    _edit(ps, lambda d: d["sets"]["general"].update(version=2))
    assert reg.reload_if_changed() is True
    assert reg.for_camera("parking-1").set_version == 2
    _edit(ps, lambda d: d["sets"]["general"]["prompts"].append("nope"))
    assert reg.reload_if_changed() is False
    assert reg.for_camera("parking-1").set_version == 2   # old config kept
    assert reg.reload_if_changed() is False               # broken file not re-parsed each poll


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["sets"]["general"]["prompts"].append("ghost"), "unknown prompt ref 'ghost'"),
    (lambda d: d["sets"]["general"]["prompts"].append("summary"), "duplicate prompt ids"),
    (lambda d: d["library"][0].update(schema="string"), "JSON-schema object"),
    (lambda d: d["shared_system"].pop("version"), "shared_system"),
    (lambda d: d["sets"]["retail_entrance"]["routing"].update(queue_length="gpu"), "routing"),
])
def test_validation_errors(cfg, mutate, msg):
    ps, cm = cfg
    _edit(ps, mutate)
    with pytest.raises(PromptSetError, match=msg):
        PromptSetRegistry.load(ps, cm)


def test_camera_unknown_set(cfg):
    ps, cm = cfg
    _edit(cm, lambda d: d["cameras"].update({"x": {"set": "missing"}}))
    with pytest.raises(PromptSetError, match="unknown set 'missing'"):
        PromptSetRegistry.load(ps, cm)
