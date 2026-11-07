"""Per-camera prompt-set registry.

configs/promptsets.yaml : global shared_system, prompt library, named versioned sets
configs/cameras.yaml    : camera_id -> set (explicit > first matching glob group > default),
                          sampling overrides merged default < group < camera

The shared system prompt is global so every camera's media prefix / describe call is uniform.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .prompts import Prompt, Registry

log = logging.getLogger(__name__)

TIERS = ("text", "visual")


class PromptSetError(ValueError):
    """Invalid promptsets/cameras configuration."""


@dataclass(frozen=True)
class CameraPromptSet:
    camera_id: str
    set_id: str
    set_version: int
    registry: Registry
    routing: dict[str, str] = field(default_factory=dict, hash=False, compare=True)
    sampling: dict = field(default_factory=dict, hash=False, compare=True)
    # per-prompt extras not carried by vlmpipe.prompts.Prompt: {id: {"required", "deadline_s"}}
    options: dict[str, dict] = field(default_factory=dict, hash=False, compare=True)


@dataclass(frozen=True)
class _PromptDef:
    prompt: Prompt
    tier: str | None
    required: bool
    deadline_s: float


@dataclass(frozen=True)
class _SetDef:
    id: str
    version: int
    prompts: tuple[_PromptDef, ...]
    routing: dict


def _parse_prompt(d: dict, where: str) -> _PromptDef:
    for k in ("id", "version", "text", "max_tokens", "schema"):
        if k not in d:
            raise PromptSetError(f"{where}: prompt missing '{k}'")
    pid = d["id"]
    schema = d["schema"]
    if not isinstance(schema, dict) or "type" not in schema:
        raise PromptSetError(f"{where}: prompt '{pid}' schema must be a JSON-schema object with 'type'")
    tier = d.get("tier")
    if tier is not None and tier not in TIERS:
        raise PromptSetError(f"{where}: prompt '{pid}' tier must be one of {TIERS}, got {tier!r}")
    if not isinstance(d["max_tokens"], int) or d["max_tokens"] <= 0:
        raise PromptSetError(f"{where}: prompt '{pid}' max_tokens must be a positive integer")
    return _PromptDef(Prompt(pid, int(d["version"]), str(d["text"]).strip(), d["max_tokens"], schema),
                      tier, bool(d.get("required", True)), float(d.get("deadline_s", 180.0)))


def _parse_promptsets(d: dict, path: str) -> tuple[str, int, dict[str, _SetDef]]:
    ss = d.get("shared_system") or {}
    if "version" not in ss or not str(ss.get("text", "")).strip():
        raise PromptSetError(f"{path}: shared_system needs 'version' and non-empty 'text'")
    library: dict[str, _PromptDef] = {}
    for i, p in enumerate(d.get("library") or []):
        pd = _parse_prompt(p, f"{path}: library[{i}]")
        if pd.prompt.id in library:
            raise PromptSetError(f"{path}: duplicate library prompt id '{pd.prompt.id}'")
        library[pd.prompt.id] = pd
    sets_raw = d.get("sets") or {}
    if not sets_raw:
        raise PromptSetError(f"{path}: no sets defined")
    sets: dict[str, _SetDef] = {}
    for sid, s in sets_raw.items():
        where = f"{path}: set '{sid}'"
        if "version" not in s:
            raise PromptSetError(f"{where}: missing 'version'")
        items: list[_PromptDef] = []
        for ref in s.get("prompts") or []:
            if ref not in library:
                raise PromptSetError(f"{where}: unknown prompt ref '{ref}' (not in library)")
            items.append(library[ref])
        for j, p in enumerate(s.get("inline") or []):
            items.append(_parse_prompt(p, f"{where} inline[{j}]"))
        if not items:
            raise PromptSetError(f"{where}: has no prompts")
        ids = [x.prompt.id for x in items]
        dups = sorted({x for x in ids if ids.count(x) > 1})
        if dups:
            raise PromptSetError(f"{where}: duplicate prompt ids {dups}")
        routing = dict(s.get("routing") or {})
        for pid, tier in routing.items():
            if pid not in ids:
                raise PromptSetError(f"{where}: routing for '{pid}' which is not in the set")
            if tier not in TIERS:
                raise PromptSetError(f"{where}: routing '{pid}' must be one of {TIERS}, got {tier!r}")
        sets[sid] = _SetDef(sid, int(s["version"]), tuple(items), routing)
    return str(ss["text"]).strip(), int(ss["version"]), sets


def _parse_cameras(d: dict, path: str, sets: dict[str, _SetDef]) -> tuple[dict, list, dict]:
    def check_set(sid, where):
        if sid is not None and sid not in sets:
            raise PromptSetError(f"{path}: {where} references unknown set '{sid}'")
    defaults = d.get("defaults") or {}
    check_set(defaults.get("set"), "defaults")
    groups = list(d.get("groups") or [])
    for i, g in enumerate(groups):
        if "match" not in g:
            raise PromptSetError(f"{path}: groups[{i}] missing 'match'")
        check_set(g.get("set"), f"group '{g['match']}'")
    cams = dict(d.get("cameras") or {})
    for cid, c in cams.items():
        check_set((c or {}).get("set"), f"camera '{cid}'")
    return defaults, groups, cams


class PromptSetRegistry:
    def __init__(self, promptsets_path: str | Path, cameras_path: str | Path) -> None:
        self.promptsets_path = Path(promptsets_path)
        self.cameras_path = Path(cameras_path)
        self._lock = threading.Lock()
        self._state = self._build()
        self._mtimes = self._current_mtimes()

    @classmethod
    def load(cls, promptsets_path: str | Path = "configs/promptsets.yaml",
             cameras_path: str | Path = "configs/cameras.yaml") -> "PromptSetRegistry":
        return cls(promptsets_path, cameras_path)

    # ---- loading ----
    def _build(self) -> dict:
        ps = yaml.safe_load(self.promptsets_path.read_text()) or {}
        cm = yaml.safe_load(self.cameras_path.read_text()) or {}
        system_text, system_version, sets = _parse_promptsets(ps, str(self.promptsets_path))
        defaults, groups, cams = _parse_cameras(cm, str(self.cameras_path), sets)
        return {"system_text": system_text, "system_version": system_version, "sets": sets,
                "defaults": defaults, "groups": groups, "cameras": cams}

    def _current_mtimes(self) -> tuple[int, int]:
        return (os.stat(self.promptsets_path).st_mtime_ns, os.stat(self.cameras_path).st_mtime_ns)

    def reload_if_changed(self) -> bool:
        """Reload when either file's mtime changed. On a validation error keep the old config."""
        try:
            m = self._current_mtimes()
        except OSError as e:
            log.error("prompt-set config unreadable, keeping previous: %s", e)
            return False
        if m == self._mtimes:
            return False
        try:
            state = self._build()
        except (PromptSetError, yaml.YAMLError, OSError) as e:
            log.error("invalid prompt-set config, keeping previous: %s", e)
            self._mtimes = m          # don't re-parse the same broken file every poll
            return False
        with self._lock:
            self._state, self._mtimes = state, m
        log.info("prompt-set config reloaded")
        return True

    # ---- resolution ----
    @property
    def set_ids(self) -> list[str]:
        return sorted(self._state["sets"])

    def for_camera(self, camera_id: str) -> CameraPromptSet:
        with self._lock:
            st = self._state
        d, cams = st["defaults"], st["cameras"]
        sampling = dict(d.get("sampling") or {})
        set_id = d.get("set")
        group = next((g for g in st["groups"] if fnmatch.fnmatchcase(camera_id, g["match"])), None)
        if group:
            set_id = group.get("set", set_id)
            sampling.update(group.get("sampling") or {})
        if camera_id in cams:
            c = cams[camera_id] or {}
            set_id = c.get("set", set_id)
            sampling.update(c.get("sampling") or {})
        if set_id is None:
            raise KeyError(f"no prompt set for camera '{camera_id}' and no default set")
        s: _SetDef = st["sets"][set_id]
        routing = {p.prompt.id: p.tier for p in s.prompts if p.tier}
        routing.update(s.routing)
        return CameraPromptSet(
            camera_id=camera_id, set_id=s.id, set_version=s.version,
            registry=Registry(st["system_text"], st["system_version"], tuple(p.prompt for p in s.prompts)),
            routing=routing, sampling=sampling,
            options={p.prompt.id: {"required": p.required, "deadline_s": p.deadline_s} for p in s.prompts})
