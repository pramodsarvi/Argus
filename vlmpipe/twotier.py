"""Two-tier analysis (design §10.1, §10.7): describe once with the VLM, answer in text.

Visual tier: one dense structured description per segment, built on the same shared media
prefix as every other request (so it also warms the prefix cache for escalations).
Text tier:   use cases answered from the description by a (cheaper) text model - by default
             the same vLLM endpoint, text-only. Escalate to a direct visual call when the
             use case is routed `visual` or the text answer's confidence is below threshold.
"""
import asyncio
import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .bundle import Result, _call
from .prompts import Prompt, Registry, build_messages

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "describe.yaml"


@dataclass(frozen=True)
class Route:
    tier: str                      # "text" | "visual"
    question: str | None = None


@dataclass(frozen=True)
class TwoTierConfig:
    version: int
    threshold: float
    describe: Prompt
    text_system: str
    routing: dict = field(hash=False)
    reduce_text: str = ""
    reduce_schema: dict = field(default_factory=dict, hash=False)
    reduce_max_tokens: int = 768

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "TwoTierConfig":
        d = yaml.safe_load(Path(path).read_text())
        ds = d["describe"]
        routing = {k: Route(v.get("tier", "visual"), (v.get("question") or "").strip() or None)
                   for k, v in d.get("routing", {}).items()}
        for k, r in routing.items():
            if r.tier not in ("text", "visual"):
                raise ValueError(f"bad tier for {k}: {r.tier}")
        red = d.get("reduce", {})
        return cls(d["version"], float(d.get("confidence_threshold", 0.6)),
                   Prompt(ds["id"], ds["version"], ds["text"].strip(), ds["max_tokens"], ds["schema"]),
                   d["text_system"].strip(), routing, red.get("text", "").strip(),
                   red.get("schema", {}), red.get("max_tokens", 768))

    def route(self, prompt_id: str, overrides: dict[str, str] | None = None) -> Route:
        """Routing precedence: per-call overrides > describe.yaml routing > "visual".

        Unknown prompt ids (e.g. camera-specific prompts) default to "visual": the safe choice,
        since an unvalidated text-tier answer could silently miss pixel-level detail. Opt a
        prompt into the cheap text tier explicitly via `routing_overrides` or describe.yaml."""
        base = self.routing.get(prompt_id, Route("visual"))
        if overrides and prompt_id in overrides:
            tier = overrides[prompt_id]
            if tier not in ("text", "visual"):
                raise ValueError(f"bad tier override for {prompt_id}: {tier}")
            return Route(tier, base.question)
        return base


def with_confidence(schema: dict) -> dict:
    """Prompt schema + required numeric `confidence` in [0, 1]."""
    s = copy.deepcopy(schema) if schema else {"type": "object", "properties": {}}
    s.setdefault("properties", {})["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    s["required"] = list(dict.fromkeys([*s.get("required", []), "confidence"]))
    return s


MAX_PREVIOUS_CHARS = 4000


def describe_question(cfg: "TwoTierConfig", previous_description: dict | str | None = None) -> str:
    """Describe instruction (+ previous chunk context). Goes AFTER the media, so the cached
    media prefix is unaffected by per-chunk context."""
    q = cfg.describe.text
    if previous_description:
        prev = (previous_description if isinstance(previous_description, str)
                else json.dumps(previous_description, ensure_ascii=False, separators=(",", ":")))
        if len(prev) > MAX_PREVIOUS_CHARS:
            prev = prev[:MAX_PREVIOUS_CHARS] + "...(truncated)"
        q += ("\n\nContext: description of the PREVIOUS chunk from the same camera (JSON). "
              "Use it only to identify events that continue into this chunk; set "
              "continues_previous and list ongoing_events accordingly. Describe only what is "
              "visible in THIS chunk.\n" + prev)
    return q


async def describe_segment(client, model: str, system_text: str, frames: list,
                           cfg: TwoTierConfig | None = None, timeout: float = 300.0,
                           input_mode: str = "images", fps: float = 1.0,
                           mm_processor_kwargs: dict | None = None,
                           previous_description: dict | str | None = None) -> Result:
    cfg = cfg or TwoTierConfig.load()
    d = cfg.describe
    msgs = build_messages(system_text, frames, describe_question(cfg, previous_description),
                          input_mode=input_mode, fps=fps)
    return await _call(client, model, msgs, d.max_tokens, d.id, d.schema, timeout,
                       mm_processor_kwargs=mm_processor_kwargs)


def _text_messages(system: str, description: dict | str, question: str) -> list[dict]:
    desc = description if isinstance(description, str) else json.dumps(description, ensure_ascii=False)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Segment description (JSON):\n{desc}\n\nQuestion: {question}"}]


async def answer_from_description(text_client, text_model: str, description: dict | str,
                                  prompt: Prompt, cfg: TwoTierConfig | None = None,
                                  timeout: float = 120.0) -> Result:
    cfg = cfg or TwoTierConfig.load()
    q = cfg.route(prompt.id).question or prompt.text   # camera prompts: their own text
    return await _call(text_client, text_model, _text_messages(cfg.text_system, description, q),
                       prompt.max_tokens + 16, prompt.id, with_confidence(prompt.schema), timeout)


def _confidence(r: Result) -> float:
    if r.ok and isinstance(r.output, dict):
        try:
            return float(r.output.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


async def run_two_tier(client, model: str, reg: Registry, frames: list,
                       text_client=None, text_model: str | None = None,
                       cfg: TwoTierConfig | None = None, threshold: float | None = None,
                       timeout: float = 300.0, prompts: tuple[Prompt, ...] | None = None,
                       input_mode: str = "images", fps: float = 1.0,
                       mm_processor_kwargs: dict | None = None,
                       previous_description: dict | str | None = None,
                       routing_overrides: dict[str, str] | None = None) -> dict:
    """Same shape as bundle.run_bundle (+ 'description', 'escalated', 'tiers').

    previous_description: the previous chunk's description from the same camera; added as text
        context after the media so events spanning chunk boundaries can be linked.
    routing_overrides: prompt_id -> "text"|"visual"; overrides describe.yaml (camera prompt sets)."""
    cfg = cfg or TwoTierConfig.load()
    thr = cfg.threshold if threshold is None else threshold
    text_client = text_client or client
    text_model = text_model or model
    prompts = prompts or reg.prompts
    t0 = time.perf_counter()

    async def visual(p: Prompt) -> Result:
        msgs = build_messages(reg.system_text, frames, p.text, input_mode=input_mode, fps=fps)
        return await _call(client, model, msgs, p.max_tokens, p.id, p.schema, timeout,
                           mm_processor_kwargs=mm_processor_kwargs)

    # 1) describe once (also warms the shared media prefix for any visual calls)
    desc = await describe_segment(client, model, reg.system_text, frames, cfg, timeout,
                                  input_mode=input_mode, fps=fps,
                                  mm_processor_kwargs=mm_processor_kwargs,
                                  previous_description=previous_description)
    description = desc.output if desc.ok else None

    # 2) visual-routed prompts run immediately (prefix is warm); text-routed answered from text
    async def one(p: Prompt) -> tuple[Result, str, bool]:
        route = cfg.route(p.id, routing_overrides)
        if route.tier == "visual" or description is None:
            return await visual(p), "visual", description is None and route.tier == "text"
        r = await answer_from_description(text_client, text_model, description, p, cfg, timeout)
        if _confidence(r) < thr:
            return await visual(p), "visual", True
        if isinstance(r.output, dict):
            r.output.pop("confidence", None)   # keep output conformant to the prompt schema
        return r, "text", False

    triples = await asyncio.gather(*[one(p) for p in prompts])
    return {"mode": "two_tier", "wall_s": time.perf_counter() - t0,
            "warm": asdict(desc), "description": description,
            "results": [asdict(r) for r, _, _ in triples],
            "tiers": {r.prompt_id: t for r, t, _ in triples},
            "escalated": [r.prompt_id for r, _, e in triples if e]}


async def reduce_video(text_client, text_model: str, segment_descriptions: list,
                       cfg: TwoTierConfig | None = None, timeout: float = 300.0) -> Result:
    """Hierarchical video summary + cross-segment events from ordered segment descriptions.

    segment_descriptions: list of {"segment_idx", "start_s", "description"} (or bare dicts,
    in which case index/start are inferred from order)."""
    cfg = cfg or TwoTierConfig.load()
    items = []
    for i, s in enumerate(segment_descriptions):
        if isinstance(s, dict) and "description" in s:
            items.append({"segment": s.get("segment_idx", i), "start_s": s.get("start_s"),
                          "description": s["description"]})
        else:
            items.append({"segment": i, "start_s": None, "description": s})
    msgs = [{"role": "system", "content": cfg.text_system},
            {"role": "user", "content": cfg.reduce_text + "\n\nSegments (JSON):\n" +
             json.dumps(items, ensure_ascii=False)}]
    return await _call(text_client, text_model, msgs, cfg.reduce_max_tokens, "_reduce",
                       cfg.reduce_schema, timeout)
