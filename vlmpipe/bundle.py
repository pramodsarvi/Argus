"""Run a prompt bundle against one vLLM replica: warm the shared prefix, then fan out.

Modes (for the week-1 spike):
  concurrent - fire all prompts at once (expected: poor cache reuse)
  warm       - 1-token warm request on the prefix, then fire all prompts
  merged     - single request answering every prompt
"""
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass

from openai import AsyncOpenAI

from .prompts import Prompt, Registry, build_messages, merged_question


@dataclass
class Result:
    prompt_id: str
    ok: bool
    output: dict | str | None
    ttft_s: float
    total_s: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    error: str | None = None


# How JSON output is requested (measured on Qwen2.5-VL-7B-FP8, vLLM 0.33, xgrammar):
#   "prompt"  (default) schema is put in the prompt; reply is parsed tolerantly and validated.
#   "grammar" response_format=json_schema (guided decoding). With any-whitespace grammar the model
#             loops on newlines until max_tokens; with disable_any_whitespace it emits degenerate
#             near-empty JSON (e.g. {"summary": ""}). Use only for models/servers where verified.
STRUCTURED_MODE = os.environ.get("STRUCTURED_MODE", "prompt")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def with_schema_instruction(messages: list, schema: dict) -> list:
    """Append the JSON-only instruction + schema to the last user message (after the media, so
    the cached media prefix is unaffected)."""
    note = ("\n\nReply with ONLY a compact JSON object (no code fences, no prose) matching this "
            "JSON Schema:\n" + json.dumps(schema, separators=(",", ":")))
    out = [dict(m) for m in messages]
    last = out[-1]
    if isinstance(last["content"], list):
        last["content"] = [*last["content"], {"type": "text", "text": note.lstrip()}]
    else:
        last["content"] = f"{last['content']}{note}"
    return out


def parse_json_reply(raw: str, schema: dict | None = None):
    """Tolerant parse: strips code fences / surrounding prose, then validates against `schema`
    when jsonschema is installed. Raises ValueError (message starts with the cause)."""
    t = _FENCE.sub("", raw.strip()).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise
        obj = json.loads(t[i:j + 1])
    if schema:
        try:
            import jsonschema
        except ImportError:
            return obj
        try:
            jsonschema.validate(obj, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"schema: {e.message[:200]}") from e
    return obj


def _schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


async def _call(client: AsyncOpenAI, model: str, messages: list, max_tokens: int,
                pid: str, schema: dict | None, timeout: float,
                mm_processor_kwargs: dict | None = None, structured: str | None = None) -> Result:
    mode = structured or STRUCTURED_MODE
    if schema and mode == "prompt":
        messages = with_schema_instruction(messages, schema)
    t0 = time.perf_counter()
    ttft = None
    text = []
    usage = None
    try:
        kw = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0,
                  stream=True, stream_options={"include_usage": True}, timeout=timeout)
        if schema and mode == "grammar":
            kw["response_format"] = _schema_format(pid, schema)
        if mm_processor_kwargs:
            # vLLM-specific request field (e.g. {"fps": 1.0} for Qwen2.5-VL video). Must be
            # identical across a bundle or the multimodal hash / prefix may differ.
            kw["extra_body"] = {"mm_processor_kwargs": mm_processor_kwargs}
        stream = await client.chat.completions.create(**kw)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(chunk.choices[0].delta.content)
            if chunk.usage:
                usage = chunk.usage
    except Exception as e:  # noqa: BLE001 - surfaced in Result
        return Result(pid, False, None, ttft or 0.0, time.perf_counter() - t0, 0, 0, 0, repr(e))
    total = time.perf_counter() - t0
    raw = "".join(text)
    out, ok, err = raw, True, None
    if schema:
        try:
            out = parse_json_reply(raw, schema) if mode == "prompt" else json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            ok, err = False, f"invalid json: {e}"
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return Result(pid, ok, out, ttft if ttft is not None else total, total,
                  usage.prompt_tokens if usage else 0,
                  (getattr(details, "cached_tokens", 0) or 0) if details else 0,
                  usage.completion_tokens if usage else 0, err)


async def run_bundle(client: AsyncOpenAI, model: str, reg: Registry, frames: list,
                     mode: str = "warm", timeout: float = 300.0,
                     prompts: tuple[Prompt, ...] | None = None, input_mode: str = "images",
                     fps: float = 1.0, mm_processor_kwargs: dict | None = None) -> dict:
    prompts = prompts or reg.prompts

    def msgs(q):
        return build_messages(reg.system_text, frames, q, input_mode=input_mode, fps=fps)

    def call(m, max_tokens, pid, schema):
        return _call(client, model, m, max_tokens, pid, schema, timeout, mm_processor_kwargs)

    t0 = time.perf_counter()
    warm = None
    if mode == "merged":
        q, schema = merged_question(prompts)
        res = [await call(msgs(q), sum(p.max_tokens for p in prompts), "merged", schema)]
    else:
        if mode == "warm":
            warm = await call(msgs(None), 1, "_warm", None)
        elif mode != "concurrent":
            raise ValueError(mode)
        res = await asyncio.gather(*[
            call(msgs(p.text), p.max_tokens, p.id, p.schema) for p in prompts])
    return {"mode": mode, "input_mode": input_mode, "wall_s": time.perf_counter() - t0,
            "warm": asdict(warm) if warm else None,
            "results": [asdict(r) for r in res]}
