"""Run a prompt bundle against one vLLM replica: warm the shared prefix, then fan out.

Modes (for the week-1 spike):
  concurrent - fire all prompts at once (expected: poor cache reuse)
  warm       - 1-token warm request on the prefix, then fire all prompts
  merged     - single request answering every prompt
"""
import asyncio
import json
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


def _schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}


async def _call(client: AsyncOpenAI, model: str, messages: list, max_tokens: int,
                pid: str, schema: dict | None, timeout: float,
                mm_processor_kwargs: dict | None = None) -> Result:
    t0 = time.perf_counter()
    ttft = None
    text = []
    usage = None
    try:
        kw = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0,
                  stream=True, stream_options={"include_usage": True}, timeout=timeout)
        if schema:
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
            out = json.loads(raw)
        except json.JSONDecodeError as e:
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
