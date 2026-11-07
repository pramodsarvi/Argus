"""One pooled AsyncOpenAI client per (base_url, event loop), reused across chunks/requests.

Creating a client per chunk (the old behaviour) opens a fresh connection pool every time. Auth:
VLLM_API_KEY (or OPENAI_API_KEY) when the server runs with --api-key; "EMPTY" otherwise.
"""
import asyncio
import os

_CLIENTS: dict[tuple[str, int], object] = {}


def get_client(base_url: str):
    import httpx
    from openai import AsyncOpenAI
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = 0
    key = (base_url, loop_id)
    c = _CLIENTS.get(key)
    if c is None:
        if len(_CLIENTS) > 8:                       # loops come and go in tests; don't leak
            _CLIENTS.clear()
        c = _CLIENTS[key] = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            max_retries=0,                           # retries are owned by SegmentRunner/Temporal
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
                timeout=httpx.Timeout(300.0, connect=5.0)))
    return c
