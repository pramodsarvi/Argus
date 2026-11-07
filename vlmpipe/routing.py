"""Cache-affinity routing, health/failover, vLLM metrics scraping, admission control.

Prefix cache is per replica, so every request for a video_id must hit the same replica.
Rendezvous (HRW) hashing keeps assignments stable when replicas come and go.
Failover order: healthy replicas on the frame-holding node, then any node (frames must then be
reachable there, e.g. via object-store fallback). Any failover means the bundle must RE-WARM.
"""
import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml


@dataclass(frozen=True)
class Replica:
    name: str
    node: str
    gpu: int
    url: str          # OpenAI base, e.g. http://node-a:8000/v1

    @property
    def root(self) -> str:
        return self.url.rsplit("/v1", 1)[0]


@dataclass
class Route:
    replica: Replica
    failover: bool          # True -> not the preferred replica; caller must re-warm
    cross_node: bool        # True -> frames are not local to this replica's node


def _score(key: str, replica: Replica) -> int:
    return int.from_bytes(hashlib.blake2b(f"{key}|{replica.name}".encode(), digest_size=8).digest(), "big")


def rank(key: str, replicas: list[Replica]) -> list[Replica]:
    return sorted(replicas, key=lambda r: _score(key, r), reverse=True)


@dataclass
class Metrics:
    waiting: float = 0.0
    running: float = 0.0
    kv_usage: float = 0.0               # 0..1
    prefix_hits: float | None = None    # cumulative counters (tokens or queries, version-dependent)
    prefix_queries: float | None = None

    @property
    def prefix_hit_rate(self) -> float | None:
        if self.prefix_hits is None or not self.prefix_queries:
            return None
        return self.prefix_hits / self.prefix_queries


_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)')

# Name variants across vLLM versions (V0 vs V1, counters with/without _total).
_WAITING = ("vllm:num_requests_waiting",)
_RUNNING = ("vllm:num_requests_running",)
_KV = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
_HITS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits_total")
_QUERIES = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries",
            "vllm:gpu_prefix_cache_queries_total")


def parse_metrics(text: str) -> Metrics:
    vals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        vals[m.group(1)] = vals.get(m.group(1), 0.0) + v   # sum across label sets

    def first(names, default=None):
        for n in names:
            if n in vals:
                return vals[n]
        return default

    kv = first(_KV, 0.0)
    if kv > 1.0:            # some versions report percent
        kv /= 100.0
    return Metrics(first(_WAITING, 0.0), first(_RUNNING, 0.0), kv, first(_HITS), first(_QUERIES))


@dataclass
class _Health:
    healthy: bool = True
    checked_at: float = 0.0
    metrics: Metrics = field(default_factory=Metrics)


class Router:
    def __init__(self, replicas: list[Replica], client: httpx.AsyncClient | None = None,
                 health_ttl_s: float = 5.0, timeout_s: float = 2.0):
        if len({r.name for r in replicas}) != len(replicas):
            raise ValueError("duplicate replica names")
        self.replicas = replicas
        self.client = client or httpx.AsyncClient(timeout=timeout_s)
        self.health_ttl_s = health_ttl_s
        self._h: dict[str, _Health] = {r.name: _Health() for r in replicas}

    @classmethod
    def from_yaml(cls, path: str | Path, **kw) -> "Router":
        d = yaml.safe_load(Path(path).read_text())
        return cls([Replica(r["name"], r["node"], int(r["gpu"]), r["url"]) for r in d["replicas"]], **kw)

    async def probe(self, r: Replica) -> _Health:
        h = self._h[r.name]
        try:
            ok = (await self.client.get(f"{r.root}/health")).status_code == 200
            if ok:
                resp = await self.client.get(f"{r.root}/metrics")
                if resp.status_code == 200:
                    h.metrics = parse_metrics(resp.text)
            h.healthy = ok
        except httpx.HTTPError:
            h.healthy = False
        h.checked_at = time.monotonic()
        return h

    async def health(self, r: Replica) -> _Health:
        h = self._h[r.name]
        if time.monotonic() - h.checked_at > self.health_ttl_s:
            await self.probe(r)
        return h

    def mark_down(self, r: Replica) -> None:
        """Call on request failure so the next route() fails over immediately."""
        self._h[r.name].healthy = False
        self._h[r.name].checked_at = time.monotonic()

    async def route(self, video_id: str, frames_node: str | None) -> Route:
        local = [r for r in self.replicas if frames_node is None or r.node == frames_node]
        remote = [r for r in self.replicas if r not in local]
        order = rank(video_id, local) + rank(video_id, remote)
        preferred = order[0] if order else None
        for r in order:
            if (await self.health(r)).healthy:
                return Route(r, failover=r is not preferred,
                             cross_node=frames_node is not None and r.node != frames_node)
        raise RuntimeError(f"no healthy replica for {video_id}")

    def metrics(self, r: Replica) -> Metrics:
        return self._h[r.name].metrics


class AdmissionController:
    """Per-replica concurrency cap + queue-depth throttle.

    Keeps vLLM saturated (small non-zero queue) without piling up requests that would evict
    each other's prefix KV between warm and fan-out.
    """

    def __init__(self, router: Router, max_inflight: int = 32, target_waiting: int = 8,
                 poll_s: float = 0.5, max_wait_s: float = 600.0):
        self.router = router
        self.target_waiting = target_waiting
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self._sems = {r.name: asyncio.Semaphore(max_inflight) for r in router.replicas}

    async def _wait_for_queue(self, r: Replica) -> None:
        deadline = time.monotonic() + self.max_wait_s
        while (await self.router.health(r)).metrics.waiting > self.target_waiting:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{r.name} queue above {self.target_waiting} for {self.max_wait_s}s")
            await asyncio.sleep(self.poll_s)
            await self.router.probe(r)

    def slot(self, r: Replica) -> "_Slot":
        return _Slot(self, r)


class _Slot:
    def __init__(self, ac: AdmissionController, r: Replica):
        self.ac, self.r = ac, r

    async def __aenter__(self):
        await self.ac._sems[self.r.name].acquire()
        try:
            await self.ac._wait_for_queue(self.r)
        except BaseException:
            self.ac._sems[self.r.name].release()
            raise
        return self.r

    async def __aexit__(self, *exc):
        self.ac._sems[self.r.name].release()
