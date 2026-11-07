import asyncio
from pathlib import Path

import httpx
import pytest

from vlmpipe.routing import AdmissionController, Replica, Router, parse_metrics, rank

R = [Replica("a0", "A", 0, "http://a:8000/v1"), Replica("a1", "A", 1, "http://a:8001/v1"),
     Replica("b0", "B", 0, "http://b:8000/v1")]

METRICS_V1 = """# HELP vllm:num_requests_waiting x
vllm:num_requests_waiting{model_name="m"} 3.0
vllm:num_requests_running{model_name="m"} 5.0
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:prefix_cache_hits_total{model_name="m"} 750.0
vllm:prefix_cache_queries_total{model_name="m"} 1000.0
"""
METRICS_V0 = """vllm:num_requests_waiting{model_name="m"} 1
vllm:gpu_cache_usage_perc{model_name="m"} 42
"""


def test_parse_v1():
    m = parse_metrics(METRICS_V1)
    assert (m.waiting, m.running, m.kv_usage) == (3, 5, 0.42)
    assert m.prefix_hit_rate == 0.75


def test_parse_v0_variants_and_garbage():
    m = parse_metrics(METRICS_V0 + "garbage line\n")
    assert m.waiting == 1 and m.kv_usage == pytest.approx(0.42)
    assert m.prefix_hit_rate is None


def test_rank_stable_and_minimal_movement():
    keys = [f"vid{i}" for i in range(300)]
    before = {k: rank(k, R)[0].name for k in keys}
    assert before == {k: rank(k, R)[0].name for k in keys}
    after = {k: rank(k, R[:2])[0].name for k in keys}       # remove b0
    moved = [k for k in keys if before[k] != after[k]]
    assert all(before[k] == "b0" for k in moved)            # only b0's keys move
    assert len(set(before.values())) == 3


def make_router(down=(), waiting=None):
    waiting = waiting or {}

    def handler(req: httpx.Request):
        host = f"{req.url.host}:{req.url.port}"
        name = {"a:8000": "a0", "a:8001": "a1", "b:8000": "b0"}[host]
        if name in down:
            raise httpx.ConnectError("down")
        if req.url.path == "/health":
            return httpx.Response(200)
        w = waiting.get(name, [0])
        v = w.pop(0) if len(w) > 1 else w[0]
        return httpx.Response(200, text=f"vllm:num_requests_waiting {v}\n")

    return Router(R, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), health_ttl_s=0)


def test_route_prefers_local_node():
    async def go():
        r = await make_router().route("vid1", "B")
        assert r.replica.name == "b0" and not r.failover and not r.cross_node
    asyncio.run(go())


def test_failover_same_node_then_cross_node():
    async def go():
        key = next(k for k in (f"v{i}" for i in range(100)) if rank(k, R[:2])[0].name == "a0")
        r = await make_router(down={"a0"}).route(key, "A")
        assert r.replica.name == "a1" and r.failover and not r.cross_node
        r = await make_router(down={"a0", "a1"}).route(key, "A")
        assert r.replica.name == "b0" and r.failover and r.cross_node
        with pytest.raises(RuntimeError):
            await make_router(down={"a0", "a1", "b0"}).route(key, "A")
    asyncio.run(go())


def test_from_yaml():
    r = Router.from_yaml(Path(__file__).parent.parent / "configs" / "replicas.yaml")
    assert len(r.replicas) == 4 and len({x.node for x in r.replicas}) == 2


def test_admission_waits_for_queue_and_caps_inflight():
    async def go():
        router = make_router(waiting={"a0": [20, 20, 2]})
        ac = AdmissionController(router, max_inflight=1, target_waiting=8, poll_s=0.01)
        async with ac.slot(R[0]):
            pass                                  # had to wait out two high readings
        # inflight cap: second slot blocks while first held
        router2 = make_router()
        ac2 = AdmissionController(router2, max_inflight=1, poll_s=0.01)
        order = []

        async def job(i, hold):
            async with ac2.slot(R[0]):
                order.append(f"in{i}")
                await asyncio.sleep(hold)
                order.append(f"out{i}")
        await asyncio.gather(job(1, 0.05), job(2, 0))
        assert order == ["in1", "out1", "in2", "out2"]
    asyncio.run(go())


def test_admission_timeout():
    async def go():
        ac = AdmissionController(make_router(waiting={"a0": [99]}), target_waiting=1,
                                 poll_s=0.01, max_wait_s=0.05)
        with pytest.raises(TimeoutError):
            async with ac.slot(R[0]):
                pass
        async with asyncio.timeout(1):            # semaphore was released on failure
            await ac._sems["a0"].acquire()
    asyncio.run(go())
