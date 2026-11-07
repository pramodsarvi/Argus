"""No-GPU end-to-end smoke test of the camera pipeline.

Real: ffmpeg (synthetic chunks + sampling), ingest watcher + SQLite ledger, a local Temporal dev
server, the real worker/activities/workflow, the OpenAI client, result validation + store,
chunk assembler, camera rules.
Fake: the VLM. A tiny local OpenAI-compatible server answers chat completions with
schema-conformant JSON (it checks the media prefix is identical across requests of a chunk).
Not covered: real model quality, prefix-cache behaviour, Kafka, Postgres, deploy manifests.

    python scripts/smoke_e2e.py [bundle|two_tier]   # from the repo root; exit code 0 = pass
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

MODE = sys.argv[1] if len(sys.argv) > 1 else "bundle"        # bundle (default) | two_tier
CAMERA = "dock-07"
N_CHUNKS = 3


# ---------------------------------------------------------------------------- fake VLM server
def instance(schema: dict, name: str = ""):
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object" or "properties" in schema:
        return {k: instance(v, k) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        item = instance(schema.get("items", {}), name)
        return [item] if name in ("violations",) else []
    if t == "string":
        return "x"
    if t == "integer":
        return 1
    if t == "number":
        return 0.9
    if t == "boolean":
        return True
    return None


class Fake(BaseHTTPRequestHandler):
    log: list = []
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        user = msgs[1]["content"] if len(msgs) > 1 and isinstance(msgs[1]["content"], list) else []
        # default "prompt" mode: the schema is appended to the last text part of the user turn
        # ("...JSON Schema:\n{...}"); a request without one is a warm request.
        marker = "JSON Schema:\n"
        schema = None
        for part in reversed(user):
            if part.get("type") == "text" and marker in part["text"]:
                schema = json.loads(part["text"].split(marker, 1)[1])
                break
        if schema is None:
            name, text = "_warm", "x"
        else:
            props = set(schema.get("properties", {}))
            name = "_describe" if "ongoing_events" in props else (sorted(props)[0] if props else "_none")
            text = json.dumps(instance(schema))
        media = [p for p in user if p.get("type") != "text" or p["text"].startswith("Frame ")]   # frames + labels
        with Fake.lock:
            urls = [p["image_url"]["url"] for p in user if p.get("type") == "image_url"]
            exist = all(os.path.exists(u[len("file://"):]) for u in urls)
            Fake.log.append({"name": name, "system": msgs[0]["content"], "media": media, "exist": exist,
                             "has_images": any(p.get("type") == "image_url" for p in user)})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        half = max(1, len(text) // 2)
        for piece in (text[:half], text[half:]):
            self.wfile.write(b"data: " + json.dumps({
                "id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
            }).encode() + b"\n\n")
        self.wfile.write(b"data: " + json.dumps({
            "id": "c", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
            "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                                     "prompt_tokens_details": {"cached_tokens": 64}}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


# ---------------------------------------------------------------------------- the test
def make_chunk(path: Path, seconds: int = 24) -> None:
    # distinct scenes (different luminance structure; dHash dedup is luminance-only, so a
    # hue-only change would be collapsed to one frame)
    srcs = ["testsrc2", "mandelbrot", "smptebars", "life=ratio=0.3:death_color=#000000:life_color=#ffffff"]
    seg = seconds // len(srcs)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for sname in srcs:
        cmd += ["-f", "lavfi", "-i", f"{sname}=size=640x360:rate=10" if "=" not in sname
                else f"{sname}:size=640x360:rate=10"]
    # -t per input via trim in filter graph
    fc = "".join(f"[{k}:v]trim=duration={seg},setpts=PTS-STARTPTS,format=yuv420p[v{k}];"
                 for k in range(len(srcs)))
    fc += "".join(f"[v{k}]" for k in range(len(srcs))) + f"concat=n={len(srcs)}:v=1[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


async def main() -> int:
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from vlmpipe.ingest.watcher import Watcher
    from vlmpipe.orchestration.activities import Activities
    from vlmpipe.orchestration.camera import default_config, submit_chunk
    from vlmpipe.orchestration.workflows import CameraWorkflow, VideoAnalysisWorkflow
    from vlmpipe.results.consumer import default_schemas, process_message
    from vlmpipe.results.publisher import AsyncRecordPublisher, InMemoryPublisher
    from vlmpipe.results.service import build_default
    from vlmpipe.results.store import SQLiteStore

    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    problems: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            problems.append(msg)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ.update(VLLM_BASE_URL=f"http://127.0.0.1:{srv.server_port}/v1", MODEL="fake-vlm",
                      PROMPTSETS_PATH=str(ROOT / "configs/promptsets.yaml"),
                      FRAMES_DIR=str(tmp / "frames"))
    os.environ.pop("KAFKA_BOOTSTRAP", None)

    inbox, work = tmp / "work/inbox", tmp / "work"
    inbox.mkdir(parents=True)
    now = int(time.time())
    starts = [now - 30 + 10 * i for i in range(N_CHUNKS)]       # near real time => not degraded/stale
    src = tmp / "src.mp4"
    make_chunk(src)

    inner = InMemoryPublisher()
    print("starting local Temporal dev server ...")
    async with await WorkflowEnvironment.start_local() as env:
        client = env.client
        node = "smoke"
        acts = Activities(publisher=AsyncRecordPublisher(inner), heartbeat_every_s=2.0)
        async with Worker(client, task_queue=f"node-{node}",
                          workflows=[CameraWorkflow, VideoAnalysisWorkflow], activities=acts.all()):
            async def submit(cam, path, ts, nd):
                cfg = default_config(cam, nd, idle_timeout_s=6.0, reorder_window_s=1.0,
                                    analysis_mode=MODE)
                await submit_chunk(client, cam, path, ts, nd, config=cfg)

            w = Watcher(inbox, submit, work_root=work, stable_s=1.0, node=node)
            for ts in starts:                                    # camera semantics: tmp name, then rename
                shutil.copy(src, inbox / f".{CAMERA}_{ts}.tmp")
                os.rename(inbox / f".{CAMERA}_{ts}.tmp", inbox / f"{CAMERA}_{ts}.mp4")
            (inbox / "garbage_1700000000.mp4").write_bytes(b"not a video" * 100)

            submitted, t0 = 0, time.time()
            while submitted < N_CHUNKS and time.time() - t0 < 60:
                submitted += await w.step()
                await asyncio.sleep(0.5)
            check(submitted == N_CHUNKS, f"watcher admitted + submitted {submitted}/{N_CHUNKS} chunks")
            check(any((work / "rejected").glob("garbage*")), "corrupt file moved to rejected/")

            from vlmpipe.orchestration.types import CameraSummary
            summary = await asyncio.wait_for(
                client.get_workflow_handle(f"camera-{CAMERA}", result_type=CameraSummary).result(),
                timeout=240)
            check(summary.total_chunks == N_CHUNKS and summary.ended == "idle",
                  f"workflow processed {summary.total_chunks} chunks, ended={summary.ended}")
            check([r.status for r in summary.recent] == ["complete"] * N_CHUNKS,
                  f"chunk statuses {[r.status for r in summary.recent]}")

    # ---- inspect what the fake VLM saw
    reqs = Fake.log
    check(all(r["exist"] for r in reqs if r["has_images"]), "image files existed when the VLM read them")
    medias = {json.dumps(r["media"]) for r in reqs if r["has_images"]}
    check(len(medias) == N_CHUNKS, f"{len(medias)} distinct frame sets for {N_CHUNKS} chunks")
    n_frames = max(sum(1 for p in json.loads(m) if p.get("type") == "image_url") for m in medias)
    check(n_frames >= 2, f"frames sent per chunk: {n_frames}")
    if MODE == "two_tier":
        d = [r for r in reqs if r["name"] == "_describe"]
        check(len(d) == N_CHUNKS, f"one describe call per chunk ({len(d)})")
    else:
        warm = [r for r in reqs if r["name"] == "_warm"]
        check(not warm, "no warm request for a small prefix (auto-warm policy)")
        check(len([r for r in reqs if r["name"] != "_warm"]) >= N_CHUNKS * 5,
              f"{len(reqs)} model requests for {N_CHUNKS} chunks x 5 prompts")
    check(not any(Path(tmp / "frames").rglob("*.png")), "sampled frames cleaned up after analysis")

    # ---- results layer
    store = SQLiteStore(str(tmp / "results.db"))
    asm = build_default(str(ROOT / "configs/rules.yaml"), None, state_store=store)
    alerts: list[dict] = []
    asm.on_alert = alerts.append
    schemas = default_schemas()
    statuses = defaultdict(int)
    for _topic, _key, value in inner.messages:
        rec = process_message(value, store, schemas)
        statuses[rec.effective_status()] += 1
        asm.add(rec)
    n_prompts = 5
    check(len(inner.messages) == N_CHUNKS * n_prompts, f"{len(inner.messages)} results published "
          f"(expected {N_CHUNKS * n_prompts})")
    check(set(statuses) == {"ok"}, f"all results validate against schemas: {dict(statuses)}")
    check(len(store.get_camera(CAMERA)) == N_CHUNKS * n_prompts, "all results stored under the camera")
    check(not asm.pending, "no chunks left unassembled")
    ids = {a["rule_id"] for a in alerts}
    check("persistent_safety_violation" in ids, f"consecutive rule fired: {sorted(ids)}")
    check("recurring_anomaly" in ids, "window rule fired")
    ts_ok = {round(r.chunk_start_ts) for r in store.get_camera(CAMERA)} == set(starts)
    check(ts_ok, "chunk_start_ts matches the filenames")

    srv.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
