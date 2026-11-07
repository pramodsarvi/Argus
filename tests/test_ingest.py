import asyncio
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vlmpipe.ingest.naming import parse_chunk, parse_ts
from vlmpipe.ingest.watcher import Watcher, ffprobe_ok

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("src") / "tiny.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=64x64:rate=5:duration=1", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeSubmit:
    def __init__(self, fail_times=0):
        self.calls, self.fail_times = [], fail_times

    async def __call__(self, cam, path, ts, node):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("temporal down")
        self.calls.append((cam, Path(path).name, ts, node))


def make_watcher(tmp_path, submit, clock, **kw):
    return Watcher(tmp_path / "inbox", submit, work_root=tmp_path, stable_s=5, node="n1",
                   clock=clock, backoff_base_s=10, **kw)


def settle(w, clock):
    """Two scans with time advanced past stable_s."""
    w.scan()
    clock.t += 10
    return w.scan()


# ---------- naming ----------
def test_parse_iso_and_epoch(tmp_path):
    a = tmp_path / "cam01_20261107T120300Z.mp4"; a.write_bytes(b"x")
    b = tmp_path / "cam02_1793995380.mkv"; b.write_bytes(b"x")
    ra, rb = parse_chunk(a), parse_chunk(b)
    assert ra.camera_id == "cam01"
    assert ra.start_ts == datetime(2026, 11, 7, 12, 3, tzinfo=timezone.utc).timestamp()
    assert (rb.camera_id, rb.start_ts) == ("cam02", 1793995380.0)
    assert parse_ts("20261107T120300") == ra.start_ts      # no Z => UTC


def test_parse_fallback_parent_dir(tmp_path):
    inbox = tmp_path / "inbox"; (inbox / "dock-3").mkdir(parents=True)
    f = inbox / "dock-3" / "whatever.mp4"; f.write_bytes(b"x")
    r = parse_chunk(f, inbox=inbox)
    assert r.camera_id == "dock-3" and r.start_ts == f.stat().st_mtime
    g = inbox / "loose.mp4"; g.write_bytes(b"x")
    assert parse_chunk(g, inbox=inbox) is None            # no camera derivable
    assert parse_chunk(f, fallback=False) is None


def test_custom_pattern(tmp_path):
    f = tmp_path / "site-A.camX.1793995380.ts"; f.write_bytes(b"x")
    r = parse_chunk(f, pattern=r"\.(?P<camera_id>cam\w+)\.(?P<ts>\d+)\.ts$")
    assert (r.camera_id, r.start_ts) == ("camX", 1793995380.0)


# ---------- watcher ----------
@needs_ffmpeg
def test_ffprobe(tiny_video, tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a video")
    assert ffprobe_ok(tiny_video) and not ffprobe_ok(bad)


@needs_ffmpeg
def test_growing_file_not_submitted_until_stable(tmp_path, tiny_video):
    clock, sub = Clock(), FakeSubmit()
    w = make_watcher(tmp_path, sub, clock)
    dst = w.inbox / "cam01_1793995380.mp4"
    shutil.copy(tiny_video, dst)
    w.scan(); clock.t += 3
    with open(dst, "ab") as fh:                             # writer still appending
        fh.write(b"\0" * 10)
    os.utime(dst, (clock.t, clock.t))
    assert w.scan() == []
    clock.t += 3
    assert w.scan() == []                                   # only 3 s stable
    clock.t += 3
    # appended junk after moov may or may not probe; replace with clean copy to keep test about stability
    shutil.copy(tiny_video, dst); w.scan(); clock.t += 6
    assert len(w.scan()) == 1
    asyncio.run(w.submit_pending())
    assert sub.calls == [("cam01", "cam01_1793995380.mp4", 1793995380.0, "n1")]
    assert (tmp_path / "processing" / "cam01" / "cam01_1793995380.mp4").exists()
    assert not dst.exists()


@needs_ffmpeg
def test_dedup_across_restart(tmp_path, tiny_video):
    clock, sub = Clock(), FakeSubmit()
    w = make_watcher(tmp_path, sub, clock)
    shutil.copy(tiny_video, w.inbox / "cam01_1793995380.mp4")
    settle(w, clock); asyncio.run(w.submit_pending())
    w.ledger.close()

    sub2 = FakeSubmit()
    w2 = make_watcher(tmp_path, sub2, clock)                # restart: same ledger
    assert asyncio.run(w2.submit_pending()) == 0            # nothing resubmitted
    shutil.copy(tiny_video, w2.inbox / "cam01_1793995380.mp4")   # same chunk dumped again
    assert settle(w2, clock) == []
    asyncio.run(w2.submit_pending())
    assert sub2.calls == []
    assert (tmp_path / "duplicates" / "cam01_1793995380.mp4").exists()


@needs_ffmpeg
def test_restart_resubmits_admitted_but_unsubmitted(tmp_path, tiny_video):
    clock = Clock()
    w = make_watcher(tmp_path, FakeSubmit(), clock)
    shutil.copy(tiny_video, w.inbox / "cam01_1793995380.mp4")
    settle(w, clock)                                        # admitted, crash before submit
    w.ledger.close()
    sub = FakeSubmit()
    w2 = make_watcher(tmp_path, sub, clock)
    asyncio.run(w2.submit_pending())
    assert len(sub.calls) == 1


@needs_ffmpeg
def test_corrupt_and_unparsable_rejected(tmp_path):
    clock, sub = Clock(), FakeSubmit()
    w = make_watcher(tmp_path, sub, clock)
    (w.inbox / "cam01_1793995380.mp4").write_bytes(b"garbage" * 100)
    (w.inbox / "loose.mp4").write_bytes(b"garbage")         # no camera derivable
    assert settle(w, clock) == []
    assert sorted(p.name for p in (tmp_path / "rejected").iterdir()) == ["cam01_1793995380.mp4", "loose.mp4"]
    assert sub.calls == []


@needs_ffmpeg
def test_submit_failure_retry_with_backoff(tmp_path, tiny_video):
    clock, sub = Clock(), FakeSubmit(fail_times=2)
    w = make_watcher(tmp_path, sub, clock)
    shutil.copy(tiny_video, w.inbox / "cam01_1793995380.mp4")
    settle(w, clock)
    asyncio.run(w.submit_pending())
    row = w.ledger.get("cam01", 1793995380.0)
    assert row["state"] == "failed" and row["attempts"] == 1
    asyncio.run(w.submit_pending())                         # backoff not elapsed
    assert w.ledger.get("cam01", 1793995380.0)["attempts"] == 1
    clock.t += 10; asyncio.run(w.submit_pending())          # 2nd failure, backoff 20 s
    clock.t += 10; asyncio.run(w.submit_pending())
    assert sub.calls == []
    clock.t += 10; asyncio.run(w.submit_pending())
    assert w.ledger.get("cam01", 1793995380.0)["state"] == "submitted"
    assert len(sub.calls) == 1


@needs_ffmpeg
def test_exhausted_chunk_does_not_block_camera(tmp_path, tiny_video):
    clock, sub = Clock(), FakeSubmit(fail_times=1)
    w = make_watcher(tmp_path, sub, clock, max_attempts=1)
    shutil.copy(tiny_video, w.inbox / "cam01_1793995380.mp4")
    shutil.copy(tiny_video, w.inbox / "cam01_1793995560.mp4")
    settle(w, clock)
    asyncio.run(w.submit_pending())                         # first fails -> exhausted, second blocked this round
    clock.t += 1; asyncio.run(w.submit_pending())
    assert [c[2] for c in sub.calls] == [1793995560.0]


@needs_ffmpeg
def test_per_camera_ordering(tmp_path, tiny_video):
    clock, sub = Clock(), FakeSubmit()
    w = make_watcher(tmp_path, sub, clock)
    for name in ["cam02_1793995560.mp4", "cam01_1793995560.mp4", "cam01_1793995380.mp4",
                 "cam02_1793995380.mp4", "cam01_1793995740.mp4"]:
        shutil.copy(tiny_video, w.inbox / name)
    settle(w, clock)
    asyncio.run(w.submit_pending())
    for cam in ("cam01", "cam02"):
        ts = [c[2] for c in sub.calls if c[0] == cam]
        assert ts == sorted(ts) and len(ts) == (3 if cam == "cam01" else 2)


@needs_ffmpeg
def test_failure_blocks_later_chunks_same_camera_only(tmp_path, tiny_video):
    clock = Clock()

    class FailCam1First(FakeSubmit):
        async def __call__(self, cam, path, ts, node):
            if cam == "cam01" and ts == 1793995380.0 and not getattr(self, "done", False):
                self.done = True
                raise RuntimeError("boom")
            await super().__call__(cam, path, ts, node)

    sub = FailCam1First()
    w = make_watcher(tmp_path, sub, clock)
    for name in ["cam01_1793995380.mp4", "cam01_1793995560.mp4", "cam02_1793995380.mp4"]:
        shutil.copy(tiny_video, w.inbox / name)
    settle(w, clock)
    asyncio.run(w.submit_pending())
    assert [(c[0], c[2]) for c in sub.calls] == [("cam02", 1793995380.0)]
    clock.t += 10; asyncio.run(w.submit_pending())
    assert [(c[0], c[2]) for c in sub.calls][1:] == [("cam01", 1793995380.0), ("cam01", 1793995560.0)]
