import os
import time

from vlmpipe.ingest.retention import RetentionPolicy, RetentionTimer, prune_tree, sweep
from vlmpipe.ingest.watcher import Watcher


async def _noop(*a):
    return None


def _touch(p, age_s, now):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    os.utime(p, (now - age_s, now - age_s))


def test_prune_tree_removes_old_files_and_empty_dirs(tmp_path):
    now = time.time()
    _touch(tmp_path / "camA" / "old.mp4", 7200, now)
    _touch(tmp_path / "camB" / "new.mp4", 10, now)
    assert prune_tree(tmp_path, 3600, now) == 1
    assert not (tmp_path / "camA").exists()
    assert (tmp_path / "camB" / "new.mp4").exists()
    assert tmp_path.exists()


def test_sweep_prunes_dirs_frames_and_submitted_ledger_rows(tmp_path):
    now = time.time()
    w = Watcher(tmp_path / "inbox", _noop, probe=lambda p: True, clock=lambda: now)
    _touch(w.processing / "cam" / "old.mp4", 7200, now)
    _touch(w.rejected / "bad.mp4", 8 * 86400, now)
    frames = tmp_path / "frames"
    _touch(frames / "cam" / "123" / "f.png", 7200, now)
    w.ledger.add("cam", 1.0, "x", now - 8 * 86400)
    w.ledger.mark("cam", 1.0, "submitted", now - 8 * 86400)
    w.ledger.add("cam", 2.0, "y", now - 8 * 86400)          # still 'seen': must be kept
    st = sweep(w, RetentionPolicy(), frames_dir=frames, now=now)
    assert st == {"processing": 1, "rejected": 1, "duplicates": 0, "frames": 1, "ledger_rows": 1}
    assert w.ledger.get("cam", 2.0) is not None and w.ledger.get("cam", 1.0) is None


def test_timer_rate_limits(tmp_path):
    t = [1000.0]
    w = Watcher(tmp_path / "inbox", _noop, probe=lambda p: True, clock=lambda: t[0])
    rt = RetentionTimer(w, RetentionPolicy(interval_s=300), clock=lambda: t[0])
    assert rt.maybe_sweep() is not None
    t[0] += 100
    assert rt.maybe_sweep() is None
    t[0] += 300
    assert rt.maybe_sweep() is not None
