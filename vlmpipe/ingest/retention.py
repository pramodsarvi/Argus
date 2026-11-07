"""Retention: keep node-local disks and the ingest ledger from growing forever.

Ages are by file mtime. Defaults are generous relative to the stale threshold (2 chunk
durations, ~6 min): a chunk older than `processing_ttl_s` can no longer be analyzed anyway
(the camera workflow would mark it stale), so its source file and frames are disposable.
Results live in Postgres/Kafka, not on these disks.
"""
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    processing_ttl_s: float = 3600.0        # source chunks after admission
    frames_ttl_s: float = 3600.0            # sampled frames / packed videos
    rejected_ttl_s: float = 7 * 86400.0     # keep for debugging
    duplicates_ttl_s: float = 86400.0
    ledger_ttl_s: float = 7 * 86400.0       # submitted rows; must exceed any resubmit window
    interval_s: float = 300.0               # how often run_forever triggers a sweep


def prune_tree(root: Path, ttl_s: float, now: float) -> int:
    """Delete files older than ttl under root, then empty directories (never root itself)."""
    if not root.exists():
        return 0
    removed = 0
    cutoff = now - ttl_s
    for dirpath, _dirs, files in os.walk(root, topdown=False):
        for f in files:
            p = Path(dirpath) / f
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        d = Path(dirpath)
        if d != root:
            try:
                d.rmdir()                     # only succeeds if empty
            except OSError:
                pass
    return removed


def prune_ledger(db, ttl_s: float, now: float) -> int:
    """Drop ledger rows for chunks submitted long ago. Rows still seen/failed are kept."""
    cur = db.execute("DELETE FROM chunks WHERE state='submitted' AND updated < ?", (now - ttl_s,))
    db.commit()
    return cur.rowcount or 0


def sweep(watcher, policy: RetentionPolicy, frames_dir: str | Path | None = None,
          now: float | None = None) -> dict:
    now = watcher.clock() if now is None else now
    stats = {
        "processing": prune_tree(watcher.processing, policy.processing_ttl_s, now),
        "rejected": prune_tree(watcher.rejected, policy.rejected_ttl_s, now),
        "duplicates": prune_tree(watcher.duplicates, policy.duplicates_ttl_s, now),
        "frames": prune_tree(Path(frames_dir), policy.frames_ttl_s, now) if frames_dir else 0,
        "ledger_rows": prune_ledger(watcher.ledger.db, policy.ledger_ttl_s, now),
    }
    if any(stats.values()):
        log.info("retention sweep: %s", stats)
    return stats


class RetentionTimer:
    """Call .maybe_sweep() from the watcher loop; sweeps at most every interval_s."""

    def __init__(self, watcher, policy: RetentionPolicy, frames_dir=None, clock=time.time):
        self.w, self.policy, self.frames_dir, self.clock = watcher, policy, frames_dir, clock
        self.last = 0.0

    def maybe_sweep(self) -> dict | None:
        now = self.clock()
        if now - self.last < self.policy.interval_s:
            return None
        self.last = now
        return sweep(self.w, self.policy, self.frames_dir, now)
