"""Polling inbox watcher.

Lifecycle of a file:
  inbox/...            -> waits until size+mtime unchanged for `stable_s` and ffprobe succeeds
  processing/<cam>/    -> atomic rename (same filesystem); ledger row state=seen
  submit()             -> state=submitted, or state=failed with exponential backoff
  rejected/            -> unparsable name or corrupt file
  duplicates/          -> (camera_id, start_ts) already in ledger

Per camera, chunks are submitted strictly in start_ts order: a failed earlier chunk blocks later
ones for that camera until it succeeds or exhausts max_attempts (then it is skipped).
Polling only (no extra deps); a filesystem-event library could replace scan() later.
"""
import asyncio
import logging
import os
import socket
import sqlite3
import subprocess
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from .naming import DEFAULT_PATTERN, VIDEO_EXTS, parse_chunk

log = logging.getLogger(__name__)

SubmitFn = Callable[[str, str, float, str], Awaitable[None]]   # (camera_id, path, start_ts, node)


def ffprobe_ok(path: str | Path, timeout: float = 30.0) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False
    try:
        return r.returncode == 0 and float(r.stdout.strip().splitlines()[0]) > 0
    except (ValueError, IndexError):
        return False


class Ledger:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("""CREATE TABLE IF NOT EXISTS chunks (
            camera_id TEXT NOT NULL, start_ts REAL NOT NULL, path TEXT NOT NULL,
            state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            next_retry REAL NOT NULL DEFAULT 0, error TEXT, updated REAL NOT NULL,
            PRIMARY KEY (camera_id, start_ts))""")
        self.db.commit()

    def exists(self, camera_id: str, start_ts: float) -> bool:
        return self.db.execute("SELECT 1 FROM chunks WHERE camera_id=? AND start_ts=?",
                               (camera_id, start_ts)).fetchone() is not None

    def add(self, camera_id: str, start_ts: float, path: str, now: float) -> None:
        self.db.execute("INSERT INTO chunks(camera_id,start_ts,path,state,updated) VALUES(?,?,?,?,?)",
                        (camera_id, start_ts, path, "seen", now))
        self.db.commit()

    def mark(self, camera_id: str, start_ts: float, state: str, now: float,
             error: str | None = None, next_retry: float = 0.0, inc: bool = False) -> None:
        self.db.execute("""UPDATE chunks SET state=?, error=?, next_retry=?, updated=?,
                           attempts = attempts + ? WHERE camera_id=? AND start_ts=?""",
                        (state, error, next_retry, now, 1 if inc else 0, camera_id, start_ts))
        self.db.commit()

    def pending(self) -> list[tuple]:
        """(camera_id, start_ts, path, state, attempts, next_retry) not yet submitted, ordered."""
        return self.db.execute("""SELECT camera_id,start_ts,path,state,attempts,next_retry FROM chunks
                                  WHERE state IN ('seen','failed') ORDER BY camera_id, start_ts""").fetchall()

    def get(self, camera_id: str, start_ts: float) -> dict | None:
        r = self.db.execute("SELECT state,attempts,error,path FROM chunks WHERE camera_id=? AND start_ts=?",
                            (camera_id, start_ts)).fetchone()
        return dict(zip(("state", "attempts", "error", "path"), r)) if r else None

    def close(self) -> None:
        self.db.close()


def _unique(dst: Path) -> Path:
    if not dst.exists():
        return dst
    i = 1
    while (c := dst.with_name(f"{dst.stem}.{i}{dst.suffix}")).exists():
        i += 1
    return c


class Watcher:
    def __init__(self, inbox: str | Path, submit: SubmitFn, work_root: str | Path | None = None,
                 stable_s: float = 10.0, pattern: str = DEFAULT_PATTERN, fallback: bool = True,
                 max_attempts: int = 8, backoff_base_s: float = 5.0, backoff_max_s: float = 300.0,
                 node: str | None = None, probe: Callable[[Path], bool] = ffprobe_ok,
                 clock: Callable[[], float] = time.time):
        self.inbox = Path(inbox)
        root = Path(work_root) if work_root else self.inbox.parent
        self.processing, self.rejected, self.duplicates = (root / "processing", root / "rejected",
                                                           root / "duplicates")
        for d in (self.inbox, self.processing, self.rejected, self.duplicates):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(root / "ingest_ledger.sqlite")
        self.submit = submit
        self.stable_s, self.pattern, self.fallback = stable_s, pattern, fallback
        self.max_attempts, self.backoff_base_s, self.backoff_max_s = max_attempts, backoff_base_s, backoff_max_s
        self.node = node or socket.gethostname()
        self.probe, self.clock = probe, clock
        self._obs: dict[Path, tuple[int, float, float]] = {}   # path -> (size, mtime, since)

    # ---------- discovery ----------
    def _candidates(self) -> list[Path]:
        return sorted(p for p in self.inbox.rglob("*")
                      if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_EXTS)

    def _move(self, src: Path, dst_dir: Path) -> Path:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = _unique(dst_dir / src.name)
        os.rename(src, dst)          # atomic on the same filesystem
        return dst

    def scan(self) -> list[Path]:
        """Admit stable files into processing/ + ledger. Returns admitted paths."""
        now = self.clock()
        seen_now: set[Path] = set()
        admitted: list[Path] = []
        for p in self._candidates():
            seen_now.add(p)
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            prev = self._obs.get(p)
            if prev is None or (prev[0], prev[1]) != (st.st_size, st.st_mtime):
                self._obs[p] = (st.st_size, st.st_mtime, now)
                continue
            if now - prev[2] < self.stable_s or st.st_size == 0:
                continue
            self._obs.pop(p, None)
            ref = parse_chunk(p, self.pattern, self.fallback, self.inbox)
            if ref is None:
                log.warning("rejected (unparsable name): %s", p)
                self._move(p, self.rejected)
                continue
            if not self.probe(p):
                log.warning("rejected (ffprobe failed): %s", p)
                self._move(p, self.rejected)
                continue
            if self.ledger.exists(ref.camera_id, ref.start_ts):
                log.info("duplicate chunk %s @ %s: %s", ref.camera_id, ref.start_ts, p)
                self._move(p, self.duplicates)
                continue
            dst = self._move(p, self.processing / ref.camera_id)
            self.ledger.add(ref.camera_id, ref.start_ts, str(dst), now)
            admitted.append(dst)
        for gone in set(self._obs) - seen_now:      # files removed before becoming stable
            self._obs.pop(gone, None)
        return admitted

    # ---------- submission ----------
    async def submit_pending(self) -> int:
        now = self.clock()
        blocked: set[str] = set()
        n = 0
        for cam, ts, path, state, attempts, next_retry in self.ledger.pending():
            if cam in blocked:
                continue
            if attempts >= self.max_attempts:
                continue                              # exhausted: skip, do not block the camera
            if next_retry > now:
                blocked.add(cam)                      # keep per-camera order
                continue
            try:
                await self.submit(cam, path, ts, self.node)
            except Exception as e:  # noqa: BLE001 - recorded in ledger
                delay = min(self.backoff_max_s, self.backoff_base_s * 2 ** attempts)
                self.ledger.mark(cam, ts, "failed", now, repr(e)[:500], now + delay, inc=True)
                log.warning("submit failed %s @ %s (attempt %d): %r", cam, ts, attempts + 1, e)
                blocked.add(cam)
                continue
            self.ledger.mark(cam, ts, "submitted", now, inc=True)
            n += 1
        return n

    async def step(self) -> int:
        self.scan()
        return await self.submit_pending()

    async def run_forever(self, poll_s: float = 2.0) -> None:
        while True:
            try:
                await self.step()
            except Exception:  # noqa: BLE001 - keep the watcher alive
                log.exception("watcher step failed")
            await asyncio.sleep(poll_s)
