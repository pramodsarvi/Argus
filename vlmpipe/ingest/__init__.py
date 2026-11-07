"""Folder-watcher ingest: camera chunks dumped into an inbox -> ledger -> submit to orchestration."""
from .naming import ChunkRef, parse_chunk
from .watcher import Ledger, Watcher, ffprobe_ok

__all__ = ["ChunkRef", "parse_chunk", "Ledger", "Watcher", "ffprobe_ok"]
