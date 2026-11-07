"""Parse camera_id and chunk start time from a chunk file path."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATTERN = r"(?P<camera_id>[^_/]+)_(?P<ts>\d{8}T\d{6}Z?|\d{9,})\.(?:mp4|mkv|ts)$"
VIDEO_EXTS = {".mp4", ".mkv", ".ts"}


@dataclass(frozen=True)
class ChunkRef:
    camera_id: str
    path: Path
    start_ts: float          # UTC epoch seconds
    size: int


def parse_ts(s: str) -> float:
    """'20261107T120300Z' / '20261107T120300' (treated as UTC) or epoch seconds (>= 9 digits)."""
    if s.isdigit() and len(s) >= 9:
        return float(s)
    dt = datetime.strptime(s.rstrip("Z"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_chunk(path: str | Path, pattern: str = DEFAULT_PATTERN, fallback: bool = True,
                inbox: str | Path | None = None) -> ChunkRef | None:
    """Regex on the file name first. Fallback: camera_id = parent dir name (not the inbox root
    itself), start_ts = file mtime. Returns None if nothing usable."""
    p = Path(path)
    st = p.stat()
    m = re.search(pattern, p.name)
    if m:
        try:
            return ChunkRef(m.group("camera_id"), p, parse_ts(m.group("ts")), st.st_size)
        except (ValueError, IndexError):
            pass
    if not fallback or p.suffix.lower() not in VIDEO_EXTS:
        return None
    parent = p.parent
    if inbox is not None and parent.resolve() == Path(inbox).resolve():
        return None
    return ChunkRef(parent.name, p, st.st_mtime, st.st_size)
