"""Split a video into keyframe-aligned segments without re-encoding."""
import json
import subprocess
from pathlib import Path


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": float(num) / float(den), "duration": float(d["format"]["duration"])}


def segment(path: str | Path, out_dir: str | Path, seconds: int = 60) -> list[Path]:
    """Stream-copy split; segment boundaries snap to keyframes so lengths are approximate."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "seg_%05d.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-c", "copy",
         "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1",
         "-segment_list", str(out_dir / SEGMENT_LIST), "-segment_list_type", "csv",
         str(pattern)],
        check=True)
    return sorted(out_dir.glob("seg_*.mp4"))


SEGMENT_LIST = "segments.csv"


def segment_times(segment_path: str | Path) -> tuple[float, float] | None:
    """(start_s, end_s) of a segment within the source video, from the list written by segment().
    Segments snap to keyframes, so index * seconds is NOT a reliable start time."""
    seg = Path(segment_path)
    lst = seg.parent / SEGMENT_LIST
    if not lst.exists():
        return None
    for line in lst.read_text().splitlines():
        name, start, end = line.rsplit(",", 2)
        if name == seg.name:
            return float(start), float(end)
    return None
