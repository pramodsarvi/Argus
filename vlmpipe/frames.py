"""Sample frames from a segment at a fixed fps, resized to the processor grid, with dedup.

Backends:
  ffmpeg  - CPU or `-hwaccel cuda` (NVDEC). Default; simplest.
  dali    - see vlmpipe/dali_decode.py (GPU box only).
At ~1 fps sampling decode is cheap; benchmark before assuming DALI is needed.
"""
import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .resize import smart_resize, visual_tokens
from .segment import probe


@dataclass
class Frame:
    path: Path
    index: int          # sample index within segment
    t: float            # seconds from segment start
    sha256: str


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


_PTS_RE = re.compile(r"\bn:\s*(\d+)\b.*?\bpts_time:\s*([-\d.]+)")


def _scene_select_expr(threshold: float, min_fps: float, max_fps: float) -> str:
    """ffmpeg select expression: always keep the first frame; afterwards keep a frame only if at
    least 1/max_fps has passed since the last kept frame AND (it is a scene change OR at least
    1/min_fps has passed). Note: a cut inside the max_fps guard window is dropped."""
    gap = "(t-prev_selected_t)"
    return (f"isnan(prev_selected_t)"
            f"+gte({gap},{1.0 / max_fps:.6f})*(gt(scene,{threshold})+gte({gap},{1.0 / min_fps:.6f}))")


def _downselect(n: int, k: int) -> list[int]:
    """k indices spread uniformly over range(n), always including first and last."""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _extract_fixed(segment, out_dir: Path, fps: float, w: int, h: int, hwaccel) -> list[tuple[Path, float]]:
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(segment), "-vf", f"fps={fps},scale={w}:{h}:flags=bicubic",
            str(out_dir / "raw_%05d.png")]
    subprocess.run(cmd, check=True)
    return [(p, i / fps) for i, p in enumerate(sorted(out_dir.glob("raw_*.png")))]


def _extract_scene(segment, out_dir: Path, w: int, h: int, hwaccel, threshold: float,
                   min_fps: float, max_fps: float) -> list[tuple[Path, float]]:
    """Single pass: select on scene score, then showinfo logs the true pts of each kept frame."""
    expr = _scene_select_expr(threshold, min_fps, max_fps)
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(segment),
            "-vf", f"select='{expr}',showinfo,scale={w}:{h}:flags=bicubic",
            "-fps_mode", "vfr", str(out_dir / "raw_%05d.png")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    times = [float(m.group(2)) for line in r.stderr.splitlines()
             if "Parsed_showinfo" in line and (m := _PTS_RE.search(line))]
    raw = sorted(out_dir.glob("raw_*.png"))
    if len(times) != len(raw):
        raise RuntimeError(f"showinfo reported {len(times)} frames but {len(raw)} were written")
    return list(zip(raw, times))


def sample_frames(segment: str | Path, out_dir: str | Path, fps: float = 1.0,
                  max_pixels: int = 640 * 28 * 28, hwaccel: str | None = None,
                  dedup_threshold: int = 4, *, strategy: str = "fixed",
                  scene_threshold: float = 0.3, min_fps: float = 0.1, max_fps: float = 2.0,
                  max_frames: int | None = None) -> tuple[list[Frame], dict]:
    """Returns kept frames and stats. dedup_threshold = max Hamming distance treated as duplicate
    (0 disables).

    strategy="fixed": uniform `fps`.
    strategy="scene": keep scene changes (score > scene_threshold), at most `max_fps`, and at
    least one frame every 1/min_fps seconds; Frame.t is the true segment-local pts.
    max_frames: if more frames were sampled, downselect uniformly (first/last kept) before dedup.
    """
    if strategy not in ("fixed", "scene"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if strategy == "scene" and not (0 < min_fps <= max_fps):
        raise ValueError("need 0 < min_fps <= max_fps")
    meta = probe(segment)
    h, w = smart_resize(meta["height"], meta["width"], max_pixels=max_pixels)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if strategy == "fixed":
        raw = _extract_fixed(segment, out_dir, fps, w, h, hwaccel)
    else:
        raw = _extract_scene(segment, out_dir, w, h, hwaccel, scene_threshold, min_fps, max_fps)
    sampled = len(raw)
    if max_frames is not None and sampled > max_frames:
        keep = set(_downselect(sampled, max_frames))
        for i, (p, _) in enumerate(raw):
            if i not in keep:
                p.unlink()
        raw = [r for i, r in enumerate(raw) if i in keep]

    kept: list[Frame] = []
    last_hash = None
    for i, (p, t) in enumerate(raw):
        img = Image.open(p)
        dh = dhash(img)
        if dedup_threshold and last_hash is not None and bin(dh ^ last_hash).count("1") <= dedup_threshold:
            p.unlink()
            continue
        last_hash = dh
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        final = out_dir / f"{digest[:16]}.png"     # content-addressed
        p.rename(final)
        kept.append(Frame(final, i, t, digest))
    stats = {"strategy": strategy, "sampled": sampled, "after_cap": len(raw), "kept": len(kept),
             "height": h, "width": w,
             "tokens_per_frame": visual_tokens(h, w),
             "prompt_visual_tokens": visual_tokens(h, w) * len(kept),
             "frame_times": [f.t for f in kept]}
    return kept, stats
