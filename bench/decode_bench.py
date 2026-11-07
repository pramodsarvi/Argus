"""Decode benchmark: ffmpeg CPU vs ffmpeg -hwaccel cuda vs DALI. Unavailable backends are skipped.

  python -m bench.decode_bench a.mp4 b.mp4 --fps 1 --max-pixels 501760
Metric: seconds of decode per minute of video (lower is better) and sampled frames/s.
"""
import argparse
import shutil
import subprocess
import time

from vlmpipe.resize import smart_resize
from vlmpipe.segment import probe


def ffmpeg_has_cuda() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
    return "cuda" in out


def dali_available() -> bool:
    try:
        import nvidia.dali  # noqa: F401
        return True
    except ImportError:
        return False


def run_ffmpeg(path, fps, h, w, hwaccel=None) -> float:
    cmd = ["ffmpeg", "-v", "error", "-y"] + (["-hwaccel", hwaccel] if hwaccel else []) + \
          ["-i", str(path), "-vf", f"fps={fps},scale={w}:{h}", "-f", "null", "-"]
    t = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t


def run_dali(path, fps, h, w, src_fps) -> float:
    from vlmpipe.dali_decode import dali_sample
    t = time.perf_counter()
    for _ in dali_sample([str(path)], h, w, stride=max(1, round(src_fps / fps))):
        pass
    return time.perf_counter() - t


def bench(files, fps=1.0, max_pixels=640 * 28 * 28, backends=("cpu", "cuda", "dali")) -> list[dict]:
    avail = {"cpu": bool(shutil.which("ffmpeg")), "cuda": ffmpeg_has_cuda(), "dali": dali_available()}
    rows = []
    for f in files:
        m = probe(f)
        h, w = smart_resize(m["height"], m["width"], max_pixels=max_pixels)
        frames = m["duration"] * fps
        for b in backends:
            row = {"file": str(f), "backend": b, "src": f"{m['width']}x{m['height']}", "duration_s": m["duration"]}
            if not avail.get(b):
                rows.append({**row, "status": "skipped"})
                continue
            try:
                s = run_dali(f, fps, h, w, m["fps"]) if b == "dali" else \
                    run_ffmpeg(f, fps, h, w, "cuda" if b == "cuda" else None)
                rows.append({**row, "status": "ok", "seconds": s,
                             "s_per_video_min": s / (m["duration"] / 60), "sampled_fps": frames / s})
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "status": f"error: {e}"})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda", "dali"])
    a = ap.parse_args(argv)
    print("| file | src | backend | status | s / video-min | sampled frames/s |\n|---|---|---|---|---|---|")
    for r in bench(a.files, a.fps, a.max_pixels, a.backends):
        ok = r["status"] == "ok"
        print(f"| {r['file']} | {r['src']} | {r['backend']} | {r['status']} | "
              f"{r['s_per_video_min']:.2f} | {r['sampled_fps']:.1f} |" if ok else
              f"| {r['file']} | {r['src']} | {r['backend']} | {r['status']} | - | - |")


if __name__ == "__main__":
    main()
