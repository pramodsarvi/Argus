"""Scene-adaptive sampling on synthetic lavfi videos (CPU ffmpeg only)."""
import shutil
import subprocess

import pytest

from vlmpipe.frames import _downselect, sample_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")

COLORS = ["red", "black", "white", "blue", "green"]   # adjacent pairs are visually distinct
CLIP_S = 3.0


def _make(path, inputs, n):
    args = ["ffmpeg", "-v", "error", "-y"]
    for i in inputs:
        args += ["-f", "lavfi", "-i", i]
    fc = "".join(f"[{i}]" for i in range(n)) + f"concat=n={n}:v=1[v]"
    args += ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-g", "30",
             "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)
    return path


@pytest.fixture(scope="module")
def static_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("s") / "static.mp4"
    return _make(p, ["color=gray:size=320x240:rate=30:duration=20"], 1)


@pytest.fixture(scope="module")
def cuts_video(tmp_path_factory):
    p = tmp_path_factory.mktemp("c") / "cuts.mp4"
    srcs = [f"color={c}:size=320x240:rate=30:duration={CLIP_S}" for c in COLORS]
    return _make(p, srcs, len(srcs))


def test_static_falls_back_to_min_fps(static_video, tmp_path):
    frames, st = sample_frames(static_video, tmp_path, strategy="scene", min_fps=0.25,
                               max_fps=2.0, dedup_threshold=0)
    # 20 s at >= one frame per 4 s, starting at t=0 -> 5 frames (t=0,4,8,12,16)
    assert st["strategy"] == "scene"
    assert 4 <= st["sampled"] <= 6
    assert frames[0].t == pytest.approx(0.0, abs=0.05)


def test_cuts_detected_with_true_timestamps(cuts_video, tmp_path):
    frames, st = sample_frames(cuts_video, tmp_path, strategy="scene", min_fps=0.1,
                               max_fps=2.0, dedup_threshold=0)
    ts = st["frame_times"]
    assert ts == sorted(ts) and ts == [f.t for f in frames]
    cuts = [i * CLIP_S for i in range(len(COLORS))]          # includes t=0 (first frame)
    for c in cuts:
        assert any(abs(t - c) < 0.1 for t in ts), f"no frame near cut {c}: {ts}"
    # nothing but cuts: min_fps=0.1 -> no filler within 3 s clips
    assert len(ts) == len(cuts)


def test_max_fps_spacing(tmp_path):
    p = tmp_path / "flicker.mp4"
    # hard black/white cut every 0.2 s for 4 s -> scene fires ~20 times; max_fps=2 must cap spacing.
    # (Uniform noise does NOT work: ffmpeg's scene score is the change in frame difference.)
    n = 20
    _make(p, [f"color={'black' if i % 2 else 'white'}:size=320x240:rate=30:duration=0.2"
              for i in range(n)], n)
    frames, st = sample_frames(p, tmp_path / "f", strategy="scene", scene_threshold=0.01,
                               min_fps=0.1, max_fps=2.0, dedup_threshold=0)
    ts = st["frame_times"]
    assert len(ts) >= 2
    assert all(b - a >= 0.5 - 1e-3 for a, b in zip(ts, ts[1:]))
    assert len(ts) <= 4 * 2 + 1


def test_max_frames_downselect(static_video, tmp_path):
    frames, st = sample_frames(static_video, tmp_path, 2.0, dedup_threshold=0, max_frames=5)
    assert st["sampled"] == 40 and st["after_cap"] == 5 and len(frames) == 5
    assert frames[0].t == 0.0 and frames[-1].t == pytest.approx(19.5)


def test_downselect_indices():
    assert _downselect(10, 3) == [0, 4, 9] or _downselect(10, 3) == [0, 5, 9]
    assert _downselect(3, 10) == [0, 1, 2]
    assert _downselect(10, 1) == [0]


def test_fixed_unchanged_and_positional(static_video, tmp_path):
    frames, st = sample_frames(static_video, tmp_path, 1.0, 640 * 28 * 28, None, 0)
    assert st["strategy"] == "fixed" and st["sampled"] == 20
    assert [f.t for f in frames] == [float(i) for i in range(20)]


def test_bad_strategy(static_video, tmp_path):
    with pytest.raises(ValueError):
        sample_frames(static_video, tmp_path, strategy="nope")
