"""Local integration test using a synthetic ffmpeg video (no GPU / vLLM needed)."""
import shutil
import subprocess

import pytest

from vlmpipe.frames import sample_frames
from vlmpipe.segment import probe, segment

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "test.mp4"
    # 10 s: 5 s moving test pattern + 5 s static black (dedup should collapse the static part)
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=5",
                    "-f", "lavfi", "-i", "color=black:size=1280x720:rate=30:duration=5",
                    "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]",
                    "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_probe(video):
    m = probe(video)
    assert (m["width"], m["height"]) == (1280, 720)
    assert 9.5 < m["duration"] < 10.5


def test_segment(video, tmp_path):
    segs = segment(video, tmp_path, seconds=4)
    assert len(segs) >= 2


def test_sample_and_dedup(video, tmp_path):
    frames, st = sample_frames(video, tmp_path, fps=1.0, max_pixels=640 * 28 * 28)
    assert st["height"] % 28 == 0 and st["width"] % 28 == 0
    assert st["sampled"] >= 9
    assert st["kept"] < st["sampled"]          # static black tail collapsed
    assert len({f.sha256 for f in frames}) == len(frames)
    assert all(f.path.exists() for f in frames)


def test_segment_times(video, tmp_path):
    from vlmpipe.segment import segment_times
    segs = segment(video, tmp_path, seconds=4)
    times = [segment_times(s) for s in segs]
    assert all(t is not None for t in times)
    assert times[0][0] == 0.0
    for (s0, e0), (s1, _) in zip(times, times[1:]):
        assert abs(e0 - s1) < 1e-3 and s1 > s0          # contiguous, increasing


def test_frame_times_in_stats(video, tmp_path):
    frames, st = sample_frames(video, tmp_path, fps=1.0, max_pixels=640 * 28 * 28)
    assert st["frame_times"] == [f.t for f in frames]
