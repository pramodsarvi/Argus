import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlmpipe.prompts import Registry, build_messages, frames_to_video

REG = Registry.load(Path(__file__).parent.parent / "configs" / "prompts.yaml")
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def make_frames(d: Path, n: int, h: int = 252, w: int = 448) -> list[Path]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        p = d / f"f{i}.png"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(p)
        out.append(p)
    return out


def probe(p: Path) -> dict:
    o = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,nb_read_frames", "-of", "json",
                        str(p)], check=True, capture_output=True, text=True).stdout
    return json.loads(o)["streams"][0]


@needs_ffmpeg
@pytest.mark.parametrize("n,expected", [(4, 4), (5, 6), (1, 2)])
def test_frames_to_video_even_count_and_dims(tmp_path, n, expected):
    v = frames_to_video(make_frames(tmp_path, n), tmp_path / "v", fps=1.0)
    s = probe(v)
    assert (s["width"], s["height"]) == (448, 252)
    assert s["width"] % 28 == 0 and s["height"] % 28 == 0
    assert int(s["nb_read_frames"]) == expected


@needs_ffmpeg
def test_frames_to_video_content_addressed_and_idempotent(tmp_path):
    frames = make_frames(tmp_path, 4)
    a = frames_to_video(frames, tmp_path / "v")
    mtime = a.stat().st_mtime_ns
    b = frames_to_video(frames, tmp_path / "v")
    assert a == b and b.stat().st_mtime_ns == mtime          # reused, not rebuilt
    c = frames_to_video(frames[:2], tmp_path / "v")
    assert c != a
    assert frames_to_video(frames, tmp_path / "v", fps=2.0) != a


@needs_ffmpeg
def test_video_message_structure(tmp_path):
    frames = make_frames(tmp_path, 4)
    m = build_messages(REG.system_text, frames, "q?", input_mode="video", video_dir=tmp_path / "v")
    c = m[1]["content"]
    assert [x["type"] for x in c] == ["text", "video_url", "text"]
    assert "4 frames" in c[0]["text"] and "numbered 1..4" in c[0]["text"]
    assert c[1]["video_url"]["url"].startswith("file://") and c[1]["video_url"]["url"].endswith(".mp4")
    assert c[2]["text"] == "q?"


@needs_ffmpeg
@pytest.mark.parametrize("mode", ["images", "video"])
def test_prefix_identical_across_prompts(tmp_path, mode):
    frames = make_frames(tmp_path, 3)
    kw = dict(input_mode=mode, video_dir=tmp_path / "v")
    warm = build_messages(REG.system_text, frames, None, **kw)
    n = len(warm[1]["content"])
    for p in REG.prompts:
        m = build_messages(REG.system_text, frames, p.text, **kw)
        assert m[0] == warm[0]
        assert json.dumps(m[1]["content"][:n]) == json.dumps(warm[1]["content"])
        assert m[1]["content"][n:] == [{"type": "text", "text": p.text}]


def test_unknown_input_mode():
    with pytest.raises(ValueError):
        build_messages("s", [], None, input_mode="audio")


def test_spike_arg_parsing():
    from bench.spike import build_parser
    a = build_parser().parse_args(["--video", "x.mp4", "--input-modes", "images", "video",
                                   "--modes", "warm", "two_tier", "--video-mm-fps", "1"])
    assert a.input_modes == ["images", "video"]
    assert a.modes == ["warm", "two_tier"]
    assert a.video_mm_fps == 1.0
    d = build_parser().parse_args(["--video", "x.mp4"])
    assert d.input_modes == ["images"] and d.video_mm_fps is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--video", "x.mp4", "--input-modes", "audio"])


def test_run_bundle_passes_input_mode(monkeypatch):
    import asyncio

    import vlmpipe.bundle as b
    seen = []

    async def fake_call(client, model, messages, max_tokens, pid, schema, timeout, mm=None):
        seen.append((pid, [x["type"] for x in messages[1]["content"]], mm))
        return b.Result(pid, True, {}, 0.1, 0.2, 100, 90, 5)

    monkeypatch.setattr(b, "_call", fake_call)
    monkeypatch.setattr(b, "build_messages",
                        lambda s, f, q, input_mode="images", fps=1.0: [
                            {"role": "system", "content": s},
                            {"role": "user", "content": [{"type": input_mode}]}])
    run = asyncio.run(b.run_bundle(None, "m", REG, ["a.png"], mode="warm", input_mode="video",
                                   mm_processor_kwargs={"fps": 1.0}))
    assert run["input_mode"] == "video"
    assert seen[0][0] == "_warm" and all(t == ["video"] for _, t, _ in seen)
    assert all(mm == {"fps": 1.0} for *_, mm in seen)
