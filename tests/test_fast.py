import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_twotier import REG, FakeClient
from vlmpipe.fast import process_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _video(p: Path) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=10:duration=6",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_process_video_samples_n_frames_runs_all_prompts_and_cleans_up(tmp_path):
    answers = {("safety", True): {"violations": []}, ("ocr", True): {"texts": []},
               ("actions", True): {"actions": []}, ("anomaly", True): {"anomaly": False, "description": ""},
               ("summary", True): {"summary": "s"}}
    c = FakeClient(answers)
    wd = tmp_path / "w"
    out = asyncio.run(process_video(c, "m", _video(tmp_path / "v.mp4"), REG, wd, frames=4, tokens=64))
    assert out["frames"]["n"] == 4 and out["timing"]["ok"] == out["timing"]["n"] == 5
    assert out["timing"]["warm"] is False and len(c.calls) == 5       # small prefix: no warm request
    assert out["results"]["summary"] == {"summary": "s"}
    assert list(wd.iterdir()) == []                                   # frames removed
    with pytest.raises(ValueError):
        asyncio.run(process_video(c, "m", tmp_path / "v.mp4", REG, wd, only=["nope"]))
