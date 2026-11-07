import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from bench import analyze_spike, capacity, decode_bench  # noqa: E402


def _res(pid, ttft, pt, ct, comp, ok=True):
    return {"prompt_id": pid, "ok": ok, "ttft_s": ttft, "total_s": ttft + 1, "prompt_tokens": pt,
            "cached_tokens": ct, "completion_tokens": comp}


@pytest.fixture
def spike(tmp_path):
    recs = [
        {"model": "m", "mode": "concurrent", "wall_s": 4.0, "warm": None,
         "results": [_res("a", 2.0, 1000, 0, 100), _res("b", 2.2, 1000, 0, 100, ok=False)]},
        {"model": "m", "mode": "warm", "wall_s": 3.0, "warm": _res("_warm", 1.0, 1000, 0, 1),
         "results": [_res("a", 0.3, 1000, 900, 100), _res("b", 0.4, 1000, 900, 100)]},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def test_analyze(spike, capsys):
    agg = analyze_spike.aggregate(analyze_spike.load(spike))
    assert agg[("m", "concurrent")]["cache_hit"] == 0.0
    assert agg[("m", "concurrent")]["ok_rate"] == 0.5
    assert agg[("m", "warm")]["cache_hit"] == pytest.approx(0.9)
    assert agg[("m", "warm")]["median_ttft_s"] == pytest.approx(0.35)
    analyze_spike.main([str(spike)])
    assert "| m | warm |" in capsys.readouterr().out


def test_capacity(spike):
    s = capacity.per_segment(analyze_spike.load(spike), "warm")
    # warm: (1000+1) + (100+100) + (100+100) = 1401 tokens over 3 s
    assert s["effective_tokens"] == 1401
    assert s["tokens_per_s"] == pytest.approx(467.0)
    g = capacity.gpus_needed(10, 60, 1401, 467.0, 0.3)
    assert g["segments_per_hour"] == 600
    assert g["gpus_raw"] == pytest.approx(600 * 3.0 / 3600)
    assert g["gpus"] == 1
    assert capacity.main([str(spike), "--video-hours-per-hour", "100", "--tokens-per-s", "467"])["gpus"] == 7
    with pytest.raises(ValueError):
        capacity.per_segment(analyze_spike.load(spike), "merged")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_decode_bench_cpu(tmp_path):
    v = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30:duration=3", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    rows = decode_bench.bench([v], backends=("cpu", "dali"))
    cpu = [r for r in rows if r["backend"] == "cpu"][0]
    assert cpu["status"] == "ok" and cpu["s_per_video_min"] > 0
    dali = [r for r in rows if r["backend"] == "dali"][0]
    assert dali["status"] in ("skipped", "ok") or dali["status"].startswith("error")
