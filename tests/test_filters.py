import sys
from pathlib import Path

import numpy as np

from vlmpipe.filters import Chain, CLIPRelevanceFilter, PassThrough

# Fake 2-D embedding space: query "forklift" = [1,0]; frames given explicit vectors.
VEC = {"f0": [0, 1], "f1": [1, 0], "f2": [0, 1], "f3": [0.9, 0.1], "f4": [0, 1], "f5": [0, 1]}


def fake_embedders():
    img = lambda paths: np.array([VEC[Path(p).stem] for p in paths], dtype=float)
    txt = lambda texts: np.array([[1.0, 0.0] for _ in texts])
    return img, txt


PATHS = [Path(f"/x/{k}.png") for k in VEC]


def test_passthrough():
    kept, st = PassThrough().filter(PATHS)
    assert kept == PATHS and st.keep_ratio == 1.0


def test_clip_keeps_relevant_plus_endpoints():
    f = CLIPRelevanceFilter(["forklift"], threshold=0.5, embedders=fake_embedders())
    kept, st = f.filter(PATHS)
    assert [p.stem for p in kept] == ["f0", "f1", "f3", "f5"]
    assert st.input == 6 and st.kept == 4 and len(st.scores) == 6


def test_clip_min_keep_when_nothing_matches():
    f = CLIPRelevanceFilter(["forklift"], threshold=2.0, min_keep=3, embedders=fake_embedders())
    kept, _ = f.filter(PATHS)
    assert {p.stem for p in kept} >= {"f0", "f5", "f1"} and len(kept) == 3


def test_short_input_untouched():
    f = CLIPRelevanceFilter(["x"], embedders=fake_embedders())
    assert f.filter(PATHS[:2])[0] == PATHS[:2]


def test_chain():
    kept, stats = Chain(PassThrough(), CLIPRelevanceFilter(["q"], 0.5, embedders=fake_embedders())).filter(PATHS)
    assert len(stats) == 2 and len(kept) == 4


def test_no_heavy_imports():
    import vlmpipe.filters  # noqa: F401
    assert "torch" not in sys.modules and "open_clip" not in sys.modules
