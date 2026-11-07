import pytest
from vlmpipe.resize import smart_resize, visual_tokens


@pytest.mark.parametrize("h,w", [(1080, 1920), (2160, 3840), (480, 640), (720, 1280), (50, 50)])
def test_multiple_of_28_and_bounded(h, w):
    mx = 640 * 28 * 28
    rh, rw = smart_resize(h, w, max_pixels=mx)
    assert rh % 28 == 0 and rw % 28 == 0
    assert rh * rw <= mx
    assert rh * rw >= 4 * 28 * 28


def test_aspect_preserved():
    rh, rw = smart_resize(1080, 1920, max_pixels=640 * 28 * 28)
    assert abs(rw / rh - 1920 / 1080) < 0.1


def test_idempotent():
    rh, rw = smart_resize(1080, 1920, max_pixels=640 * 28 * 28)
    assert smart_resize(rh, rw, max_pixels=640 * 28 * 28) == (rh, rw)


def test_tokens():
    assert visual_tokens(560, 1008) == 20 * 36


def test_extreme_aspect():
    with pytest.raises(ValueError):
        smart_resize(10, 5000)
