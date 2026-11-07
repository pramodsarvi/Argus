"""Compute frame sizes that the Qwen2/2.5-VL processor will NOT resize again.

Mirrors qwen_vl_utils / HF `smart_resize`: dims are multiples of `factor` (patch 14 * merge 2 = 28),
pixel count clamped to [min_pixels, max_pixels], aspect ratio preserved.
"""
import math

FACTOR = 28
DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_MAX_PIXELS = 16384 * 28 * 28


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = DEFAULT_MIN_PIXELS,
                 max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too extreme: {height}x{width}")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def visual_tokens(height: int, width: int, factor: int = FACTOR) -> int:
    """LLM tokens one image of this (already smart-resized) size costs."""
    return (height // factor) * (width // factor)
