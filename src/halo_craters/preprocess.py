"""Turn raw image pixels into model input, matching how the model was trained.

Training and the original inference notebook z-scored every channel independently:
    x' = (x - mean) / (std + 1e-8)
At inference, the mean/std come from each 256x256 tile, per channel. Because of
this, the absolute bit depth of the input (8, 11 or 16 bit) mostly cancels out.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def scene_percentiles(src, bands: list[int], nodata, pct=(2.0, 98.0), max_side: int = 2048) -> np.ndarray:
    """Per-band (lo, hi) percentiles from a decimated read of the whole scene -> (C, 2)."""
    scale = max(1, int(np.ceil(max(src.height, src.width) / max_side)))
    shape = (len(bands), max(1, src.height // scale), max(1, src.width // scale))
    a = src.read(bands, out_shape=shape).reshape(len(bands), -1)
    if nodata is not None:
        a = a[:, ~np.all(a == nodata, axis=0)]
    return np.stack([np.percentile(b, pct) for b in a]).astype(np.float32)


def stretch_to_8bit(strip: np.ndarray, lohi: np.ndarray) -> np.ndarray:
    """Linear percentile stretch of an (H, W, C) array to 0-255 (like an 8-bit display product)."""
    lo, hi = lohi[:, 0], lohi[:, 1]
    out = (strip.astype(np.float32) - lo) / np.maximum(hi - lo, EPS) * 255.0
    return np.round(np.clip(out, 0, 255))


def valid_mask(tile: np.ndarray, nodata) -> np.ndarray:
    """(H, W) bool: True where a pixel holds data. `tile` is (H, W, C)."""
    if nodata is None:
        return np.ones(tile.shape[:2], dtype=bool)
    return ~np.all(tile == nodata, axis=-1)


def normalize_tile(tile: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Per-channel z-score of an (H, W, C) tile.

    If `valid` is given, mean/std are computed over valid pixels only and invalid
    pixels are set to 0 (= the channel mean) afterwards. With valid=None this is
    exactly the original `image_normalize(im, axis=(0, 1))`.
    """
    t = tile.astype(np.float32, copy=False)
    if valid is None:
        return (t - t.mean(axis=(0, 1))) / (t.std(axis=(0, 1)) + EPS)
    if not valid.any():
        return np.zeros_like(t)
    v = t[valid]                       # (N, C)
    out = (t - v.mean(axis=0)) / (v.std(axis=0) + EPS)
    out[~valid] = 0.0
    return out
