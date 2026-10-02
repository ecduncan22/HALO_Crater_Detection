"""Full-scene tiled inference, streamed row-strip by row-strip.

Memory use is bounded by one strip of tiles (tile_size rows x image width), so
scenes of any size can be processed. Output is a single-band uint8 probability
GeoTIFF (0-255 = probability 0-1), written as a Cloud-Optimized GeoTIFF.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm

from .config import RunConfig
from .model import predict_batch
from .preprocess import normalize_tile, scene_percentiles, stretch_to_8bit, valid_mask
from .tiling import keep_ranges, tile_offsets

log = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    probability_path: Path
    n_tiles: int
    n_tiles_skipped: int
    seconds: float
    gsd_m: float


@contextlib.contextmanager
def open_input(path: str | Path, target_gsd_m: float | None = None):
    """Open a raster, optionally resampled (bilinear) to `target_gsd_m` on the fly."""
    with rasterio.open(path) as src:
        if not target_gsd_m or math.isclose(abs(src.res[0]), target_gsd_m, rel_tol=1e-3):
            yield src
            return
        scale = abs(src.res[0]) / target_gsd_m
        width, height = round(src.width * scale), round(src.height * scale)
        transform = src.transform * src.transform.scale(src.width / width, src.height / height)
        with WarpedVRT(src, crs=src.crs, transform=transform, width=width, height=height,
                       resampling=Resampling.bilinear) as vrt:
            yield vrt


def run_inference(
    image_path: str | Path,
    out_path: str | Path,
    model,
    bands: list[int],
    cfg: RunConfig,
    nodata=None,
    progress: bool = True,
) -> InferenceResult:
    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.tif")
    T = cfg.tile_size

    with open_input(image_path, cfg.target_gsd_m) as src:
        if nodata is None:
            nodata = src.nodata
        H, W = src.height, src.width
        rows = tile_offsets(H, T, cfg.stride, cfg.edge_mode)
        cols = tile_offsets(W, T, cfg.stride, cfg.edge_mode)
        row_keep = keep_ranges(rows, H, T)
        col_keep = keep_ranges(cols, W, T)
        n_total = len(rows) * len(cols)
        lohi = scene_percentiles(src, bands, nodata) if cfg.rescale == "percentile" else None
        if lohi is not None:
            log.info("Percentile stretch lo/hi per band: %s", lohi.tolist())
        log.info("%s: %dx%d px, %d bands used %s, %d tiles (stride %d)",
                 Path(image_path).name, W, H, src.count, bands, n_total, cfg.stride)

        profile = dict(driver="GTiff", dtype="uint8", count=1, width=W, height=H,
                       crs=src.crs, transform=src.transform, nodata=None)
        skipped = 0
        pbar = tqdm(total=n_total, disable=not progress, unit="tile", desc=Path(image_path).stem[:30])
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for r, (k0, k1) in zip(rows, row_keep):
                h = min(T, H - r)
                strip = src.read(bands, window=Window(0, r, W, h))      # (C, h, W)
                strip = np.moveaxis(strip, 0, -1)                       # (h, W, C)
                out_strip = np.zeros((k1 - k0, W), dtype=np.uint8)

                batch, places = [], []
                for c, (c0, c1) in zip(cols, col_keep):
                    tile = strip[:, c:c + T]
                    valid = valid_mask(tile, nodata)      # all True if nodata is None
                    if not valid.any():                   # tile entirely outside the scene
                        skipped += 1
                        pbar.update(1)
                        continue
                    if lohi is not None:
                        tile = stretch_to_8bit(tile, lohi)
                    x = normalize_tile(tile, valid if cfg.mask_nodata else None)
                    if x.shape[0] < T or x.shape[1] < T:               # zero-pad edge tiles
                        padded = np.zeros((T, T, x.shape[2]), dtype=np.float32)
                        padded[: x.shape[0], : x.shape[1]] = x
                        x = padded
                    batch.append(x)
                    places.append((c, c0, c1, valid))
                    if len(batch) == cfg.batch_size:
                        _flush(model, batch, places, r, k0, k1, out_strip, cfg)
                        pbar.update(len(batch))
                        batch, places = [], []
                if batch:
                    _flush(model, batch, places, r, k0, k1, out_strip, cfg)
                    pbar.update(len(batch))
                dst.write(out_strip, 1, window=Window(0, k0, W, k1 - k0))
        pbar.close()
        gsd = abs(src.res[0])

    # Re-write as a compressed Cloud-Optimized GeoTIFF (fast to view, small on disk).
    rio_copy(tmp_path, out_path, driver="COG", compress="DEFLATE", overview_resampling="average")
    os.remove(tmp_path)
    secs = time.time() - t0
    log.info("Inference done in %.1f s (%d tiles, %d empty skipped)", secs, n_total, skipped)
    return InferenceResult(out_path, n_total, skipped, secs, gsd)


def _flush(model, batch, places, r, k0, k1, out_strip, cfg):
    probs = predict_batch(model, batch)
    for p, (c, c0, c1, valid) in zip(probs, places):
        if cfg.mask_nodata:
            p = np.where(valid, p[: valid.shape[0], : valid.shape[1]], 0.0)
        # rows [k0,k1) and cols [c0,c1) of the image, in tile coordinates:
        sub = p[k0 - r: k1 - r, c0 - c: c1 - c]
        out_strip[:, c0:c1] = np.round(sub * 255).astype(np.uint8)
