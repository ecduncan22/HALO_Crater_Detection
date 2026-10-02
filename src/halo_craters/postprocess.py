"""Probability raster -> crater polygons.

The probability raster is processed in horizontal strips with overlap so that
large scenes never need to fit in memory. A connected crater region is kept only
by the strip whose core contains its centroid, and only if it doesn't touch the
strip's (non-image) top/bottom edge, so every crater is produced exactly once,
whole. This requires `polygon_strip_overlap` > the largest crater, in pixels.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window, transform as window_transform
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from .config import RunConfig

log = logging.getLogger(__name__)
EIGHT = np.ones((3, 3), dtype=int)


def polygonize(prob_path: str | Path, cfg: RunConfig, attrs: dict | None = None) -> gpd.GeoDataFrame:
    thr = cfg.threshold * 255.0
    records = []
    with rasterio.open(prob_path) as src:
        H, W = src.height, src.width
        px_area = abs(src.res[0] * src.res[1])
        R, ov = cfg.polygon_strip_rows, cfg.polygon_strip_overlap
        for s in range(0, H, R):
            core0, core1 = s, min(s + R, H)
            e0, e1 = max(0, core0 - ov), min(H, core1 + ov)
            prob = src.read(1, window=Window(0, e0, W, e1 - e0))
            binary = prob >= thr
            if not binary.any():
                continue
            labels, n = ndimage.label(binary, structure=EIGHT)
            idx = np.arange(1, n + 1)
            slices = ndimage.find_objects(labels)
            centroids = ndimage.center_of_mass(binary, labels, idx)
            mean_p = ndimage.mean(prob, labels, idx) / 255.0
            max_p = ndimage.maximum(prob, labels, idx) / 255.0
            n_pix = ndimage.sum(binary, labels, idx)

            keep = np.zeros(n + 1, dtype=bool)
            for i, (sl, (cy, _)) in enumerate(zip(slices, centroids), start=1):
                gy = e0 + cy
                if not (core0 <= gy < core1):
                    continue
                if sl[0].start == 0 and e0 > 0:            # cut by strip top
                    continue
                if sl[0].stop == e1 - e0 and e1 < H:        # cut by strip bottom
                    continue
                keep[i] = True
            if not keep.any():
                continue

            tr = window_transform(Window(0, e0, W, e1 - e0), src.transform)
            masked = np.where(keep[labels], labels, 0).astype(np.int32)
            parts: dict[int, list] = {}
            for geom, val in shapes(masked, mask=masked > 0, transform=tr, connectivity=8):
                parts.setdefault(int(val), []).append(shape(geom))
            for lab, geoms in parts.items():
                g = geoms[0] if len(geoms) == 1 else unary_union(geoms)
                if not g.is_valid:
                    g = make_valid(g)
                records.append(dict(
                    geometry=g,
                    n_pix=int(n_pix[lab - 1]),
                    mean_prob=round(float(mean_p[lab - 1]), 4),
                    max_prob=round(float(max_p[lab - 1]), 4),
                ))
        crs = src.crs

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    if gdf.empty:
        gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
        return gdf
    gdf["area_m2"] = gdf.geometry.area.round(2)
    gdf["diam_m"] = (2 * np.sqrt(gdf["area_m2"] / np.pi)).round(2)
    if cfg.min_area_m2 > 0:
        gdf = gdf[gdf["area_m2"] >= cfg.min_area_m2]
    gdf = gdf.reset_index(drop=True)
    gdf.insert(0, "crater_id", np.arange(1, len(gdf) + 1))
    for k, v in (attrs or {}).items():
        gdf[k] = v
    log.info("Polygonized %d craters (px area %.3f m2)", len(gdf), px_area)
    return gdf


def write_vectors(gdf: gpd.GeoDataFrame, out_stem: str | Path, formats=("gpkg", "shp")) -> list[Path]:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "gpkg":
            p = out_stem.with_suffix(".gpkg")
            gdf.to_file(p, layer="craters", driver="GPKG")
        elif fmt == "shp":
            p = out_stem.with_suffix(".shp")
            gdf.to_file(p, driver="ESRI Shapefile")
        elif fmt == "geojson":
            p = out_stem.with_suffix(".geojson")
            gdf.to_file(p, driver="GeoJSON")
        else:
            raise ValueError(f"Unknown vector format {fmt}")
        written.append(p)
    return written
