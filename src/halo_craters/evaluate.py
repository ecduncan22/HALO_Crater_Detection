"""Object-level comparison of predicted crater polygons against reference polygons.

Matching is one-to-one and greedy by IoU: all intersecting (pred, ref) pairs are
ranked by IoU and accepted highest-first if neither side is already matched and
IoU >= `iou_threshold`. With iou_threshold=0 any overlap counts as a match.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import geopandas as gpd


@dataclass
class MatchResult:
    n_pred: int
    n_ref: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou_matched: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def clip_to_aoi(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep objects whose centroid (representative point) lies inside the AOI."""
    aoi = aoi.to_crs(gdf.crs)
    region = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    return gdf[gdf.geometry.representative_point().within(region)].copy()


def match(pred: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, iou_threshold: float = 0.1) -> tuple[MatchResult, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    ref = ref.to_crs(pred.crs)
    pred = pred.reset_index(drop=True).copy()
    ref = ref.reset_index(drop=True).copy()
    pred["geometry"] = pred.geometry.buffer(0)
    ref["geometry"] = ref.geometry.buffer(0)
    pairs = []
    if len(pred) and len(ref):
        j = gpd.sjoin(pred[["geometry"]], ref[["geometry"]], predicate="intersects", how="inner")
        for pi, ri in zip(j.index, j["index_right"]):
            a, b = pred.geometry.iloc[pi], ref.geometry.iloc[ri]
            inter = a.intersection(b).area
            union = a.union(b).area
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold and inter > 0:
                pairs.append((iou, pi, ri))
    pairs.sort(reverse=True)
    pm, rm, ious = {}, {}, []
    for iou, pi, ri in pairs:
        if pi in pm or ri in rm:
            continue
        pm[pi], rm[ri] = ri, pi
        ious.append(iou)
    tp = len(pm)
    fp, fn = len(pred) - tp, len(ref) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pred["matched"] = pred.index.isin(list(pm))
    ref["matched"] = ref.index.isin(list(rm))
    res = MatchResult(len(pred), len(ref), tp, fp, fn, prec, rec, f1,
                      float(sum(ious) / len(ious)) if ious else 0.0)
    return res, pred, ref
