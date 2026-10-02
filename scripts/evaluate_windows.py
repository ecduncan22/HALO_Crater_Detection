"""Evaluate runs on image windows that each have an AOI and GT polygons.

Expects, for every window NAME in DATA_DIR:  NAME.tif, NAME_aoi.gpkg, NAME_gt.gpkg
and run outputs in OUT_DIR/NAME__TAG/NAME__TAG_craters.gpkg (as written by `halo-craters run --tag TAG`).

Reports per window and pooled (per tag): n, precision, recall, F1, plus recall by GT crater
diameter bin and the effect of a minimum-area filter.

usage: python scripts/evaluate_windows.py DATA_DIR OUT_DIR [--iou 0.1] [--csv results.csv]
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from halo_craters.evaluate import clip_to_aoi, match

ap = argparse.ArgumentParser()
ap.add_argument("data_dir")
ap.add_argument("out_dir")
ap.add_argument("--iou", type=float, default=0.1)
ap.add_argument("--min-areas", default="0,10,20,30", help="min-area filters (m2) to test")
ap.add_argument("--csv")
a = ap.parse_args()
min_areas = [float(x) for x in a.min_areas.split(",")]
DBINS = [0, 5, 8, 12, 16, 1000]

rows, size_rows = [], []
for f in sorted(glob.glob(os.path.join(a.out_dir, "*__*", "*_craters.gpkg"))):
    run_dir = os.path.basename(os.path.dirname(f))
    name, tag = run_dir.split("__", 1)
    aoi_p = os.path.join(a.data_dir, f"{name}_aoi.gpkg")
    gt_p = os.path.join(a.data_dir, f"{name}_gt.gpkg")
    if not (os.path.exists(aoi_p) and os.path.exists(gt_p)):
        continue
    aoi = gpd.read_file(aoi_p)
    pred_all = clip_to_aoi(gpd.read_file(f), aoi)
    gt = clip_to_aoi(gpd.read_file(gt_p).to_crs(pred_all.crs), aoi)
    for ma in min_areas:
        pred = pred_all[pred_all.area >= ma]
        r, _, gm = match(pred, gt, a.iou)
        rows.append(dict(tag=tag, window=name, min_area=ma, n_pred=r.n_pred, n_gt=r.n_ref, tp=r.tp,
                         fp=r.fp, fn=r.fn, P=r.precision, R=r.recall, F1=r.f1, mIoU=r.mean_iou_matched))
        if ma == 0:
            d = 2 * np.sqrt(gm.area / np.pi)
            gm["dbin"] = pd.cut(d, DBINS)
            for b, grp in gm.groupby("dbin", observed=True):
                size_rows.append(dict(tag=tag, window=name, dbin=str(b), n=len(grp), hit=int(grp.matched.sum())))

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 500)
print(f"\n=== Per window (min_area=0, IoU>={a.iou}) ===")
print(df[df.min_area == 0].drop(columns="min_area").round(2).to_string(index=False))

pooled = df.groupby(["tag", "min_area"])[["n_pred", "n_gt", "tp", "fp", "fn"]].sum().reset_index()
pooled["P"] = pooled.tp / (pooled.tp + pooled.fp).clip(lower=1)
pooled["R"] = pooled.tp / (pooled.tp + pooled.fn).clip(lower=1)
pooled["F1"] = 2 * pooled.P * pooled.R / (pooled.P + pooled.R).replace(0, np.nan)
pooled["windows"] = df.groupby(["tag", "min_area"]).window.nunique().values
print("\n=== Pooled over windows ===")
print(pooled.round(3).to_string(index=False))

if size_rows:
    sr = pd.DataFrame(size_rows).groupby(["tag", "dbin"], sort=False)[["n", "hit"]].sum().reset_index()
    sr["recall"] = (sr.hit / sr.n).round(2)
    print("\n=== Recall by GT crater diameter (m), pooled ===")
    print(sr.pivot(index="tag", columns="dbin", values="recall").to_string())
if a.csv:
    df.to_csv(a.csv, index=False)
