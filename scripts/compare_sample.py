"""Compare runs in an outputs dir against a validation sample's original detections and GT.

Original detections are restricted to those made from the same acquisition (Source_Im
date prefix, e.g. 22JUN13) as each run's input image, since a sample's
_detected_craters.shp is the union of detections from several overlapping images.

usage: python scripts/compare_sample.py SAMPLE_DIR OUTPUTS_DIR [--iou 0.1] [--csv out.csv]
"""
import argparse, glob, json, os
import geopandas as gpd, numpy as np, pandas as pd
from halo_craters.evaluate import clip_to_aoi, match

ap = argparse.ArgumentParser(); ap.add_argument("sample"); ap.add_argument("outputs")
ap.add_argument("--iou", type=float, default=0.1); ap.add_argument("--csv")
a = ap.parse_args()
sid = os.path.basename(a.sample.rstrip("/"))
s = os.path.join(a.sample, sid)
aoi = gpd.read_file(s + ".shp")
det = clip_to_aoi(gpd.read_file(s + "_detected_craters.shp").to_crs(32637), aoi)
gt = clip_to_aoi(gpd.read_file(s + "_marked_craters.shp").to_crs(32637), aoi)
det["acq"] = det["Source_Im"].str[:7]
print(f"{sid}: original detections in AOI={len(det)} by acquisition {det.acq.value_counts().to_dict()}, GT={len(gt)}")
rows = []
for acq, d in det.groupby("acq"):
    r, _, _ = match(d, gt, a.iou)
    rows.append(dict(run=f"ORIGINAL {acq}", n=len(d), med_area=np.median(d.area), P_orig=np.nan, R_orig=np.nan,
                     P_gt=r.precision, R_gt=r.recall, F1_gt=r.f1))
for f in sorted(glob.glob(os.path.join(a.outputs, "*", "*.gpkg"))):
    meta = json.load(open(os.path.join(os.path.dirname(f), "run.json")))
    acq = os.path.basename(meta["image"])[:7]
    p = clip_to_aoi(gpd.read_file(f), aoi)
    d = det[det.acq == acq]
    r1, _, _ = match(p, d, a.iou); r2, _, _ = match(p, gt, a.iou)
    rows.append(dict(run=os.path.basename(os.path.dirname(f)).split("__")[-1], n=len(p),
                     med_area=np.median(p.area) if len(p) else 0, P_orig=r1.precision, R_orig=r1.recall,
                     P_gt=r2.precision, R_gt=r2.recall, F1_gt=r2.f1))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
if a.csv:
    df.to_csv(a.csv, index=False)
