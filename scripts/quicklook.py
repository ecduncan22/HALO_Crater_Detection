"""Render a window with GT (cyan outlines) and predictions (magenta outlines; filled = matched).

usage: python scripts/quicklook.py IMAGE.tif PRED.gpkg GT.gpkg OUT.jpg --sensor vantor [--size 1000] [--crop x0,y0,w,h]

Displays true colour (red, green, blue from the sensor's band map) with a per-band
1-99% stretch, which removes the blue cast of raw 16-bit data.
"""
import argparse
import geopandas as gpd, numpy as np, rasterio
from PIL import Image, ImageDraw
from halo_craters.evaluate import match

ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("pred"); ap.add_argument("gt"); ap.add_argument("out")
ap.add_argument("--sensor", default="vantor", help="sensor in configs/sensors.yaml (sets display band order)")
ap.add_argument("--bands", help="override display bands (1-based R,G,B), e.g. 3,2,1"); ap.add_argument("--size", type=int, default=1000); ap.add_argument("--crop")
ap.add_argument("--iou", type=float, default=0.1)
a = ap.parse_args()
from halo_craters.config import load_sensor
if a.bands:
    bands = [int(b) for b in a.bands.split(",")]
else:
    sb = load_sensor(a.sensor).bands
    bands = [sb["red"], sb["green"], sb["blue"]]
with rasterio.open(a.image) as r:
    win = None
    if a.crop:
        x0, y0, w, h = [int(v) for v in a.crop.split(",")]
        win = ((y0, y0 + h), (x0, x0 + w))
    img = r.read(bands, window=win).astype(float)
    tr = r.window_transform(win) if win else r.transform
    crs = r.crs
v = img[:, img.sum(0) > 0]
lo, hi = np.percentile(v, (1, 99), axis=1)
rgb = np.dstack([np.clip((b - l) / (h - l) * 255, 0, 255) for b, l, h in zip(img, lo, hi)]).astype("uint8")
I = Image.fromarray(rgb).convert("RGB")
s = a.size / max(I.size); I = I.resize((round(I.width * s), round(I.height * s)))
d = ImageDraw.Draw(I)
inv = ~tr
pred = gpd.read_file(a.pred).to_crs(crs); gt = gpd.read_file(a.gt).to_crs(crs)
_, pm, gm = match(pred, gt, a.iou)
def draw(gdf, color, width):
    for g, m in zip(gdf.geometry, gdf.matched):
        for poly in getattr(g, "geoms", [g]):
            pts = [tuple(np.array(inv * xy) * s) for xy in poly.exterior.coords]
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill=color, width=width)
draw(gm, (0, 255, 255), 2)
draw(pm[~pm.matched], (255, 0, 255), 3)
draw(pm[pm.matched], (255, 255, 0), 2)
I.save(a.out, quality=85)
print("GT cyan | matched preds yellow | unmatched preds (FP) magenta")
