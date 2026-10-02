"""Command-line interface.

    halo-craters run INPUT --sensor vantor --model vantor_2022 --out outputs/
    halo-craters evaluate PRED.gpkg REF.shp [--aoi AOI.shp] [--iou 0.1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import RunConfig


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="halo-craters", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="detect craters in a .tif or a directory of .tif files")
    r.add_argument("input", help="path to a .tif, or a directory (searched recursively)")
    r.add_argument("--sensor", required=True, help="sensor layout from configs/sensors.yaml (e.g. skysat, vantor)")
    r.add_argument("--model", required=True, help="model name from configs/models.yaml (e.g. vantor_2022)")
    r.add_argument("--out", default="outputs", help="output directory (default: outputs/)")
    r.add_argument("--model-path", help="override the weights path from models.yaml")
    r.add_argument("--config", help="run config YAML (default: configs/default.yaml)")
    r.add_argument("--bands", type=_int_list, help="override: 1-based file bands to feed, e.g. 3,2,1")
    r.add_argument("--stride", type=int)
    r.add_argument("--edge-mode", choices=["shift", "pad"])
    r.add_argument("--target-gsd", type=float, dest="target_gsd_m", help="resample to this pixel size (m)")
    r.add_argument("--rescale", choices=["none", "percentile"], help="percentile: stretch to 8-bit-like 0-255 first")
    r.add_argument("--threshold", type=float)
    r.add_argument("--min-area", type=float, dest="min_area_m2")
    r.add_argument("--batch-size", type=int)
    r.add_argument("--no-mask-nodata", action="store_true", help="include nodata in tile stats (original behaviour)")
    r.add_argument("--parity", action="store_true",
                   help="reproduce the original notebook: stride 256, zero-padded edges, nodata not masked")
    r.add_argument("--tag", help="suffix added to output names, to keep experiments apart")

    e = sub.add_parser("evaluate", help="compare predicted polygons with reference polygons")
    e.add_argument("pred")
    e.add_argument("ref")
    e.add_argument("--aoi", help="polygon file; only objects with centroid inside are counted")
    e.add_argument("--iou", type=float, default=0.1)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("botocore", "boto3", "rasterio", "pyogrio", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.cmd == "run":
        from .pipeline import run
        over = dict(stride=a.stride, edge_mode=a.edge_mode, target_gsd_m=a.target_gsd_m, rescale=a.rescale,
                    threshold=a.threshold, min_area_m2=a.min_area_m2, batch_size=a.batch_size)
        if a.parity:
            over.update(stride=256, edge_mode="pad", mask_nodata=False)
        if a.no_mask_nodata:
            over["mask_nodata"] = False
        cfg = RunConfig.load(a.config, **over)
        out = run(a.input, a.out, a.sensor, a.model, cfg, a.model_path, a.bands, a.tag)
        print(json.dumps([{k: s[k] for k in ("image", "n_craters", "inference_seconds", "outputs")} for s in out], indent=2))
        return 0

    if a.cmd == "evaluate":
        import geopandas as gpd
        from .evaluate import clip_to_aoi, match
        pred, ref = gpd.read_file(a.pred), gpd.read_file(a.ref)
        if a.aoi:
            aoi = gpd.read_file(a.aoi)
            pred, ref = clip_to_aoi(pred, aoi), clip_to_aoi(ref, aoi)
        res, _, _ = match(pred, ref, a.iou)
        print(json.dumps(res.as_dict(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
