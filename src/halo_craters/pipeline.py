"""Run the whole pipeline on one scene or a directory of scenes."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import RunConfig, band_indices, load_model_config, load_sensor
from .inference import run_inference
from .model import load_model
from .postprocess import polygonize, write_vectors

log = logging.getLogger(__name__)
RASTER_EXT = (".tif", ".tiff", ".TIF", ".TIFF")


def list_inputs(inp: str | Path) -> list[Path]:
    s = str(inp)
    if s.startswith("s3://"):
        raise NotImplementedError("S3 input is planned for Phase 7; download the scene locally for now.")
    p = Path(inp)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix in RASTER_EXT and not f.name.endswith(".tmp.tif"))
        if not files:
            raise FileNotFoundError(f"No .tif files found under {p}")
        return files
    raise FileNotFoundError(inp)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(
    inp: str | Path,
    out_dir: str | Path,
    sensor: str,
    model_name: str,
    cfg: RunConfig,
    model_path: str | None = None,
    bands_override: list[int] | None = None,
    tag: str | None = None,
) -> list[dict]:
    sensor_cfg = load_sensor(sensor)
    model_cfg = load_model_config(model_name, model_path)
    bands = band_indices(sensor_cfg, model_cfg, bands_override)
    model = load_model(model_cfg.resolved_path(), model_cfg.sha256)
    out_dir = Path(out_dir)
    summaries = []
    for img in list_inputs(inp):
        stem = img.stem + (f"__{tag}" if tag else "")
        scene_dir = out_dir / stem
        res = run_inference(img, scene_dir / f"{stem}_prob.tif", model, bands, cfg, nodata=sensor_cfg.nodata)
        gdf = polygonize(res.probability_path, cfg,
                         attrs={"source_im": img.name[:254], "model": model_name, "sensor": sensor})
        vec = write_vectors(gdf, scene_dir / f"{stem}_craters", cfg.vector_formats)
        if not cfg.write_probability:
            res.probability_path.unlink()
        summary = {
            "image": str(img), "sensor": sensor, "model": model_name, "bands_fed": bands,
            "model_input_bands": model_cfg.input_bands, "gsd_m": res.gsd_m,
            "n_tiles": res.n_tiles, "n_tiles_skipped": res.n_tiles_skipped,
            "inference_seconds": round(res.seconds, 1), "n_craters": len(gdf),
            "outputs": [str(p) for p in vec] + ([str(res.probability_path)] if cfg.write_probability else []),
            "config": asdict(cfg), "halo_craters_version": __version__, "git_commit": _git_commit(),
            "host": platform.node(), "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (scene_dir / "run.json").write_text(json.dumps(summary, indent=2))
        log.info("%s -> %d craters", img.name, len(gdf))
        summaries.append(summary)
    return summaries
