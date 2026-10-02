"""Load run, sensor and model configuration from the YAML files in configs/."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("HALO_CONFIG_DIR", REPO_ROOT / "configs"))


@dataclass
class RunConfig:
    tile_size: int = 256
    stride: int = 192
    edge_mode: str = "shift"          # "shift" | "pad"
    mask_nodata: bool = True
    target_gsd_m: float | None = None
    rescale: str = "none"             # "none" | "percentile" (scene-level 2-98% stretch to 0-255)
    batch_size: int = 8
    threshold: float = 0.5
    min_area_m2: float = 0.0
    polygon_strip_rows: int = 4096
    polygon_strip_overlap: int = 512
    write_probability: bool = True
    vector_formats: list = field(default_factory=lambda: ["gpkg", "shp"])

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "RunConfig":
        data = _read_yaml(Path(path) if path else CONFIG_DIR / "default.yaml")
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0 < self.stride <= self.tile_size:
            raise ValueError("stride must be in (0, tile_size]")
        if (self.tile_size - self.stride) % 2:
            raise ValueError("tile_size - stride must be even (symmetric overlap)")
        if self.rescale not in ("none", "percentile"):
            raise ValueError("rescale must be 'none' or 'percentile'")
        if self.edge_mode not in ("shift", "pad"):
            raise ValueError("edge_mode must be 'shift' or 'pad'")


@dataclass
class SensorConfig:
    name: str
    bands: dict
    nodata: float | None = None


@dataclass
class ModelConfig:
    name: str
    path: str
    input_bands: list
    tile_size: int = 256
    native_gsd_m: float | None = None
    sha256: str | None = None
    trained_on: str = ""

    def resolved_path(self) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(self.path)))
        return p if p.is_absolute() else REPO_ROOT / p


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_sensor(name: str) -> SensorConfig:
    sensors = _read_yaml(CONFIG_DIR / "sensors.yaml")
    if name not in sensors:
        raise KeyError(f"Unknown sensor '{name}'. Known: {sorted(sensors)}")
    return SensorConfig(name=name, **sensors[name])


def load_model_config(name: str, path_override: str | None = None) -> ModelConfig:
    models = _read_yaml(CONFIG_DIR / "models.yaml")
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(models)}")
    cfg = ModelConfig(name=name, **models[name])
    if path_override:
        cfg.path = path_override
    return cfg


def band_indices(sensor: SensorConfig, model: ModelConfig, override: list[int] | None = None) -> list[int]:
    """1-based file band indices to feed the model, in the model's expected order."""
    if override:
        return list(override)
    missing = [b for b in model.input_bands if b not in sensor.bands]
    if missing:
        raise ValueError(f"Sensor '{sensor.name}' has no band(s) {missing} required by model '{model.name}'")
    return [int(sensor.bands[b]) for b in model.input_bands]
