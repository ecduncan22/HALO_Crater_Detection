"""Load the Keras U-Net crater models (.h5, saved with Keras 2.4).

Inference only, so models are loaded with compile=False and the custom training
losses/metrics are not needed. Requires TensorFlow <= 2.15 (Keras 2).
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

log = logging.getLogger(__name__)


def sha256sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path: str | Path, expected_sha256: str | None = None, verify: bool = False):
    import tensorflow as tf

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {path}. Download them into models/ (see README)."
        )
    if verify and expected_sha256:
        actual = sha256sum(path)
        if actual != expected_sha256:
            log.warning("Model checksum mismatch for %s: %s != %s", path, actual, expected_sha256)
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    log.info("TensorFlow %s, GPUs: %s", tf.__version__, [g.name for g in gpus] or "none (CPU)")
    model = tf.keras.models.load_model(str(path), compile=False)
    log.info("Loaded model %s: input %s -> output %s", path.name, model.input_shape, model.output_shape)
    return model


def predict_batch(model, batch):
    """(N, H, W, C) float32 -> (N, H, W) float32 probabilities."""
    import numpy as np

    out = model(np.asarray(batch, dtype="float32"), training=False)
    return np.asarray(out)[..., 0]
