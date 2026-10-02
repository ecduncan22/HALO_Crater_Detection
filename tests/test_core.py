import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from halo_craters.config import RunConfig
from halo_craters.postprocess import polygonize
from halo_craters.preprocess import normalize_tile
from halo_craters.tiling import keep_ranges, tile_offsets


@pytest.mark.parametrize("length", [1, 100, 256, 257, 1000, 2213, 4096])
@pytest.mark.parametrize("stride", [128, 192, 256])
@pytest.mark.parametrize("mode", ["shift", "pad"])
def test_keep_ranges_cover_exactly_once(length, stride, mode):
    offs = tile_offsets(length, 256, stride, mode)
    ranges = keep_ranges(offs, length, 256)
    covered = np.zeros(length, int)
    for o, (a, b) in zip(offs, ranges):
        assert o <= a <= b <= min(o + 256, length) or (a == b)
        covered[a:b] += 1
    assert (covered == 1).all()


def test_normalize_matches_original():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2000, (256, 256, 3)).astype(np.uint16)
    tf_ = t.astype(np.float32)
    orig = (tf_ - tf_.mean(axis=(0, 1))) / (tf_.std(axis=(0, 1)) + 1e-8)
    np.testing.assert_allclose(normalize_tile(t), orig, rtol=1e-5, atol=1e-5)


def test_normalize_masks_nodata():
    t = np.full((10, 10, 3), 500, np.uint16)
    t[:5] = np.arange(50).reshape(5, 10, 1)
    t[5:] = 0
    valid = ~np.all(t == 0, axis=-1)
    out = normalize_tile(t, valid)
    assert np.all(out[5:] == 0)
    v = out[:5].reshape(-1, 3)
    np.testing.assert_allclose(v.mean(0), 0, atol=1e-5)


def test_polygonize_strips_give_same_result(tmp_path):
    rng = np.random.default_rng(1)
    H = W = 600
    prob = np.zeros((H, W), np.uint8)
    yy, xx = np.mgrid[:H, :W]
    for _ in range(40):
        cy, cx, r = rng.integers(10, H - 10), rng.integers(10, W - 10), rng.integers(2, 9)
        prob[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 200
    p = tmp_path / "prob.tif"
    with rasterio.open(p, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8",
                       crs="EPSG:32637", transform=from_origin(500000, 5000000, 0.5, 0.5)) as d:
        d.write(prob, 1)
    whole = polygonize(p, RunConfig(polygon_strip_rows=10000, polygon_strip_overlap=0))
    strips = polygonize(p, RunConfig(polygon_strip_rows=97, polygon_strip_overlap=40))
    assert len(whole) == len(strips)
    assert abs(whole.area.sum() - strips.area.sum()) < 1e-6
