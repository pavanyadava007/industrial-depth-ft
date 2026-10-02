import numpy as np
import pytest

from idft import mvtec3d
from idft.evaluate import aggregate, eval_image
from idft.train import stratified_subset


def _fake_xyz(n=40, f=300.0, c=19.5):
    v, u = np.mgrid[0:n, 0:n].astype(np.float64)
    z = 0.6 + 0.001 * u
    xyz = np.stack([(u - c) * z / f, (v - c) * z / f, z], -1).astype(np.float32)
    xyz[:3, :3] = 0  # a hole
    return xyz


def test_fit_intrinsics():
    K = mvtec3d.fit_intrinsics(_fake_xyz())
    assert K["fx"] == pytest.approx(300.0, rel=1e-4) and K["cx"] == pytest.approx(19.5, abs=1e-3)


def test_downsample_keeps_holes_invalid():
    z = np.ones((4, 4), np.float32)
    z[0, 0] = 0
    d = mvtec3d.downsample_depth(z, 2)
    assert d[0, 0] == 0 and d[1, 1] == 1


def test_preprocess_roundtrip(tmp_path):
    xyz = _fake_xyz()
    rgb = np.zeros((40, 40, 3), np.uint8)
    mask = np.zeros((40, 40), np.uint8)
    mask[10:20, 10:20] = 255
    d = mvtec3d.preprocess_sample(rgb, xyz, mask, res=20)
    assert d["depth"].shape == (20, 20) and d["defect"].sum() == 25
    assert d["K"][0] == pytest.approx(150.0, rel=1e-3)
    p = tmp_path / "cat" / "test" / "crack" / "000.npz"
    p.parent.mkdir(parents=True)
    np.savez_compressed(p, **d)
    s = mvtec3d.index(tmp_path)
    assert len(s) == 1 and s[0].key == "cat/test/crack/000"
    loaded = mvtec3d.load(s[0])
    assert np.array_equal(loaded["depth"], d["depth"])
    r = eval_image(loaded["depth"] + 0.001, loaded["depth"], "metric")
    r["key"] = s[0].key
    agg = aggregate([r])
    assert agg["raw"]["rmse_mm"] == pytest.approx(1.0, rel=1e-3)
    assert agg["aligned"]["rmse_mm"] < 1e-6


def test_stratified_subset_covers_categories():
    samples = [mvtec3d.Sample(path=__import__("pathlib").Path(f"{c}{i}.npz"), category=c, split="train",
                              defect_type="good") for c in "abcde" for i in range(10)]
    sub = stratified_subset(samples, 10, seed=0)
    assert len(sub) == 10 and len({s.category for s in sub}) == 5
    assert [s.path for s in sub] == [s.path for s in stratified_subset(samples, 10, seed=0)]


def test_split_is_fixed():
    assert set(mvtec3d.SEEN) | set(mvtec3d.HELDOUT) == set(mvtec3d.CATEGORIES)
    assert len(mvtec3d.HELDOUT) == 3 and not set(mvtec3d.SEEN) & set(mvtec3d.HELDOUT)


def test_non_square_is_center_cropped():
    xyz = np.concatenate([_fake_xyz(40)] * 2, axis=0)[:60]  # 60 x 40
    d = mvtec3d.preprocess_sample(np.zeros((60, 40, 3), np.uint8), xyz, None, res=20)
    assert d["depth"].shape == (20, 20) and list(d["orig_size"]) == [60, 40]


def test_downsample_non_integer_factor():
    z = np.full((6, 6), 0.5, np.float32)
    z[0, 0] = 0
    d = mvtec3d.downsample_depth(z, 4)
    assert d[0, 0] == 0 and np.allclose(d[2:, 2:], 0.5)
