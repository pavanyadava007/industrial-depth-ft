import numpy as np
import pytest

from idft.metrics import align_depth, align_disparity, depth_metrics, edge_mask, fit_scale_shift, mean_metrics


def test_perfect_prediction():
    gt = np.linspace(0.6, 0.7, 100)
    m = depth_metrics(gt.copy(), gt)
    assert m["abs_rel"] == 0 and m["rmse_mm"] == 0 and m["d1"] == 1 and m["d_1.005"] == 1


def test_known_values_mm():
    gt = np.full(4, 0.5)
    pred = gt + np.array([0.001, -0.001, 0.003, -0.003])
    m = depth_metrics(pred, gt)
    assert m["mae_mm"] == pytest.approx(2.0)
    assert m["rmse_mm"] == pytest.approx(np.sqrt((1 + 1 + 9 + 9) / 4))
    assert m["abs_rel"] == pytest.approx(0.004)
    assert m["d_1.005"] == pytest.approx(0.5)  # 0.501/0.5 < 1.005, 0.503/0.5 > 1.005


def test_shape_check():
    with pytest.raises(ValueError):
        depth_metrics(np.ones((2, 2)), np.ones((2, 2)))


def test_scale_shift_recovers_affine():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.6, 0.7, 1000)
    pred = (gt - 0.1) / 3.0
    s, t = fit_scale_shift(pred, gt)
    assert s == pytest.approx(3.0) and t == pytest.approx(0.1)
    assert np.allclose(align_depth(pred, gt), gt)


def test_disparity_alignment_recovers_inverse_depth():
    rng = np.random.default_rng(1)
    gt = rng.uniform(0.6, 0.7, 1000)
    disp = 5.0 / gt + 2.0  # affine in inverse depth
    assert np.allclose(align_disparity(disp, gt), gt)


def test_edge_mask_marks_step():
    d = np.full((10, 10), 0.6)
    d[:, 5:] = 0.65
    e = edge_mask(d, np.ones_like(d, dtype=bool), thresh_m=0.01, dilate=0)
    assert e[:, 4].all() and e[:, 5].all() and not e[:, 0].any()


def test_mean_metrics():
    assert mean_metrics([{"a": 1.0}, {"a": 3.0}]) == {"a": 2.0}
