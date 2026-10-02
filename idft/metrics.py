"""Depth metrics and alignment for close-range metric depth (numpy only).

All depths are in metres; RMSE and MAE are reported in millimetres. Metrics are computed per image on the valid
ground-truth pixels and then averaged over images (each image has equal weight).
"""

from __future__ import annotations

import numpy as np

METRIC_KEYS = ["abs_rel", "rmse_mm", "mae_mm", "d1", "d_1.01", "d_1.005"]


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Metrics on paired 1-D arrays of positive depths in metres.

    d1 is the usual delta < 1.25 accuracy. Because the whole scene spans only a few centimetres at about 0.6 m
    working distance, d1 saturates for any sensible prediction, so the much tighter thresholds 1.01 and 1.005
    (about 6 mm and 3 mm at 0.6 m) are reported as well.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 1:
        raise ValueError("pred and gt must be 1-D arrays of equal length")
    if gt.size == 0:
        return {k: float("nan") for k in METRIC_KEYS}
    pred = np.maximum(pred, 1e-6)
    err = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(err) / gt)),
        "rmse_mm": float(np.sqrt(np.mean(err**2)) * 1000.0),
        "mae_mm": float(np.mean(np.abs(err)) * 1000.0),
        "d1": float(np.mean(ratio < 1.25)),
        "d_1.01": float(np.mean(ratio < 1.01)),
        "d_1.005": float(np.mean(ratio < 1.005)),
    }


def fit_scale_shift(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Least-squares (s, t) minimising ||s * pred + t - gt||^2 on 1-D arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    A = np.stack([pred, np.ones_like(pred)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, gt, rcond=None)
    return float(s), float(t)


def align_depth(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Scale-and-shift alignment in depth space (for models that output metric or affine depth)."""
    s, t = fit_scale_shift(pred, gt)
    return s * np.asarray(pred, dtype=np.float64) + t


def align_disparity(pred_disp: np.ndarray, gt: np.ndarray, min_depth: float = 1e-3) -> np.ndarray:
    """Scale-and-shift alignment in inverse depth (for affine-invariant disparity models), returned as depth.

    The fit is done on 1 / gt; the aligned disparity is clamped to 1 / (10 * max(gt)) .. 1 / min_depth before
    inversion so that a non-positive aligned disparity cannot produce negative or infinite depth.
    """
    gt = np.asarray(gt, dtype=np.float64)
    s, t = fit_scale_shift(pred_disp, 1.0 / gt)
    disp = s * np.asarray(pred_disp, dtype=np.float64) + t
    disp = np.clip(disp, 1.0 / (10.0 * gt.max()), 1.0 / min_depth)
    return 1.0 / disp


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}


def depth_bins(gt: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index per pixel for given edges (len k+1) -> values 0..k-1, -1 outside."""
    idx = np.digitize(gt, edges) - 1
    idx[(gt < edges[0]) | (gt >= edges[-1])] = -1
    return idx


def edge_mask(depth: np.ndarray, valid: np.ndarray, thresh_m: float = 0.002, dilate: int = 2) -> np.ndarray:
    """Pixels near a depth discontinuity: |forward difference| > thresh between two valid neighbours, dilated."""
    d = depth.astype(np.float64)
    e = np.zeros_like(valid, dtype=bool)
    gx = np.abs(np.diff(d, axis=1)) > thresh_m
    gx &= valid[:, 1:] & valid[:, :-1]
    gy = np.abs(np.diff(d, axis=0)) > thresh_m
    gy &= valid[1:, :] & valid[:-1, :]
    e[:, 1:] |= gx
    e[:, :-1] |= gx
    e[1:, :] |= gy
    e[:-1, :] |= gy
    if dilate > 0:
        from scipy.ndimage import binary_dilation

        e = binary_dilation(e, iterations=dilate)
    return e & valid
