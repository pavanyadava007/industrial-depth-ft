"""Shared evaluation: per-image raw and aligned metrics, aggregation per category, prediction caching."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from idft.metrics import align_depth, align_disparity, depth_metrics, mean_metrics


def eval_image(pred: np.ndarray, gt: np.ndarray, kind: str) -> dict[str, dict[str, float]]:
    """kind: 'metric' -> raw + depth-space scale/shift aligned; 'disparity' -> disparity-space aligned only."""
    valid = gt > 0
    if kind == "metric":
        valid &= np.isfinite(pred) & (pred > 0)
    else:
        valid &= np.isfinite(pred)
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    out = {}
    if kind == "metric":
        out["raw"] = depth_metrics(p, g)
        out["aligned"] = depth_metrics(align_depth(p, g), g)
    else:
        out["aligned"] = depth_metrics(align_disparity(p, g), g)
    out["n_valid"] = int(valid.sum())
    return out


def aggregate(rows: list[dict]) -> dict:
    """rows: [{'key': 'cat/split/defect/id', 'raw': {...}, 'aligned': {...}}] -> overall + per category means."""
    res = {"n_images": len(rows)}
    modes = [m for m in ("raw", "aligned") if rows and m in rows[0]]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["key"].split("/")[0]].append(r)
    for m in modes:
        res[m] = mean_metrics([r[m] for r in rows])
        res[f"{m}_per_category"] = {c: mean_metrics([r[m] for r in rs]) for c, rs in sorted(by_cat.items())}
    return res


def pred_path(root: Path, variant: str, key: str) -> Path:
    return Path(root) / variant / (key.replace("/", "__") + ".npy")


def save_pred(root: Path, variant: str, key: str, pred: np.ndarray, dtype=np.float32) -> None:
    """Cache a prediction. The zero-shot runs of 2026-10-02 stored float16 (spacing 0.49 mm between 0.5 and 1 m,
    0.98 mm between 1 and 2 m); fine-tuned runs store float32. Only Experiment E and the figures read the cache."""
    p = pred_path(root, variant, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, pred.astype(dtype))
