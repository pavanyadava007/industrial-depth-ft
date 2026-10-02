"""Reference baselines on the test split (no learning beyond per-pixel averages of the training ground truth).

- baseline_train_mean_map: per category, the per-pixel mean of all valid training depth maps (fixed-rig prior;
  it knows the category, so it is a strong and slightly privileged reference for raw metric depth).
- baseline_constant: a constant map; after per-image scale-and-shift alignment it equals the per-image mean
  ground-truth depth, so its aligned RMSE is the per-image standard deviation of the ground truth. Any aligned
  number of a real model should be read against this floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from idft import mvtec3d
from idft.evaluate import aggregate, eval_image


def main() -> None:
    allsamples = mvtec3d.index(Path("data/proc"))
    test = mvtec3d.select(allsamples, splits=["test"])
    means = {}
    for c in mvtec3d.CATEGORIES:
        tr = mvtec3d.select(allsamples, categories=[c], splits=["train"])
        if not tr:
            continue
        s = np.zeros((mvtec3d.WORK_RES,) * 2)
        n = np.zeros_like(s)
        for t in tr:
            z = mvtec3d.load(t)["depth"]
            s += z
            n += z > 0
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        means[c] = np.where(np.isnan(m), np.nanmedian(m), m)
    rows_mean, rows_const = [], []
    for t in test:
        gt = mvtec3d.load(t)["depth"]
        r = eval_image(means[t.category], gt, "metric")
        r["key"] = t.key
        rows_mean.append(r)
        r = eval_image(np.ones_like(gt), gt, "metric")
        r.pop("raw")
        r["key"] = t.key
        rows_const.append(r)
    for name, rows, note in (
        ("baseline_train_mean_map", rows_mean, "per-category per-pixel mean of training depth maps"),
        ("baseline_constant", rows_const, "constant map; aligned = per-image mean GT (floor for aligned scores)"),
    ):
        res = {"experiment": "A_zero_shot", "variant": name, "kind": "baseline", "note": note,
               **aggregate(rows), "per_image": rows}
        Path(f"results/zeroshot_{name}.json").write_text(json.dumps(res, indent=1))
        print(name, {m: {k: round(v, 4) for k, v in res[m].items()} for m in ("raw", "aligned") if m in res})


if __name__ == "__main__":
    main()
