"""Convert one extracted MVTec 3D-AD category into data/proc/<cat>/<split>/<defect>/<id>.npz."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

from idft.mvtec3d import preprocess_sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_cat_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/proc"))
    a = ap.parse_args()
    cat = a.raw_cat_dir.name
    t0 = time.time()
    stats = {"category": cat, "n": 0, "sizes": {}, "fit_resid_max": 0.0, "valid_frac": [], "z_p1": [], "z_p99": []}
    for rgb_path in sorted(a.raw_cat_dir.glob("*/*/rgb/*.png")):
        split, dtype = rgb_path.parts[-4], rgb_path.parts[-3]
        xyz = tifffile.imread(rgb_path.parent.parent / "xyz" / (rgb_path.stem + ".tiff"))
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt_path = rgb_path.parent.parent / "gt" / rgb_path.name
        mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
        d = preprocess_sample(rgb, xyz, mask)
        out = a.out / cat / split / dtype / (rgb_path.stem + ".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **d)
        stats["n"] += 1
        key = f"{xyz.shape[0]}x{xyz.shape[1]}"
        stats["sizes"][key] = stats["sizes"].get(key, 0) + 1
        stats["fit_resid_max"] = max(stats["fit_resid_max"], float(d["fit_resid"]))
        z = d["depth"]
        v = z > 0
        stats["valid_frac"].append(float(v.mean()))
        if v.any():
            stats["z_p1"].append(float(np.percentile(z[v], 1)))
            stats["z_p99"].append(float(np.percentile(z[v], 99)))
    summary = {
        "category": cat, "n": stats["n"], "raw_sizes": stats["sizes"], "fit_resid_max": stats["fit_resid_max"],
        "valid_frac_mean": float(np.mean(stats["valid_frac"])),
        "z_p1_median_m": float(np.median(stats["z_p1"])), "z_p99_median_m": float(np.median(stats["z_p99"])),
        "seconds": time.time() - t0,
    }
    (a.out / cat / "prep_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
