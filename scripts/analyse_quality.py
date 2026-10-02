"""Experiment E: error by height above the deepest point, edge accuracy and defect-region analysis.

Uses predictions cached under data/preds by eval_zeroshot.py / train_ft.py --save-preds. For metric models both
raw and per-image aligned predictions are analysed; for the disparity model only aligned predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from idft import mvtec3d
from idft.evaluate import pred_path
from idft.metrics import align_depth, align_disparity, edge_mask

HEIGHT_EDGES_MM = [0, 5, 15, 30, 60, 1000]  # height above the deepest valid point of the image


def relief(z: np.ndarray, valid: np.ndarray, size: int = 15) -> np.ndarray:
    """Local depth relief: depth minus the mean depth of the valid pixels in a size x size box around it."""
    import cv2

    v = valid.astype(np.float64)
    num = cv2.blur(np.where(valid, z, 0.0), (size, size))
    den = cv2.blur(v, (size, size))
    return np.where(valid, z - num / np.maximum(den, 1e-9), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", required=True, help="metric or disparity per variant")
    a = ap.parse_args()
    test = mvtec3d.select(mvtec3d.index(Path("data/proc")), splits=["test"])
    out = {"experiment": "E_quality", "height_edges_mm": HEIGHT_EDGES_MM, "variants": {}}
    for var, kind in zip(a.variants, a.kinds):
        modes = ["raw", "aligned"] if kind == "metric" else ["aligned"]
        acc = {m: {"height_sq": np.zeros(5), "height_abs": np.zeros(5), "height_n": np.zeros(5),
                   "edge": [], "flat": [], "def_in": [], "def_ring": [], "relief_corr": [], "relief_ratio": []}
               for m in modes}
        n_def = 0
        for s in test:
            p = pred_path(Path("data/preds"), var, s.key)
            if not p.exists():
                continue
            d = mvtec3d.load(s)
            gt, valid = d["depth"].astype(np.float64), d["depth"] > 0
            pred = np.load(p).astype(np.float64)
            valid &= np.isfinite(pred)
            preds = {}
            if kind == "metric":
                preds["raw"] = pred
                aligned = np.zeros_like(pred)
                aligned[valid] = align_depth(pred[valid], gt[valid])
                preds["aligned"] = aligned
            else:
                aligned = np.zeros_like(pred)
                aligned[valid] = align_disparity(pred[valid], gt[valid])
                preds["aligned"] = aligned
            h_mm = (gt[valid].max() - gt) * 1000
            hb = np.digitize(h_mm, HEIGHT_EDGES_MM) - 1
            em = edge_mask(gt, valid)
            fm = valid & ~binary_dilation(em, iterations=3)
            defect = d["defect"] & valid
            is_def = s.defect_type != "good" and defect.sum() >= 20
            if is_def:
                n_def += 1
                ring = binary_dilation(d["defect"], iterations=10) & ~binary_dilation(d["defect"], iterations=2) & valid
                rg = relief(gt, valid)
            for m, pr in preds.items():
                err = pr - gt
                A = acc[m]
                for b in range(5):
                    sel = valid & (hb == b)
                    A["height_sq"][b] += float(np.sum(err[sel] ** 2))
                    A["height_abs"][b] += float(np.sum(np.abs(err[sel])))
                    A["height_n"][b] += int(sel.sum())
                if em.sum() >= 20:
                    A["edge"].append(float(np.sqrt(np.mean(err[em] ** 2)) * 1000))
                    A["flat"].append(float(np.sqrt(np.mean(err[fm] ** 2)) * 1000))
                if is_def and ring.sum() >= 20:
                    A["def_in"].append(float(np.mean(np.abs(err[defect])) * 1000))
                    A["def_ring"].append(float(np.mean(np.abs(err[ring])) * 1000))
                    rp = relief(pr, valid)
                    g_, p_ = rg[defect], rp[defect]
                    if np.std(g_) > 0 and np.std(p_) > 0:
                        A["relief_corr"].append(float(np.corrcoef(g_, p_)[0, 1]))
                    A["relief_ratio"].append(float(np.mean(np.abs(p_)) / max(np.mean(np.abs(g_)), 1e-9)))
        res = {"n_defect_images": n_def}
        for m, A in acc.items():
            n = np.maximum(A["height_n"], 1)
            di, dr = np.array(A["def_in"]), np.array(A["def_ring"])
            res[m] = {
                "height_rmse_mm": (np.sqrt(A["height_sq"] / n) * 1000).tolist(),
                "height_mae_mm": (A["height_abs"] / n * 1000).tolist(),
                "height_frac_pixels": (A["height_n"] / A["height_n"].sum()).tolist(),
                "edge_rmse_mm_mean": float(np.mean(A["edge"])), "flat_rmse_mm_mean": float(np.mean(A["flat"])),
                "n_edge_images": len(A["edge"]),
                "defect_mae_mm_mean": float(di.mean()), "ring_mae_mm_mean": float(dr.mean()),
                "frac_images_defect_err_gt_ring": float(np.mean(di > dr)),
                "defect_relief_corr_median": float(np.median(A["relief_corr"])),
                "defect_relief_amplitude_ratio_median": float(np.median(A["relief_ratio"])),
                "n_defect_images_used": len(A["def_in"]),
            }
        out["variants"][var] = res
        print(var, json.dumps({m: {k: v for k, v in res[m].items() if not isinstance(v, list)} for m in modes}))
    # ground-truth defect relief amplitude, to say how deep the defects are in the first place
    amps = []
    for s in test:
        if s.defect_type == "good":
            continue
        d = mvtec3d.load(s)
        valid = d["depth"] > 0
        defect = d["defect"] & valid
        if defect.sum() >= 20:
            amps.append(float(np.mean(np.abs(relief(d["depth"].astype(np.float64), valid)[defect])) * 1000))
    out["gt_defect_relief_mm"] = {"median": float(np.median(amps)), "p10": float(np.percentile(amps, 10)),
                                  "p90": float(np.percentile(amps, 90)), "n": len(amps)}
    print("gt defect relief", out["gt_defect_relief_mm"])
    Path("results/quality.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
