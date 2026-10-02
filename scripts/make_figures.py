"""Small own figures for docs/figures (plots from results/*.json and depth / error maps rendered from cached
predictions). No dataset RGB image is reproduced; the depth and error maps are derived from MVTec 3D-AD ground
truth and are therefore shared under CC BY-NC-SA 4.0 with attribution to MVTec Software GmbH."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from idft import mvtec3d  # noqa: E402
from idft.evaluate import pred_path  # noqa: E402
from idft.metrics import align_depth, align_disparity  # noqa: E402

OUT = Path("docs/figures")
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def load(n):
    p = Path("results") / n
    return json.loads(p.read_text()) if p.exists() else None


def fig_model_comparison():
    from make_results import LABEL, ZS_ORDER

    rows = []
    for v in ZS_ORDER:
        r = load(f"zeroshot_{v}.json")
        if r and v != "baseline_constant":
            rows.append((LABEL[v], r["aligned"]["rmse_mm"] if "aligned" in r else np.nan,
                         r["raw"]["rmse_mm"] if "raw" in r else np.nan))
    best = load("best_config.json")["best_config_by_val"]
    b = load(f"ft_B_{best}.json")
    rows.append((f"fine-tuned DA-V2 Small ({best})", b["aligned"]["rmse_mm"], b["raw"]["rmse_mm"]))
    const = load("zeroshot_baseline_constant.json")["aligned"]["rmse_mm"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    y = np.arange(len(rows))
    for ax, k, title, col in ((axes[0], 1, "aligned (per-image scale + shift), RMSE mm", BLUE),
                              (axes[1], 2, "raw metric depth, RMSE mm (log scale)", ORANGE)):
        vals = [r[k] for r in rows]
        ax.barh(y, vals, color=col, height=0.6)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.1f}", va="center", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9, loc="left")
        if k == 1:
            ax.axvline(const, color=GREY, ls="--", lw=1)
            ax.text(const, len(rows) - 0.4, " constant-map floor", color=GREY, fontsize=7)
        else:
            ax.set_xscale("log")
    axes[0].set_yticks(y, [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("MVTec 3D-AD test split (all 10 categories), lower is better", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=110)


def fig_data_curve():
    from make_results import c_runs

    xs, ms, ss = [], [], []
    for n in ("10", "50", "200", "all"):
        runs = c_runs(n)
        if runs:
            v = [r["raw"]["rmse_mm"] for r in runs]
            xs.append(runs[0]["n_train_images"])
            ms.append(np.mean(v))
            ss.append(np.std(v, ddof=1) if len(v) > 1 else 0)
    base = load("zeroshot_baseline_train_mean_map.json")["raw"]["rmse_mm"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.errorbar(xs, ms, yerr=ss, color=BLUE, marker="o", ms=5, lw=2, capsize=3, label="fine-tuned, mean +- std")
    ax.axhline(base, color=GREY, ls="--", lw=1)
    ax.text(xs[0], base, " per-category mean training depth map", color=GREY, fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("training images N (stratified over 10 categories)")
    ax.set_ylabel("test raw RMSE mm")
    ax.set_title("Training-data requirement", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "data_requirement.png", dpi=110)


def fig_error_maps(keys: list[str], zs_var: str, zs_kind: str, ft_var: str):
    fig, axes = plt.subplots(len(keys), 4, figsize=(9, 2.3 * len(keys)))
    for i, key in enumerate(keys):
        cat, split, dt, sid = key.split("/")
        d = mvtec3d.load(Path("data/proc") / cat / split / dt / f"{sid}.npz")
        gt = d["depth"].astype(np.float64)
        v = gt > 0
        zp = np.load(pred_path(Path("data/preds"), zs_var, key)).astype(np.float64)
        za = np.zeros_like(gt)
        za[v] = (align_depth if zs_kind == "metric" else align_disparity)(zp[v], gt[v])
        fp = np.load(pred_path(Path("data/preds"), ft_var, key)).astype(np.float64)
        lo, hi = np.percentile(gt[v], [1, 99])
        g = np.where(v, gt, np.nan)
        axes[i, 0].imshow(g, cmap="viridis_r", vmin=lo, vmax=hi)
        axes[i, 0].set_title(f"{cat} / {dt}: GT depth", fontsize=8)
        e1 = np.where(v, (za - gt) * 1000, np.nan)
        e2 = np.where(v, (fp - gt) * 1000, np.nan)
        lim = 10
        axes[i, 1].imshow(e1, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[i, 1].set_title(f"zero-shot {zs_var}, aligned\nRMSE {np.sqrt(np.nanmean(e1**2)):.1f} mm", fontsize=8)
        im = axes[i, 2].imshow(e2, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[i, 2].set_title(f"fine-tuned, raw metric\nRMSE {np.sqrt(np.nanmean(e2**2)):.1f} mm", fontsize=8)
        m = d["defect"]
        axes[i, 3].imshow(np.where(v, np.abs(e2), np.nan), cmap="magma", vmin=0, vmax=lim)
        if m.any():
            axes[i, 3].contour(m, levels=[0.5], colors=[AQUA], linewidths=1)
        axes[i, 3].set_title("fine-tuned |error| mm,\ndefect mask outlined", fontsize=8)
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.colorbar(im, ax=axes[:, 1:3], shrink=0.6, label="error mm (pred - GT)")
    fig.text(0.01, 0.005, "Derived from MVTec 3D-AD (c) MVTec Software GmbH, CC BY-NC-SA 4.0. Grey = no GT.",
             fontsize=7, color=GREY)
    fig.savefig(OUT / "error_maps.png", dpi=100, bbox_inches="tight")


def fig_height():
    q = load("quality.json")
    if not q:
        return
    e = q["height_edges_mm"]
    labels = [f"{e[i]}-{e[i + 1]}" if i < len(e) - 2 else f">={e[i]}" for i in range(len(e) - 1)]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(labels))
    series = [(k, m) for k in q["variants"] for m in ("aligned", "raw") if m in q["variants"][k]]
    series = [s for s in series if s[1] == "aligned"][:3]
    w = 0.8 / max(len(series), 1)
    for j, ((k, m), col) in enumerate(zip(series, (BLUE, ORANGE, AQUA))):
        ax.bar(x + j * w, q["variants"][k][m]["height_rmse_mm"], w * 0.92, color=col, label=f"{k} ({m})")
    ax.set_xticks(x + w * (len(series) - 1) / 2, labels)
    ax.set_xlabel("height above the deepest point of the image (mm)")
    ax.set_ylabel("RMSE mm")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Error by height (aligned predictions)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "error_by_height.png", dpi=110)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, "scripts")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_model_comparison()
    fig_data_curve()
    fig_height()
    keys = sys.argv[1:] if len(sys.argv) > 1 else []
    if keys:
        import os

        best = load("best_config.json")["best_config_by_val"]
        fig_error_maps(keys, os.environ.get("ZS_VAR", "depth_pro"), os.environ.get("ZS_KIND", "metric"), f"B_{best}")
    print("figures written")
