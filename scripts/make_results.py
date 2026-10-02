"""Generate every results table from results/*.json and write them between marker comments in README.md,
docs/RESULTS.md and STATUS.md. No number in those files is typed by hand."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("results")
CATS = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]
HELDOUT = ["cable_gland", "peach", "rope"]
SEEN = [c for c in CATS if c not in HELDOUT]
ZS_ORDER = ["baseline_constant", "baseline_train_mean_map", "dav2_small_rel", "dav2_metric_indoor_small",
            "dav2_metric_indoor_large", "dav2_metric_outdoor_large", "moge2", "moge2_gtfocal", "depth_pro",
            "depth_pro_gtfocal"]
ZS_MODEL = {"moge2_gtfocal": "moge2", "depth_pro_gtfocal": "depth_pro"}
LABEL = {
    "baseline_constant": "constant map (reference floor)",
    "baseline_train_mean_map": "per-category mean training depth map",
    "dav2_small_rel": "DA-V2 Small, relative",
    "dav2_metric_indoor_small": "DA-V2 metric indoor Small",
    "dav2_metric_indoor_large": "DA-V2 metric indoor Large",
    "dav2_metric_outdoor_large": "DA-V2 metric outdoor Large",
    "moge2": "MoGe-2 ViT-L (own focal)",
    "moge2_gtfocal": "MoGe-2 ViT-L (true focal)",
    "depth_pro": "Depth Pro (own focal)",
    "depth_pro_gtfocal": "Depth Pro (true focal)",
}


def load(name: str) -> dict | None:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def pm(vals, nd=2) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f} +- {np.std(vals, ddof=1):.{nd}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def subset_mean(res: dict, mode: str, cats: list[str], key: str) -> float | None:
    vals = [r[mode][key] for r in res["per_image"] if r["key"].split("/")[0] in cats and mode in r]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------- data
def data_table() -> str:
    from idft import mvtec3d

    samples = mvtec3d.index(Path("data/proc"))
    cnt = Counter((s.category, s.split) for s in samples)
    rows = []
    for c in CATS:
        p = Path("data/proc") / c / "prep_summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        n_def = sum(1 for s in samples if s.category == c and s.split == "test" and s.defect_type != "good")
        rows.append([c, "held out" if c in HELDOUT else "seen", ", ".join(d["raw_sizes"]), str(cnt[(c, "train")]),
                     str(cnt[(c, "validation")]), f"{cnt[(c, 'test')]} ({n_def} defective)",
                     f(d["z_p1_median_m"], 3) + " - " + f(d["z_p99_median_m"], 3), f(d["valid_frac_mean"] * 100, 1)])
    tot = [str(sum(cnt[(c, s)] for c in CATS)) for s in ("train", "validation", "test")]
    rows.append(["**all**", "", "", *tot, "", ""])
    return table(["category", "Exp. D role", "raw size", "train", "val", "test", "depth p1 - p99 (m, median over images)",
                  "valid GT pixels %"], rows)


# ---------------------------------------------------------------- A
def zs_results() -> dict[str, dict]:
    out = {}
    for v in ZS_ORDER:
        r = load(f"zeroshot_{v}.json")
        if r:
            out[v] = r
    return out


def table_a_main(zs) -> str:
    rows = []
    for v, r in zs.items():
        raw, al = r.get("raw"), r.get("aligned", {})
        lat = r.get("latency_ms_median")
        rows.append([LABEL[v], r.get("kind", ""),
                     f(raw["abs_rel"], 3) if raw else "n/a", f(raw["rmse_mm"], 1) if raw else "n/a",
                     f(raw["d1"], 3) if raw else "n/a",
                     f(al["abs_rel"], 4), f(al["rmse_mm"], 2), f(al["mae_mm"], 2), f(al["d_1.01"], 3), f(al["d1"], 3),
                     f(lat, 1) if lat else "n/a"])
    return table(["model", "output", "raw AbsRel", "raw RMSE mm", "raw d1", "aligned AbsRel", "aligned RMSE mm",
                  "aligned MAE mm", "aligned d<1.01", "aligned d1", "median ms/img (NVIDIA L4)"], rows)


def table_a_cat(zs, mode: str) -> str:
    rows = []
    for v, r in zs.items():
        pc = r.get(f"{mode}_per_category")
        if not pc:
            continue
        rows.append([LABEL[v]] + [f(pc[c]["rmse_mm"], 1) if c in pc else "n/a" for c in CATS])
    return table(["model (" + mode + " RMSE mm)"] + CATS, rows)


def table_licence(zs) -> str:
    rows, seen = [], set()
    for v, r in zs.items():
        if r.get("kind") == "baseline":
            continue
        m = ZS_MODEL.get(v, v)
        if m in seen:
            continue
        seen.add(m)
        rows.append([LABEL[m].split(" (")[0], r.get("licence", "?"), r.get("commercial_use", "?")])
    rows.append(["this project's fine-tuned DA-V2 Small", "weights derived from Apache-2.0 DA-V2 Small, but trained on "
                 "MVTec 3D-AD (CC BY-NC-SA 4.0) - not usable commercially without MVTec's permission", "no (data licence)"])
    return table(["model", "licence of released weights", "commercial use"], rows)


# ---------------------------------------------------------------- B, C, D
def ft(name):
    return load(f"ft_{name}.json")


def table_b() -> str:
    rows = []
    for mode, label in (("head", "(1) decoder + head only"), ("full", "(2) full fine-tuning, encoder LR / 10"),
                        ("lora", "(3) LoRA r=16 on encoder attention + full decoder")):
        r = ft(f"B_{mode}")
        if not r:
            rows.append([label] + ["not run"] * 10)
            continue
        c = r["config"]
        lr = f"dec {c['lr_decoder']:g}" + (f", enc {c['lr_encoder']:g}" if mode == "full" else "")
        rows.append([label, lr, f"{r['params']['trainable'] / 1e6:.2f} M", f(r["best_val_rmse_mm"], 2),
                     f"{r['best_step']} / {c['steps']}", f(r["raw"]["abs_rel"], 4), f(r["raw"]["rmse_mm"], 2),
                     f(r["raw"]["mae_mm"], 2), f(r["raw"]["d_1.01"], 3), f(r["aligned"]["rmse_mm"], 2),
                     f(r["train_s"] / 60, 1)])
    return table(["configuration", "learning rate", "trainable params", "val RMSE mm (selection)", "best step",
                  "test raw AbsRel", "test raw RMSE mm", "test raw MAE mm", "test raw d<1.01", "test aligned RMSE mm",
                  "train min (NVIDIA L4)"], rows)


def best_cfg() -> str | None:
    b = load("best_config.json")
    return b["best_config_by_val"] if b else None


def c_runs(n: str) -> list[dict]:
    best = best_cfg()
    runs = []
    for s in (0, 1, 2):
        if n == "all":
            r = ft(f"B_{best}") if s == 0 else ft(f"C_nall_s{s}")
        else:
            r = ft(f"C_n{n}_s{s}")
        if r:
            runs.append(r)
    return runs


def table_c() -> str:
    rows = []
    for n in ("10", "50", "200", "all"):
        runs = c_runs(n)
        if not runs:
            continue
        rows.append([str(runs[0]["n_train_images"]), str(len(runs)),
                     pm([r["raw"]["rmse_mm"] for r in runs]), pm([r["raw"]["abs_rel"] for r in runs], 4),
                     pm([r["raw"]["d_1.01"] for r in runs], 3), pm([r["aligned"]["rmse_mm"] for r in runs]),
                     pm([r["best_val_rmse_mm"] for r in runs])])
    return table(["training images N", "seeds", "test raw RMSE mm", "test raw AbsRel", "test raw d<1.01",
                  "test aligned RMSE mm", "val RMSE mm"], rows)


def table_d(zs) -> str:
    rows = []
    best = best_cfg()
    ind = ft(f"B_{best}")
    dr = [r for r in (ft(f"D_seen_s{s}") for s in (0, 1, 2)) if r]

    def line(label, res_list, modes=("raw", "aligned")):
        cells = [label]
        for grp in (SEEN, HELDOUT):
            for m in modes:
                vals = [subset_mean(r, m, grp, "rmse_mm") for r in res_list]
                cells.append(pm([v for v in vals if v is not None]) if any(v is not None for v in vals) else "n/a")
        return cells

    if dr:
        rows.append(line(f"fine-tuned on 7 seen categories ({len(dr)} seeds)", dr))
    if ind:
        rows.append(line("fine-tuned on all 10 categories (in-domain reference, seed 0)", [ind]))
    for v in ("baseline_train_mean_map", "dav2_small_rel", "moge2", "moge2_gtfocal", "depth_pro", "depth_pro_gtfocal",
              "dav2_metric_indoor_large"):
        if v in zs:
            rows.append(line(f"zero-shot / reference: {LABEL[v]}", [zs[v]]))
    return table(["model", "seen raw RMSE mm", "seen aligned RMSE mm", "held-out raw RMSE mm",
                  "held-out aligned RMSE mm"], rows)


def table_d_cat() -> str:
    best = best_cfg()
    ind = ft(f"B_{best}")
    dr = [r for r in (ft(f"D_seen_s{s}") for s in (0, 1, 2)) if r]
    rows = []
    for c in HELDOUT:
        rows.append([c, pm([subset_mean(r, "raw", [c], "rmse_mm") for r in dr]),
                     f(subset_mean(ind, "raw", [c], "rmse_mm")) if ind else "n/a",
                     pm([subset_mean(r, "aligned", [c], "rmse_mm") for r in dr]),
                     f(subset_mean(ind, "aligned", [c], "rmse_mm")) if ind else "n/a"])
    return table(["held-out category", "seen-only model raw RMSE mm", "in-domain model raw RMSE mm",
                  "seen-only aligned RMSE mm", "in-domain aligned RMSE mm"], rows)


# ---------------------------------------------------------------- E
def table_e() -> tuple[str, str, str]:
    q = load("quality.json")
    if not q:
        return ("not run",) * 3
    edges = q["height_edges_mm"]
    bins = [f"{edges[i]}-{edges[i + 1]} mm" if i < len(edges) - 2 else f">= {edges[i]} mm" for i in range(len(edges) - 1)]
    rows_h, rows_e, rows_d = [], [], []
    for v, res in q["variants"].items():
        name = LABEL.get(v, v)
        for m in ("raw", "aligned"):
            if m not in res:
                continue
            x = res[m]
            rows_h.append([name, m] + [f(a, 2) for a in x["height_rmse_mm"]])
            rows_e.append([name, m, f(x["flat_rmse_mm_mean"], 2), f(x["edge_rmse_mm_mean"], 2), str(x["n_edge_images"])])
            rows_d.append([name, m, f(x["defect_mae_mm_mean"], 2), f(x["ring_mae_mm_mean"], 2),
                           f(x["frac_images_defect_err_gt_ring"], 2), f(x["defect_relief_corr_median"], 2),
                           f(x["defect_relief_amplitude_ratio_median"], 2), str(x["n_defect_images_used"])])
    fr = next(iter(q["variants"].values()))
    fr = fr.get("raw", fr.get("aligned"))["height_frac_pixels"]
    rows_h.append(["share of valid pixels", ""] + [f(a * 100, 1) + " %" for a in fr])
    g = q["gt_defect_relief_mm"]
    th = table(["model", "mode"] + [f"RMSE mm, height {b}" for b in bins], rows_h)
    te = table(["model", "mode", "RMSE mm away from edges", "RMSE mm at depth edges", "images"], rows_e)
    td = table(["model", "mode", "MAE mm inside defect", "MAE mm in ring around defect", "share of images defect > ring",
                "median corr. of local relief (pred vs GT) in defect", "median relief amplitude ratio pred / GT",
                "images"], rows_d)
    td += (f"\n\nGround-truth local relief inside the defect masks (mean |depth - 15 px median filter|): median "
           f"{f(g['median'], 2)} mm, p10 {f(g['p10'], 2)} mm, p90 {f(g['p90'], 2)} mm over {g['n']} defective test images.")
    return th, te, td


# ---------------------------------------------------------------- F
def table_f() -> str:
    r = load("export_onnx.json")
    if not r:
        return "not run"
    lat = r["latency"]
    rows = [[k.replace("_", " "), f(v["median_ms"], 2), f(v["p90_ms"], 2), str(v["n"])] for k, v in lat.items()]
    t = table(["runtime (NVIDIA L4, batch 1, 518x518)", "median ms", "p90 ms", "runs"], rows)
    t += "\n\n" + table(["check", "value"], [
        ["exported model", r["variant"]],
        ["opset / input", f"{r['opset']} / static {'x'.join(map(str, r['input_shape']))}"],
        ["ONNX size MB", f(r["onnx_mb"], 1)],
        ["ONNX Runtime", f"{r['ort_version']}, providers {', '.join(r['ort_providers'])}"],
        [f"max abs diff ONNX Runtime vs PyTorch fp32 ({r['parity_images']} test images)",
         f(r["max_abs_diff_onnx_vs_torch_fp32_mm"], 4) + " mm"],
        [f"max abs diff PyTorch fp16 vs fp32 ({r['parity_images']} test images)",
         f(r["max_abs_diff_torch_fp16_vs_fp32_mm"], 4) + " mm"],
    ])
    return t


# ---------------------------------------------------------------- runtimes
def table_runtime() -> str:
    rows = []
    for p in sorted(R.glob("runtime_*.json")):
        d = json.loads(p.read_text())
        rows.append([d["job"], f(d["seconds"] / 60, 1)])
    return table(["job", "wall minutes (NVIDIA L4 queue, one job at a time)"], rows) + \
        f"\n\nTotal queue wall time: {f(sum(json.loads(p.read_text())['seconds'] for p in R.glob('runtime_*.json')) / 3600, 2)} h."


def headline(zs) -> str:
    best = best_cfg()
    b = ft(f"B_{best}") if best else None
    lines = []
    if b:
        runs = c_runs("all")
        lines.append(f"- Fine-tuned DA-V2 Small ({best}, all training images, {len(runs)} seeds): test raw RMSE "
                     f"{pm([r['raw']['rmse_mm'] for r in runs])} mm, raw AbsRel {pm([r['raw']['abs_rel'] for r in runs], 4)}.")
    if "baseline_train_mean_map" in zs:
        r = zs["baseline_train_mean_map"]["raw"]
        lines.append(f"- Per-category mean training depth map (no image input): raw RMSE {f(r['rmse_mm'])} mm.")
    met = [v for v in zs if zs[v].get("raw") and zs[v]["kind"] == "metric"]
    if met:
        v = min(met, key=lambda k: zs[k]["raw"]["rmse_mm"])
        lines.append(f"- Best zero-shot metric model on raw depth: {LABEL[v]}, raw RMSE {f(zs[v]['raw']['rmse_mm'], 1)} mm.")
    al = [v for v in zs if zs[v].get("kind") != "baseline"]
    if al:
        v = min(al, key=lambda k: zs[k]["aligned"]["rmse_mm"])
        lines.append(f"- Best zero-shot model after per-image scale-and-shift alignment: {LABEL[v]}, aligned RMSE "
                     f"{f(zs[v]['aligned']['rmse_mm'])} mm (constant-map floor: "
                     f"{f(zs['baseline_constant']['aligned']['rmse_mm'])} mm).")
    return "\n".join(lines)


def replace_block(path: Path, tag: str, content: str) -> None:
    if not path.exists():
        return
    s = path.read_text()
    pat = re.compile(rf"(<!-- {tag}:BEGIN[^>]*-->\n).*?(\n<!-- {tag}:END -->)", re.S)
    if not pat.search(s):
        return
    s = pat.sub(lambda m: m.group(1) + content + m.group(2), s)
    path.write_text(s)


def main() -> None:
    zs = zs_results()
    th, te, td = table_e()
    blocks = {
        "DATA": data_table(), "LICENCE": table_licence(zs), "A_MAIN": table_a_main(zs),
        "A_CAT_ALIGNED": table_a_cat(zs, "aligned"), "A_CAT_RAW": table_a_cat(zs, "raw"), "B": table_b(),
        "BEST": f"Configuration selected by validation RMSE: **{best_cfg()}**.", "C": table_c(), "D": table_d(zs),
        "D_CAT": table_d_cat(), "E_HEIGHT": th, "E_EDGE": te, "E_DEFECT": td, "F": table_f(),
        "RUNTIME": table_runtime(), "HEADLINE": headline(zs),
    }
    for p in (Path("README.md"), Path("docs/RESULTS.md"), Path("STATUS.md")):
        for tag, content in blocks.items():
            replace_block(p, tag, content)
    print("tables written")


if __name__ == "__main__":
    main()
