"""Experiments B, C, D: fine-tune DAV2 Small to metric depth and evaluate on the MVTec 3D-AD test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from idft import mvtec3d
from idft.evaluate import aggregate, eval_image, save_pred
from idft.train import TrainConfig, predict_all, stack, stratified_subset, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="full", choices=["head", "full", "lora"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=0)
    ap.add_argument("--lr-decoder", type=float, default=5e-5)
    ap.add_argument("--lr-encoder", type=float, default=5e-6)
    ap.add_argument("--split", default="all", choices=["all", "seen"], help="seen = train on the 7 seen categories")
    ap.add_argument("--experiment", default="B")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("data/proc"))
    ap.add_argument("--synth", type=int, default=0, help="number of synthetic training images to add (data/synth)")
    ap.add_argument("--synth-only", action="store_true", help="train on synthetic images only")
    a = ap.parse_args()
    cats = mvtec3d.SEEN if a.split == "seen" else mvtec3d.CATEGORIES
    cfg = TrainConfig(name=a.name, mode=a.mode, steps=a.steps, batch=a.batch, seed=a.seed, n_train=a.n_train,
                      lr_decoder=a.lr_decoder, lr_encoder=a.lr_encoder, train_categories=list(cats))
    allsamples = mvtec3d.index(a.data)
    pool = mvtec3d.select(allsamples, categories=cats, splits=["train"])
    tr = [] if a.synth_only else stratified_subset(pool, a.n_train, a.seed)
    if a.synth:
        syn = mvtec3d.select(mvtec3d.index(Path("data/synth")), splits=["train"])[: a.synth]
        tr = tr + syn
    va = mvtec3d.select(allsamples, categories=cats, splits=["validation"])
    te = mvtec3d.select(allsamples, splits=["test"])
    print(f"train {len(tr)} / pool {len(pool)}, val {len(va)}, test {len(te)}", flush=True)
    model, info = train(cfg, tr, va, log=lambda m: print(m, flush=True))
    te_rgb, te_dep = stack(te)
    t = time.time()
    pred = predict_all(model, te_rgb)
    info["test_predict_s"] = time.time() - t
    rows = []
    for s, p, g in zip(te, pred, te_dep.numpy()):
        r = eval_image(p, g, "metric")
        r["key"] = s.key
        rows.append(r)
        if a.save_preds:
            save_pred(Path("data/preds"), a.name, s.key, p)
    res = {"experiment": a.experiment, "variant": a.name, "train_split": a.split, "n_synth": a.synth,
           "synth_only": a.synth_only,
           "train_keys": [s.key for s in tr] if len(tr) <= 200 else f"{len(tr)} images (all of {a.split})",
           **info, **aggregate(rows), "per_image": rows}
    if a.split == "seen":
        for grp, cs in (("seen", mvtec3d.SEEN), ("heldout", mvtec3d.HELDOUT)):
            sub = [r for r in rows if r["key"].split("/")[0] in cs]
            res[f"test_{grp}"] = {k: v for k, v in aggregate(sub).items() if not k.endswith("per_category")}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/ft_{a.name}.json").write_text(json.dumps(res, indent=1))
    if a.save_ckpt:
        Path("data/ckpt").mkdir(parents=True, exist_ok=True)
        torch.save({"config": info["config"], "state_dict": model.state_dict()}, f"data/ckpt/{a.name}.pt")
    print(a.name, json.dumps({m: {k: round(v, 4) for k, v in res[m].items()} for m in ("raw", "aligned")}))
    print("val best", info["best_val_rmse_mm"], "at", info["best_step"], "train_s", round(info["train_s"]),
          "mean test rmse", float(np.mean([r["raw"]["rmse_mm"] for r in rows])))


if __name__ == "__main__":
    main()
