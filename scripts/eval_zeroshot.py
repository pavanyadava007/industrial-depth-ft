"""Experiment A: zero-shot monocular depth on the MVTec 3D-AD test split (all categories)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from idft import mvtec3d
from idft.evaluate import aggregate, eval_image, save_pred
from idft.models import KIND, LICENCE, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/proc"))
    ap.add_argument("--preds", type=Path, default=Path("data/preds"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    samples = mvtec3d.select(mvtec3d.index(a.data), splits=[a.split])
    if a.limit:
        samples = samples[:: max(1, len(samples) // a.limit)][: a.limit]
    t_load = time.time()
    model = load_model(a.model)
    t_load = time.time() - t_load
    rows: dict[str, list] = {v: [] for v in model.variants}
    times = []
    t0 = time.time()
    for i, s in enumerate(samples):
        d = mvtec3d.load(s)
        if i == 0:  # warm-up, not timed
            model.predict(d["rgb"], float(d["K"][0]))
        torch.cuda.synchronize()
        t = time.perf_counter()
        preds = model.predict(d["rgb"], float(d["K"][0]))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000.0)
        for v, p in preds.items():
            r = eval_image(p, d["depth"], KIND[v])
            r["key"] = s.key
            rows[v].append(r)
            save_pred(a.preds, v, s.key, p)
        if i % 100 == 0:
            print(f"{i}/{len(samples)} {times[-1]:.1f} ms", flush=True)
    wall = time.time() - t0
    a.out.mkdir(parents=True, exist_ok=True)
    for v, rs in rows.items():
        res = {
            "experiment": "A_zero_shot", "variant": v, "model": a.model, "kind": KIND[v],
            "licence": LICENCE.get(a.model, LICENCE.get(v, ("?", "?")))[0],
            "commercial_use": LICENCE.get(a.model, LICENCE.get(v, ("?", "?")))[1],
            "split": a.split, "gpu": torch.cuda.get_device_name(0),
            "latency_ms_median": float(np.median(times)), "latency_ms_mean": float(np.mean(times)),
            "latency_note": "one forward (+ post-processing for all variants of this model), batch 1, 400x400 input, "
                            "after one warm-up image",
            "load_s": t_load, "wall_s": wall, **aggregate(rs), "per_image": rs,
        }
        (a.out / f"zeroshot_{v}.json").write_text(json.dumps(res, indent=1))
        msg = {m: {k: round(x, 4) for k, x in res[m].items()} for m in ("raw", "aligned") if m in res}
        print(v, json.dumps(msg))


if __name__ == "__main__":
    main()
