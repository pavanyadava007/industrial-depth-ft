"""Experiment F: export the selected fine-tuned model to ONNX (opset 17, static 1x3x518x518), check numerical
parity against PyTorch and measure latency (PyTorch fp32 / fp16, ONNX Runtime CUDA EP fp32) on the NVIDIA L4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from idft import mvtec3d
from idft.train import TrainConfig, build_model, preprocess


class Wrapper(torch.nn.Module):
    """Normalised (1, 3, 518, 518) input -> metric depth (1, 518, 518) in metres."""

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(pixel_values=x).predicted_depth


def load_merged(cfg: TrainConfig, state: dict) -> torch.nn.Module:
    """Build the model, load the fine-tuned weights and, for LoRA, merge the adapters into the base weights."""
    model = build_model(cfg)
    model.load_state_dict(state)
    if cfg.mode == "lora":
        from peft.tuners.lora import LoraLayer

        for mod in model.modules():
            if isinstance(mod, LoraLayer):
                mod.merge()
    return model


def bench(fn, x, n: int = 100, warm: int = 10) -> dict:
    for _ in range(warm):
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, default=Path("data/onnx/dav2s_metric_ft.onnx"))
    ap.add_argument("--n-parity", type=int, default=20)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["config"])
    model = load_merged(cfg, ck["state_dict"])
    w = Wrapper(model).eval().cuda()
    a.onnx.parent.mkdir(parents=True, exist_ok=True)
    x0 = torch.randn(1, 3, 518, 518, device="cuda")
    t = time.time()
    torch.onnx.export(w, (x0,), str(a.onnx), opset_version=17, input_names=["image"], output_names=["depth"],
                      dynamo=False)
    export_s = time.time() - t
    import onnx

    onnx.checker.check_model(str(a.onnx))
    import onnxruntime as ort

    sess = ort.InferenceSession(str(a.onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = sess.get_providers()
    test = mvtec3d.select(mvtec3d.index(Path("data/proc")), splits=["test"])
    test = test[:: max(1, len(test) // a.n_parity)][: a.n_parity]
    diffs, diffs16 = [], []
    w16 = Wrapper(load_merged(cfg, ck["state_dict"])).eval().cuda().half()
    with torch.no_grad():
        for s in test:
            rgb = torch.from_numpy(mvtec3d.load(s)["rgb"])[None].cuda()
            x = preprocess(rgb)
            ref = w(x).float().cpu().numpy()
            o = sess.run(None, {"image": x.cpu().numpy()})[0]
            diffs.append(float(np.abs(o - ref).max()))
            diffs16.append(float(np.abs(w16(x.half()).float().cpu().numpy() - ref).max()))
        xb = preprocess(torch.from_numpy(mvtec3d.load(test[0])["rgb"])[None].cuda())
        lat = {
            "torch_fp32": bench(lambda x: w(x), xb),
            "torch_fp16": bench(lambda x: w16(x), xb.half()),
        }
        io = sess.io_binding()
        xb_np = xb.cpu().numpy()

        def run_ort(_):
            io.bind_cpu_input("image", xb_np)
            io.bind_output("depth")
            sess.run_with_iobinding(io)
            io.copy_outputs_to_cpu()

        lat["onnxruntime_fp32"] = bench(run_ort, None)
    res = {
        "experiment": "F_export", "ckpt": str(a.ckpt), "variant": cfg.name, "opset": 17,
        "input_shape": [1, 3, 518, 518], "onnx_mb": a.onnx.stat().st_size / 1e6, "export_s": export_s,
        "ort_version": ort.__version__, "ort_providers": providers, "torch_version": torch.__version__,
        "parity_images": len(test),
        "max_abs_diff_onnx_vs_torch_fp32_mm": float(np.max(diffs) * 1000),
        "max_abs_diff_torch_fp16_vs_fp32_mm": float(np.max(diffs16) * 1000),
        "latency": lat, "gpu": torch.cuda.get_device_name(0),
        "latency_note": "batch 1, 518x518 network input, forward only (no resize to 400x400); ORT timing includes "
                        "host-to-device copy of the input and copy of the output back to the host",
    }
    Path("results/export_onnx.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
