"""Fine-tuning Depth Anything V2 Small (Apache-2.0) to metric depth on close-range industrial data.

Model: transformers DepthAnythingForDepthEstimation initialised from depth-anything/Depth-Anything-V2-Small-hf
(relative depth checkpoint), switched to the metric head (sigmoid * max_depth, max_depth = 1 m). The last 1x1
conv of the head (32 -> 1) is re-initialised (small weights, bias = logit(init_depth / max_depth)) because the
relative checkpoint's output layer regresses unbounded disparity and would saturate the sigmoid.

Loss (documented in the README): SILog on log depth with lambda = 0.85 (as in the Depth Anything V2 metric
fine-tuning recipe) plus 0.5 x a 4-scale gradient-matching term on the log-depth residual (MegaDepth / MiDaS
style), both over valid ground-truth pixels only.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from idft import mvtec3d

DAV2_SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
NET_RES = 518


@dataclass
class TrainConfig:
    name: str
    mode: str = "full"  # head | full | lora
    lr_decoder: float = 5e-5
    lr_encoder: float = 5e-6
    lora_r: int = 16
    lora_alpha: int = 16
    steps: int = 2000
    batch: int = 8
    eval_every: int = 250
    seed: int = 0
    n_train: int = 0  # 0 = all
    train_categories: list[str] = field(default_factory=lambda: list(mvtec3d.CATEGORIES))
    test_categories: list[str] = field(default_factory=lambda: list(mvtec3d.CATEGORIES))
    max_depth: int = 1  # metres; transformers 5 requires an int here
    init_depth: float = 0.65
    silog_lambda: float = 0.85
    grad_weight: float = 0.5
    weight_decay: float = 0.01
    init: str = DAV2_SMALL


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModelForDepthEstimation

    conf = AutoConfig.from_pretrained(cfg.init)
    conf.depth_estimation_type = "metric"
    conf.max_depth = cfg.max_depth
    model = AutoModelForDepthEstimation.from_pretrained(cfg.init, config=conf)
    conv3 = model.head.conv3
    torch.nn.init.normal_(conv3.weight, std=1e-3)
    p = cfg.init_depth / cfg.max_depth
    torch.nn.init.constant_(conv3.bias, math.log(p / (1 - p)))
    if cfg.mode == "head":
        for prm in model.backbone.parameters():
            prm.requires_grad = False
    elif cfg.mode == "lora":
        from peft import LoraConfig, inject_adapter_in_model

        for prm in model.backbone.parameters():
            prm.requires_grad = False
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model.backbone = inject_adapter_in_model(lcfg, model.backbone)
    elif cfg.mode != "full":
        raise ValueError(cfg.mode)
    fp32_head(model)
    return model


def fp32_head(model: torch.nn.Module) -> None:
    """Run the DPT head (3 convs + sigmoid) in fp32 even under bf16 autocast.

    The metric output is sigmoid * 1 m; in bf16 (8 significant bits) values in [0.5, 1) m are quantised to
    2^-8 m = 3.9 mm steps, which is larger than the errors being measured. The first runs of this project had
    exactly this problem (see STATUS.md); the backbone and neck still run in bf16.
    """
    head = model.head
    orig = head.forward

    def forward(hidden_states, patch_height, patch_width):
        with torch.autocast("cuda", enabled=False):
            dt = head.conv1.weight.dtype  # fp32 when training under autocast, fp16 for a .half() model
            return orig([h.to(dt) for h in hidden_states], patch_height, patch_width)

    head.forward = forward


def param_groups(model: torch.nn.Module, cfg: TrainConfig) -> list[dict]:
    enc, dec = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith("backbone.") else dec).append(p)
    groups = [{"params": dec, "lr": cfg.lr_decoder, "name": "decoder"}]
    if enc:
        # LoRA adapters are new parameters, so they use the decoder learning rate; the pretrained encoder (full FT)
        # uses the lower encoder learning rate
        groups.append({"params": enc, "lr": cfg.lr_decoder if cfg.mode == "lora" else cfg.lr_encoder,
                       "name": "encoder"})
    return groups


def count_params(model: torch.nn.Module) -> dict[str, int]:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def preprocess(rgb_u8: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, 518, 518) normalised float."""
    x = rgb_u8.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(NET_RES, NET_RES), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def forward_depth(model: torch.nn.Module, rgb_u8: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    x = preprocess(rgb_u8)
    d = model(pixel_values=x).predicted_depth  # (B, 518, 518)
    return F.interpolate(d[:, None], size=out_hw, mode="bilinear", align_corners=False)[:, 0]


def silog_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    d = torch.log(pred.clamp_min(1e-4)) - torch.log(gt.clamp_min(1e-4))
    d = d[valid]
    if d.numel() == 0:
        return pred.sum() * 0
    return torch.sqrt((d**2).mean() - lam * d.mean() ** 2 + 1e-8)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, scales: int = 4) -> torch.Tensor:
    r = torch.log(pred.clamp_min(1e-4)) - torch.log(gt.clamp_min(1e-4))
    r = torch.where(valid, r, torch.zeros_like(r))
    m = valid.float()
    total = pred.new_zeros(())
    for s in range(scales):
        step = 2**s
        rr, mm = r[:, ::step, ::step], m[:, ::step, ::step]
        gx = (rr[:, :, 1:] - rr[:, :, :-1]).abs() * mm[:, :, 1:] * mm[:, :, :-1]
        gy = (rr[:, 1:, :] - rr[:, :-1, :]).abs() * mm[:, 1:, :] * mm[:, :-1, :]
        n = (mm[:, :, 1:] * mm[:, :, :-1]).sum() + (mm[:, 1:, :] * mm[:, :-1, :]).sum()
        total = total + (gx.sum() + gy.sum()) / n.clamp_min(1.0)
    return total / scales


def total_loss(pred, gt, valid, cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    ls = silog_loss(pred, gt, valid, cfg.silog_lambda)
    lg = gradient_loss(pred, gt, valid)
    return ls + cfg.grad_weight * lg, {"silog": float(ls), "grad": float(lg)}


def stratified_subset(samples: list[mvtec3d.Sample], n: int, seed: int) -> list[mvtec3d.Sample]:
    """n samples drawn round-robin over categories (each category shuffled with the seed)."""
    if n <= 0 or n >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s)
    for v in by_cat.values():
        rng.shuffle(v)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    out, i = [], 0
    while len(out) < n:
        c = cats[i % len(cats)]
        j = i // len(cats)
        if j < len(by_cat[c]):
            out.append(by_cat[c][j])
        i += 1
    return out


def stack(samples: list[mvtec3d.Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    rgb, dep = [], []
    for s in samples:
        d = mvtec3d.load(s)
        rgb.append(d["rgb"])
        dep.append(d["depth"])
    return torch.from_numpy(np.stack(rgb)), torch.from_numpy(np.stack(dep))


def augment(rgb: torch.Tensor, dep: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Flips (Z does not depend on the pixel position, so flipped depth stays valid) and photometric jitter.
    No scale or crop augmentation: the camera is fixed, and changing the apparent focal length would make metric
    depth ambiguous."""
    B = rgb.shape[0]
    rgb = rgb.float()
    for i in range(B):
        if torch.rand(1, generator=gen) < 0.5:
            rgb[i], dep[i] = rgb[i].flip(1), dep[i].flip(1)
        if torch.rand(1, generator=gen) < 0.5:
            rgb[i], dep[i] = rgb[i].flip(0), dep[i].flip(0)
        b = 1 + (torch.rand(1, generator=gen) - 0.5) * 0.4
        c = 1 + (torch.rand(1, generator=gen) - 0.5) * 0.4
        m = rgb[i].mean()
        rgb[i] = ((rgb[i] - m) * c + m) * b
    return rgb.clamp(0, 255), dep


@torch.no_grad()
def predict_all(model, rgb: torch.Tensor, batch: int = 16) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, rgb.shape[0], batch):
        x = rgb[i:i + batch].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = forward_depth(model, x, x.shape[1:3])
        out.append(d.float().cpu().numpy())
    return np.concatenate(out)


def val_rmse_mm(model, rgb: torch.Tensor, dep: torch.Tensor) -> float:
    pred = predict_all(model, rgb)
    gt = dep.numpy()
    vals = []
    for p, g in zip(pred, gt):
        v = g > 0
        vals.append(float(np.sqrt(np.mean((p[v] - g[v]) ** 2)) * 1000))
    return float(np.mean(vals))


def train(cfg: TrainConfig, train_samples, val_samples, log=print) -> tuple[torch.nn.Module, dict]:
    set_seed(cfg.seed)
    model = build_model(cfg).cuda()
    params = count_params(model)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    warm = min(100, cfg.steps // 10)
    tr_rgb, tr_dep = stack(train_samples)
    va_rgb, va_dep = stack(val_samples)
    gen = torch.Generator().manual_seed(cfg.seed)
    history = []
    best = (float("inf"), -1)
    best_state = None
    t0 = time.time()
    n = tr_rgb.shape[0]
    perm, pi = torch.randperm(n, generator=gen), 0
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = []
        while len(idx) < cfg.batch:
            if pi >= n:
                perm, pi = torch.randperm(n, generator=gen), 0
            idx.append(int(perm[pi]))
            pi += 1
        rgb, dep = augment(tr_rgb[idx].clone(), tr_dep[idx].clone(), gen)
        rgb, dep = rgb.cuda(non_blocking=True), dep.cuda(non_blocking=True)
        valid = dep > 0
        f = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (cfg.steps - warm)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = forward_depth(model, rgb, dep.shape[1:3])
        loss, parts = total_loss(pred.float(), dep, valid, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            v = val_rmse_mm(model, va_rgb, va_dep)
            history.append({"step": step, "loss": float(loss), **parts, "val_rmse_mm": v,
                            "elapsed_s": time.time() - t0})
            log(f"[{cfg.name}] step {step} loss {float(loss):.4f} val_rmse_mm {v:.3f} ({time.time() - t0:.0f}s)")
            if v < best[0]:
                best = (v, step)
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    info = {"config": asdict(cfg), "params": params, "n_train_images": n, "n_val_images": va_rgb.shape[0],
            "best_val_rmse_mm": best[0], "best_step": best[1], "history": history, "train_s": train_s,
            "steps_per_s": cfg.steps / train_s, "gpu": torch.cuda.get_device_name(0)}
    return model, info
