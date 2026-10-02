"""Zero-shot monocular depth models behind one interface.

Wrappers for Depth Pro and MoGe-2 follow depth3d-lab (same author): one forward pass per image, and when the
true focal length is known a second variant re-uses the same network output with that focal. Every ``predict``
returns {variant: (H, W) float32} at the input resolution; ``KIND[variant]`` says whether the map is metric depth
in metres or affine-invariant disparity.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

DEPTH_PRO_REPO = "apple/DepthPro"
MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"
HF_DAV2 = {
    "dav2_metric_indoor_large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "dav2_metric_indoor_small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "dav2_metric_outdoor_large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "dav2_small_rel": "depth-anything/Depth-Anything-V2-Small-hf",
}

# variant -> "metric" (depth in metres) or "disparity" (affine-invariant inverse depth)
KIND = {
    "depth_pro": "metric",
    "depth_pro_gtfocal": "metric",
    "moge2": "metric",
    "moge2_gtfocal": "metric",
    "dav2_metric_indoor_large": "metric",
    "dav2_metric_indoor_small": "metric",
    "dav2_metric_outdoor_large": "metric",
    "dav2_small_rel": "disparity",
}

# Licence of the released weights as stated by the upstream repositories (checked 2026-10-02).
LICENCE = {
    "depth_pro": ("Apple ML Research Model licence (apple-amlr), research use", "no"),
    "moge2": ("MIT (HF model card Ruicheng/moge-2-vitl-normal)", "yes"),
    "dav2_metric_indoor_large": ("CC-BY-NC-4.0 (upstream README: Base/Large/Giant); HF card has no tag", "no"),
    "dav2_metric_outdoor_large": (
        "CC-BY-NC-4.0 per upstream README; HF card says apache-2.0 (conflict)", "unclear, treat as no"
    ),
    "dav2_metric_indoor_small": ("Apache-2.0 per upstream README (Small); HF card has no tag", "yes (verify)"),
    "dav2_small_rel": ("Apache-2.0 (HF card and upstream README)", "yes"),
}


class MonoDepthModel:
    name = "base"
    variants: tuple[str, ...] = ()

    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        raise NotImplementedError


class DepthProModel(MonoDepthModel):
    name = "depth_pro"
    variants = ("depth_pro", "depth_pro_gtfocal")

    def __init__(self, device: str = "cuda", precision: torch.dtype = torch.half):
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(DEPTH_PRO_REPO, "depth_pro.pt")
        config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ckpt)
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device(device), precision=precision
        )
        self.model.eval()
        self.last_focal_px: float | None = None

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        x = self.transform(image)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        size = self.model.img_size
        xr = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        canonical_inv, fov_deg = self.model.forward(xr)
        f_est = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        self.last_focal_px = float(f_est.squeeze())
        focals = {"depth_pro": f_est}
        if f_px is not None:
            focals["depth_pro_gtfocal"] = torch.tensor(float(f_px), device=canonical_inv.device)
        out = {}
        for key, f in focals.items():
            inv = canonical_inv.float() * (W / f)
            inv = torch.nn.functional.interpolate(inv, size=(H, W), mode="bilinear", align_corners=False)
            out[key] = (1.0 / torch.clamp(inv, min=1e-4, max=1e4)).squeeze().float().cpu().numpy()
        return out


class MoGe2Model(MonoDepthModel):
    """MoGe-2 ViT-L; post-processing mirrors MoGeModel.infer (no validity mask applied)."""

    name = "moge2"
    variants = ("moge2", "moge2_gtfocal")

    def __init__(self, device: str = "cuda", resolution_level: int = 9):
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(MOGE2_REPO).to(device).eval()
        self.device = device
        lo, hi = self.model.num_tokens_range
        self.num_tokens = int(lo + (resolution_level / 9) * (hi - lo))
        self.last_focal_px: float | None = None

    @torch.no_grad()
    def _depth(self, points, mask, metric_scale, aspect, fov_x):
        from moge.utils.geometry_torch import recover_focal_shift

        with torch.autocast(device_type=self.model.device.type, dtype=torch.float32):
            if fov_x is None:
                focal, shift = recover_focal_shift(points, mask)
            else:
                focal = aspect / (1 + aspect**2) ** 0.5 / torch.tan(
                    torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2)
                )
                if focal.ndim == 0:
                    focal = focal[None].expand(points.shape[0])
                _, shift = recover_focal_shift(points, mask, focal=focal)
            fx = focal / 2 * (1 + aspect**2) ** 0.5 / aspect
            depth = (points[..., 2] + shift[..., None, None]) * metric_scale[:, None, None]
        return depth, float(fx.squeeze())

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        H, W = image.shape[:2]
        aspect = W / H
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(dtype=self.model.dtype)
        use_fp16 = self.model.dtype != torch.float16
        with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=use_fp16):
            o = self.model.forward(x, num_tokens=self.num_tokens)
        points = o["points"].float()
        mask = o["mask"].float() > 0.5 if o.get("mask") is not None else None
        ms = o["metric_scale"].float()
        depth, fx = self._depth(points, mask, ms, aspect, None)
        self.last_focal_px = fx * W
        out = {"moge2": depth.squeeze(0).float().cpu().numpy()}
        if f_px is not None:
            fov_x = float(np.degrees(2 * np.arctan(0.5 * W / f_px)))
            d2, _ = self._depth(points, mask, ms, aspect, fov_x)
            out["moge2_gtfocal"] = d2.squeeze(0).float().cpu().numpy()
        return out


class DAV2HF(MonoDepthModel):
    """Depth Anything V2 through transformers (metric or relative checkpoints)."""

    def __init__(self, name: str, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.name = name
        self.variants = (name,)
        self.processor = AutoImageProcessor.from_pretrained(HF_DAV2[name])
        self.model = AutoModelForDepthEstimation.from_pretrained(HF_DAV2[name]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image: np.ndarray, f_px: float | None) -> dict[str, np.ndarray]:
        H, W = image.shape[:2]
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pred = self.model(**inputs).predicted_depth  # (1, h, w)
        pred = torch.nn.functional.interpolate(pred[:, None], size=(H, W), mode="bilinear", align_corners=False)
        return {self.name: pred[0, 0].float().cpu().numpy()}


def load_model(name: str, device: str = "cuda") -> MonoDepthModel:
    if name == "depth_pro":
        return DepthProModel(device=device)
    if name == "moge2":
        return MoGe2Model(device=device)
    return DAV2HF(name, device=device)
