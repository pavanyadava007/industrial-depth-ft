"""MVTec 3D-AD preprocessing and loading.

The raw dataset stores per sample an RGB PNG, an organised XYZ point map (TIFF, float32, metres, zeros for
missing structured-light returns) and, for defective test samples, a binary defect mask. We keep the Z channel
as metric depth ground truth, fit the pinhole intrinsics from the point map (X/Z and Y/Z are linear in the pixel
coordinates), and write one compressed npz per sample at a fixed working resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
]
# Fixed generalisation split (Experiment D): 7 seen categories, 3 held out. Chosen before any training so that
# the held-out set mixes a rigid machined part (cable_gland), a food item (peach) and a deformable part (rope).
HELDOUT = ["cable_gland", "peach", "rope"]
SEEN = [c for c in CATEGORIES if c not in HELDOUT]

WORK_RES = 400  # every sample is stored at WORK_RES x WORK_RES


def fit_intrinsics(xyz: np.ndarray) -> dict[str, float]:
    """Least-squares fit of fx, fy, cx, cy from an organised point map (H, W, 3) with Z > 0 where valid."""
    z = xyz[..., 2]
    valid = z > 0
    if valid.sum() < 100:
        raise ValueError("too few valid points to fit intrinsics")
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xr = xyz[..., 0][valid] / z[valid]
    yr = xyz[..., 1][valid] / z[valid]
    px = np.polyfit(u[valid], xr, 1)
    py = np.polyfit(v[valid], yr, 1)
    fx, fy = 1.0 / px[0], 1.0 / py[0]
    resid = float(np.sqrt(np.mean((xr - np.polyval(px, u[valid])) ** 2)))
    return {"fx": float(fx), "fy": float(fy), "cx": float(-px[1] * fx), "cy": float(-py[1] * fy), "resid": resid}


def downsample_depth(z: np.ndarray, out: int) -> np.ndarray:
    """Valid-aware area resampling of a square depth map with zeros as invalid.

    An output pixel is kept only if every input pixel it covers is valid, so that depth edges are not blended
    with holes; for integer factors this is exactly the block mean over fully valid blocks.
    """
    import cv2

    H, W = z.shape
    if H == out and W == out:
        return z.astype(np.float32)
    valid = (z > 0).astype(np.float32)
    zs = cv2.resize(np.where(z > 0, z, 0).astype(np.float64), (out, out), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid.astype(np.float64), (out, out), interpolation=cv2.INTER_AREA)
    return np.where(vs > 1 - 1e-6, zs / np.maximum(vs, 1e-12), 0).astype(np.float32)


def center_square(a: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Centre crop to a square; returns the crop and the (row, column) offset."""
    H, W = a.shape[:2]
    s = min(H, W)
    r0, c0 = (H - s) // 2, (W - s) // 2
    return a[r0:r0 + s, c0:c0 + s], r0, c0


def preprocess_sample(rgb: np.ndarray, xyz: np.ndarray, mask: np.ndarray | None, res: int = WORK_RES) -> dict:
    import cv2

    orig = np.array(xyz.shape[:2])
    xyz, _, _ = center_square(xyz)  # some categories are not square (e.g. 800x600); crop the centre square
    rgb, _, _ = center_square(rgb)
    if mask is not None:
        mask, _, _ = center_square(mask)
    H, W = xyz.shape[:2]
    K = fit_intrinsics(xyz)
    s = res / W
    z = np.nan_to_num(xyz[..., 2].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    z[z < 0] = 0
    z = downsample_depth(z, res)
    if rgb.shape[:2] != (res, res):
        rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
    m = np.zeros((res, res), dtype=bool)
    if mask is not None:
        m = cv2.resize((mask > 0).astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0
    # pixel-centre convention: scaling a pinhole by s maps c -> (c + 0.5) * s - 0.5
    intr = np.array([K["fx"] * s, K["fy"] * s, (K["cx"] + 0.5) * s - 0.5, (K["cy"] + 0.5) * s - 0.5], np.float32)
    return {"rgb": rgb.astype(np.uint8), "depth": z, "defect": m, "K": intr, "orig_size": orig,
            "fit_resid": np.float32(K["resid"])}


@dataclass
class Sample:
    path: Path
    category: str
    split: str  # train / validation / test
    defect_type: str  # good or defect name

    @property
    def key(self) -> str:
        return f"{self.category}/{self.split}/{self.defect_type}/{self.path.stem}"


def index(root: Path) -> list[Sample]:
    out = []
    for p in sorted(Path(root).glob("*/*/*/*.npz")):
        cat, split, dtype = p.parts[-4], p.parts[-3], p.parts[-2]
        out.append(Sample(p, cat, split, dtype))
    return out


def load(sample: Sample | Path) -> dict[str, np.ndarray]:
    p = sample.path if isinstance(sample, Sample) else Path(sample)
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def select(samples: list[Sample], categories=None, splits=None) -> list[Sample]:
    return [s for s in samples if (categories is None or s.category in categories)
            and (splits is None or s.split in splits)]


def subset_file(root: Path) -> Path:
    return Path(root) / "subsets.json"


def read_subsets(root: Path) -> dict:
    return json.loads(subset_file(root).read_text())
