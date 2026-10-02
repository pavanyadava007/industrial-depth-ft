"""Optional synthetic track: render close-range top-down scenes of simple machined-looking parts (metallic and
textureless painted materials, random lights) with exact depth, in the same npz format as the real data.

Camera model matches the range seen in MVTec 3D-AD (focal 1200 to 2750 px at 400 x 400, off-centre principal
point, 0.5 to 0.72 m working distance). Rendering: pyrender with headless EGL on the NVIDIA driver.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

RES = 400


def gear(rng, r):
    n = int(rng.integers(8, 20))
    pts = []
    for i in range(n * 2):
        a = np.pi * i / n
        rr = r if i % 2 == 0 else r * rng.uniform(0.75, 0.9)
        pts.append((rr * np.cos(a), rr * np.sin(a)))
    poly = Polygon(pts)
    if rng.random() < 0.6:
        hole = Polygon([(r * 0.3 * np.cos(a), r * 0.3 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24)[:-1]])
        poly = poly.difference(hole)
    return trimesh.creation.extrude_polygon(poly, rng.uniform(0.003, 0.012))


def random_part(rng, scale):
    k = rng.integers(0, 7)
    r = scale * rng.uniform(0.1, 0.3)
    if k == 0:
        m = trimesh.creation.cylinder(radius=r, height=rng.uniform(0.004, 0.03), sections=64)
    elif k == 1:
        m = trimesh.creation.annulus(r_min=r * rng.uniform(0.3, 0.7), r_max=r, height=rng.uniform(0.004, 0.02),
                                     sections=64)
    elif k == 2:
        m = trimesh.creation.box(extents=(r * 2, r * rng.uniform(0.6, 2), rng.uniform(0.003, 0.02)))
    elif k == 3:
        m = trimesh.creation.torus(major_radius=r, minor_radius=r * rng.uniform(0.15, 0.35))
    elif k == 4:
        m = gear(rng, r)
    elif k == 5:
        m = trimesh.creation.capsule(height=r * 2, radius=r * rng.uniform(0.2, 0.4))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    else:
        m = trimesh.creation.icosphere(subdivisions=3, radius=r * 0.6)
        m.apply_scale([1, 1, rng.uniform(0.3, 1.0)])
    # rest the part on z = 0 (top of the plate)
    m.apply_transform(trimesh.transformations.rotation_matrix(rng.uniform(0, 2 * np.pi), [0, 0, 1]))
    m.apply_translation([0, 0, -m.bounds[0, 2]])
    return m


def material(rng):
    if rng.random() < 0.55:  # metal
        base = np.array([rng.uniform(0.5, 0.95)] * 3) * (np.array([1.0, 0.85, 0.55]) if rng.random() < 0.2 else 1.0)
        return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*base, 1.0], metallicFactor=rng.uniform(0.7, 1.0),
                                                  roughnessFactor=rng.uniform(0.08, 0.5))
    c = rng.uniform(0.05, 0.9, 3)
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*c, 1.0], metallicFactor=0.0,
                                              roughnessFactor=rng.uniform(0.4, 1.0))


def render_one(rng, renderer):
    fx = rng.uniform(1200, 2750)
    cx, cy = rng.uniform(50, 300, 2)
    dist = rng.uniform(0.5, 0.72)
    half_w = 0.5 * RES / fx * dist  # half field width at the plate
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=np.full(3, rng.uniform(0.02, 0.2)))
    plate = trimesh.creation.box(extents=(1.0, 1.0, 0.01))
    plate.apply_translation([0, 0, -0.005])
    world = np.eye(4)
    world[:3, :3] = trimesh.transformations.euler_matrix(*rng.uniform(-0.08, 0.08, 2), 0)[:3, :3]
    world[2, 3] = -dist
    scene.add(pyrender.Mesh.from_trimesh(plate, material=material(rng) if rng.random() < 0.3 else
                                         pyrender.MetallicRoughnessMaterial(
                                             baseColorFactor=[*rng.uniform(0.02, 0.6, 3), 1.0], metallicFactor=0.0,
                                             roughnessFactor=rng.uniform(0.5, 1.0))), pose=world)
    # field centre in camera coordinates (principal point is off-centre)
    ux, uy = (RES / 2 - cx) / fx * dist, (RES / 2 - cy) / fx * dist
    for _ in range(int(rng.integers(1, 4))):
        part = random_part(rng, 2 * half_w)
        pose = world.copy()
        off = np.array([ux, -uy, 0]) + np.r_[rng.uniform(-0.6, 0.6, 2) * half_w, 0]
        pose[:3, 3] = world[:3, 3] + world[:3, :3] @ off
        scene.add(pyrender.Mesh.from_trimesh(part, material=material(rng)), pose=pose)
    for _ in range(int(rng.integers(1, 4))):
        lp = np.eye(4)
        lp[:3, 3] = [*rng.uniform(-0.3, 0.3, 2), rng.uniform(-0.2, 0.2)]
        if rng.random() < 0.5:
            scene.add(pyrender.PointLight(color=np.ones(3), intensity=rng.uniform(0.05, 0.8)), pose=lp)
        else:
            lp[:3, :3] = trimesh.transformations.euler_matrix(*rng.uniform(-0.5, 0.5, 2), 0)[:3, :3]
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=rng.uniform(0.5, 3.0)), pose=lp)
    # pyrender cameras look along -z with +y up; image row v grows downwards, so flip y via cy
    cam = pyrender.IntrinsicsCamera(fx, fx, cx, cy, znear=0.05, zfar=2.0)
    scene.add(cam, pose=np.eye(4))
    color, depth = renderer.render(scene)
    return color, depth.astype(np.float32), np.array([fx, fx, cx, cy], np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/synth"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    renderer = pyrender.OffscreenRenderer(RES, RES)
    t0 = time.time()
    stats = []
    for split, n in (("train", a.n_train), ("validation", a.n_val)):
        d = a.out / "synthetic" / split / "good"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            rgb, depth, K = render_one(rng, renderer)
            v = depth > 0
            stats.append([float(v.mean()), float(depth[v].min()), float(depth[v].max())])
            np.savez_compressed(d / f"{i:05d}.npz", rgb=rgb, depth=depth, defect=np.zeros_like(v), K=K,
                                orig_size=np.array([RES, RES]), fit_resid=np.float32(0))
    s = np.array(stats)
    summ = {"n_train": a.n_train, "n_val": a.n_val, "seconds": time.time() - t0,
            "valid_frac_mean": float(s[:, 0].mean()),
            "depth_min_median_m": float(np.median(s[:, 1])), "depth_max_median_m": float(np.median(s[:, 2]))}
    (a.out / "synthetic" / "prep_summary.json").write_text(json.dumps(summ, indent=1))
    Path("results/synth_render.json").write_text(json.dumps(summ, indent=1))
    print(summ)


if __name__ == "__main__":
    main()
