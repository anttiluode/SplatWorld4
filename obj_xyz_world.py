#!/usr/bin/env python3
"""OBJ-backed 3-D scene generator for SplatWorld4.

Purpose
-------
Give the address-conditioned operator a *real* three-coordinate camera query.

    address = (camera_x, camera_y, camera_z)

The third coordinate is not an arbitrary style knob here: it is the physical
camera depth used by a perspective renderer.  A scene is made from several OBJ
meshes placed at different world depths, so changing camera z affects scale,
parallax and occlusion rather than only one global zoom.

The module intentionally has no OpenGL dependency.  It contains a small
NumPy z-buffer renderer so the experiment is reproducible in CI and on a
Windows desktop.  Training can then happen on CUDA in xyz_operator_train.py.

Downloadable presets are from Adobe's lagrange-test-data repository; the
selected files are listed as CC0 in that repository's license table.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

ASSETS = {
    "spot": {
        "url": "https://raw.githubusercontent.com/adobe/lagrange-test-data/main/core/spot/spot_triangulated.obj",
        "license": "CC0-1.0",
        "source": "https://github.com/adobe/lagrange-test-data",
    },
    "rounded_cube": {
        "url": "https://raw.githubusercontent.com/adobe/lagrange-test-data/main/core/rounded_cube.obj",
        "license": "CC0-1.0",
        "source": "https://github.com/adobe/lagrange-test-data",
    },
    "avocado": {
        "url": "https://raw.githubusercontent.com/adobe/lagrange-test-data/main/io/avocado/avocado.obj",
        "license": "CC0-1.0",
        "source": "https://github.com/adobe/lagrange-test-data",
    },
}

BASE_COLORS = np.asarray(
    [
        [0.86, 0.35, 0.25],
        [0.25, 0.68, 0.90],
        [0.42, 0.82, 0.42],
        [0.88, 0.72, 0.25],
        [0.69, 0.42, 0.88],
    ],
    np.float64,
)


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"


@dataclass
class TriangleSoup:
    vertices: np.ndarray  # (T,3,3)
    colors: np.ndarray    # (T,3)
    labels: np.ndarray    # (T,)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "SplatWorld4-OBJ-downloader/1.0"})
    with urlopen(req, timeout=60) as r:
        data = r.read()
    if len(data) < 64:
        raise RuntimeError(f"downloaded file is suspiciously small: {url}")
    path.write_bytes(data)


def ensure_asset(name: str, out_dir: str | Path = "assets/obj") -> Path:
    if name not in ASSETS:
        raise KeyError(f"unknown asset {name!r}; choices: {', '.join(sorted(ASSETS))}")
    spec = ASSETS[name]
    path = Path(out_dir) / f"{name}.obj"
    if not path.exists():
        print(f"download {name}: {spec['url']}")
        _download(spec["url"], path)
    return path


def download_assets(names: Iterable[str], out_dir: str | Path = "assets/obj") -> list[Path]:
    return [ensure_asset(n, out_dir) for n in names]


def _parse_face_token(tok: str, nverts: int) -> int:
    raw = tok.split("/", 1)[0]
    idx = int(raw)
    if idx < 0:
        idx = nverts + idx
    else:
        idx -= 1
    if idx < 0 or idx >= nverts:
        raise ValueError(f"OBJ vertex index out of range: {tok}")
    return idx


def load_obj(path: str | Path, max_faces: int | None = None) -> Mesh:
    path = Path(path)
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            fields = s.split()
            if fields[0] == "v" and len(fields) >= 4:
                verts.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and len(fields) >= 4:
                ids = [_parse_face_token(t, len(verts)) for t in fields[1:]]
                for j in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[j], ids[j + 1]))
    if not verts or not faces:
        raise ValueError(f"{path} has no usable vertices/faces")
    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int32)
    if max_faces is not None and len(F) > int(max_faces):
        ii = np.linspace(0, len(F) - 1, int(max_faces), dtype=np.int64)
        F = F[ii]
    return Mesh(V, F, path.stem)


def normalize_mesh(mesh: Mesh) -> Mesh:
    V = np.asarray(mesh.vertices, np.float64).copy()
    lo, hi = V.min(axis=0), V.max(axis=0)
    V -= 0.5 * (lo + hi)
    radius = np.max(np.linalg.norm(V, axis=1))
    if radius <= 1e-12:
        raise ValueError("degenerate mesh")
    V /= radius
    return Mesh(V, np.asarray(mesh.faces, np.int32).copy(), mesh.name)


def tiny_cube_mesh() -> Mesh:
    V = np.asarray(
        [
            [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
            [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1],
        ],
        np.float64,
    )
    F = np.asarray(
        [
            [0,1,2],[0,2,3], [4,6,5],[4,7,6],
            [0,4,5],[0,5,1], [1,5,6],[1,6,2],
            [2,6,7],[2,7,3], [3,7,4],[3,4,0],
        ],
        np.int32,
    )
    return normalize_mesh(Mesh(V, F, "cube"))


def _transform_mesh(mesh: Mesh, scale: float, offset, yaw: float = 0.0) -> np.ndarray:
    V = normalize_mesh(mesh).vertices
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float64)
    return (V @ R.T) * float(scale) + np.asarray(offset, np.float64)


def build_scene(meshes: list[Mesh]) -> TriangleSoup:
    """Place meshes at deliberately different world depths."""
    if not meshes:
        raise ValueError("need at least one mesh")
    placements = [
        (0.72, (-0.58, -0.18, +0.48), +0.25),
        (0.82, (+0.05, +0.18, -0.05), -0.35),
        (0.62, (+0.62, -0.22, -0.72), +0.55),
        (0.48, (-0.05, +0.58, -0.95), -0.65),
    ]
    tris, cols, labels = [], [], []
    for i, mesh in enumerate(meshes):
        scale, off, yaw = placements[i % len(placements)]
        V = _transform_mesh(mesh, scale, off, yaw)
        tri = V[np.asarray(mesh.faces, np.int32)]
        tris.append(tri)
        cols.append(np.tile(BASE_COLORS[i % len(BASE_COLORS)], (len(tri), 1)))
        labels.append(np.full(len(tri), i, np.int32))
    return TriangleSoup(np.concatenate(tris), np.concatenate(cols), np.concatenate(labels))


def camera_frame(position, target=(0.0, 0.0, -0.1)):
    pos = np.asarray(position, np.float64)
    tgt = np.asarray(target, np.float64)
    forward = tgt - pos
    forward /= np.linalg.norm(forward) + 1e-12
    up0 = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up0)
    if np.linalg.norm(right) < 1e-8:
        up0 = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up0)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    return right, up, forward


def _edge(a, b, p):
    return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])


def render(scene: TriangleSoup, camera_position, height: int = 64, width: int = 64,
           fov_deg: float = 48.0, near: float = 0.08, return_labels: bool = False):
    """Perspective rasterizer with a z-buffer and flat two-sided shading."""
    H, W = int(height), int(width)
    pos = np.asarray(camera_position, np.float64)
    right, up, forward = camera_frame(pos)
    tri_world = scene.vertices
    rel = tri_world - pos[None, None, :]
    xc = np.einsum("tvi,i->tv", rel, right)
    yc = np.einsum("tvi,i->tv", rel, up)
    zc = np.einsum("tvi,i->tv", rel, forward)
    focal = 0.5 * W / math.tan(math.radians(fov_deg) * 0.5)
    u = focal * xc / np.maximum(zc, 1e-9) + (W - 1) * 0.5
    v = -focal * yc / np.maximum(zc, 1e-9) + (H - 1) * 0.5
    P = np.stack([u, v], axis=-1)

    e1 = tri_world[:, 1] - tri_world[:, 0]
    e2 = tri_world[:, 2] - tri_world[:, 0]
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.asarray([0.35, 0.8, 0.45], np.float64)
    light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.abs(n @ light)

    bg_y = np.linspace(0.06, 0.12, H, dtype=np.float64)[:, None, None]
    img = np.tile(bg_y, (1, W, 3))
    depth = np.full((H, W), np.inf, np.float64)
    lab = np.full((H, W), -1, np.int32)

    for t in range(len(tri_world)):
        if np.any(zc[t] <= near):
            continue
        pt = P[t]
        xmin = max(0, int(math.floor(np.min(pt[:, 0]))))
        xmax = min(W - 1, int(math.ceil(np.max(pt[:, 0]))))
        ymin = max(0, int(math.floor(np.min(pt[:, 1]))))
        ymax = min(H - 1, int(math.ceil(np.max(pt[:, 1]))))
        if xmin > xmax or ymin > ymax:
            continue
        area = _edge(pt[0], pt[1], pt[2])
        if abs(area) < 1e-10:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float64)
        ys = np.arange(ymin, ymax + 1, dtype=np.float64)
        xx, yy = np.meshgrid(xs + 0.5, ys + 0.5)
        q = np.stack([xx, yy], axis=-1)
        w0 = _edge(pt[1], pt[2], q) / area
        w1 = _edge(pt[2], pt[0], q) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        if not np.any(inside):
            continue
        invz = w0 / zc[t, 0] + w1 / zc[t, 1] + w2 / zc[t, 2]
        zz = 1.0 / np.maximum(invz, 1e-12)
        subd = depth[ymin:ymax + 1, xmin:xmax + 1]
        take = inside & (zz < subd)
        if not np.any(take):
            continue
        subd[take] = zz[take]
        color = np.clip(scene.colors[t] * shade[t], 0.0, 1.0)
        subi = img[ymin:ymax + 1, xmin:xmax + 1]
        subi[take] = color
        subl = lab[ymin:ymax + 1, xmin:xmax + 1]
        subl[take] = scene.labels[t]

    if H > 4:
        ramp = np.linspace(1.0, 0.78, H)[:, None, None]
        mask = (lab < 0)[..., None]
        img = np.where(mask, img * ramp, img)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    if return_labels:
        return img, depth.astype(np.float32), lab
    return img


def camera_grid(nx: int = 5, ny: int = 3, nz: int = 5,
                x_span: float = 0.95, y_span: float = 0.58,
                z_near: float = 2.45, z_far: float = 4.05):
    xs = np.linspace(-x_span, x_span, int(nx))
    ys = np.linspace(-y_span, y_span, int(ny))
    zs = np.linspace(z_near, z_far, int(nz))
    poses, ijk = [], []
    for kz, z in enumerate(zs):
        for jy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                poses.append([x, y, z])
                ijk.append([ix, jy, kz])
    return np.asarray(poses, np.float64), np.asarray(ijk, np.int32), (xs, ys, zs)


def normalized_addresses(poses: np.ndarray, axes) -> np.ndarray:
    xs, ys, zs = axes
    p = np.asarray(poses, np.float64)
    out = np.empty_like(p)
    out[:, 0] = p[:, 0] / max(abs(float(xs[0])), abs(float(xs[-1])), 1e-9)
    out[:, 1] = p[:, 1] / max(abs(float(ys[0])), abs(float(ys[-1])), 1e-9)
    zmid = 0.5 * (float(zs[0]) + float(zs[-1]))
    zhalf = max(0.5 * (float(zs[-1]) - float(zs[0])), 1e-9)
    out[:, 2] = (p[:, 2] - zmid) / zhalf
    return out


def split_indices(ijk: np.ndarray):
    """Train on z planes 0,2,4; hold entire intermediate z planes 1,3 out."""
    kz = ijk[:, 2]
    available = (kz % 2) == 0
    test = ~available
    val = available & (((ijk[:, 0] + 2 * ijk[:, 1] + kz) % 5) == 0)
    train = available & ~val
    return np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)


def scene_signature(scene: TriangleSoup) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(scene.vertices, np.float32).tobytes())
    h.update(np.asarray(scene.colors, np.float32).tobytes())
    return h.hexdigest()[:16]


def render_dataset(scene: TriangleSoup, poses: np.ndarray, height: int, width: int,
                   cache_path: str | Path | None = None) -> np.ndarray:
    cache = Path(cache_path) if cache_path else None
    sig = scene_signature(scene)
    if cache and cache.exists():
        z = np.load(cache)
        if (str(z["scene_signature"]) == sig
                and tuple(z["images"].shape) == (len(poses), height, width, 3)
                and np.allclose(z["poses"], poses)):
            print(f"dataset cache hit: {cache}")
            return z["images"].astype(np.float32)
    images = []
    for i, p in enumerate(poses):
        if i % max(1, len(poses) // 10) == 0:
            print(f"render {i+1}/{len(poses)} camera={np.round(p,3)}")
        images.append(render(scene, p, height, width))
    images = np.stack(images).astype(np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, images=images, poses=poses, scene_signature=np.asarray(sig))
    return images


def make_atlas(images: np.ndarray, ijk: np.ndarray, axes, scale: int = 2) -> Image.Image:
    xs, ys, zs = axes
    H, W = images.shape[1:3]
    pad = 2
    slice_w = len(xs) * (W + pad) + pad
    canvas = Image.new("RGB", (len(zs) * slice_w, len(ys) * (H + pad) + pad), (18, 18, 18))
    for im, (ix, iy, iz) in zip(images, ijk):
        tile = Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255))
        x0 = iz * slice_w + pad + ix * (W + pad)
        y0 = pad + (len(ys) - 1 - iy) * (H + pad)
        canvas.paste(tile, (x0, y0))
    if scale != 1:
        canvas = canvas.resize((canvas.width * scale, canvas.height * scale), Image.Resampling.NEAREST)
    return canvas


def _selftest():
    mesh = tiny_cube_mesh()
    scene = build_scene([mesh, mesh, mesh])
    poses, ijk, axes = camera_grid(3, 2, 5)
    tr, va, te = split_indices(ijk)
    assert len(te) == 3 * 2 * 2
    assert set(ijk[te, 2].tolist()) == {1, 3}
    a = render(scene, poses[0], 24, 24)
    b = render(scene, poses[-1], 24, 24)
    assert a.shape == (24, 24, 3)
    assert np.isfinite(a).all()
    assert float(np.mean(np.abs(a - b))) > 1e-4
    addr = normalized_addresses(poses, axes)
    assert np.max(np.abs(addr)) <= 1.000001
    print(f"obj_xyz_world selftest: PASS ({len(poses)} views, train={len(tr)}, val={len(va)}, test-z={len(te)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", nargs="*", choices=sorted(ASSETS), help="download CC0 OBJ presets")
    ap.add_argument("--asset-dir", default="assets/obj")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--models", nargs="*", default=["spot", "rounded_cube", "avocado"])
    ap.add_argument("--obj", nargs="*", default=[], help="local OBJ path(s); overrides --models for preview")
    ap.add_argument("--max-faces", type=int, default=5000)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", default="xyz_scene_preview.png")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.download is not None:
        names = args.download or list(args.models)
        paths = download_assets(names, args.asset_dir)
        for n, p in zip(names, paths):
            print(f"{n}: {p} ({ASSETS[n]['license']})")
        if not args.preview:
            return
    if args.preview:
        paths = [Path(x) for x in args.obj] if args.obj else download_assets(args.models, args.asset_dir)
        meshes = [load_obj(p, args.max_faces) for p in paths]
        scene = build_scene(meshes)
        poses, ijk, axes = camera_grid()
        images = render_dataset(scene, poses, args.height, args.width, None)
        atlas = make_atlas(images, ijk, axes, scale=2)
        atlas.save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
