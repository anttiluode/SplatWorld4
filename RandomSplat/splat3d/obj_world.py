"""
obj_world.py — the rotating-object training ground (free ground-truth depth)
================================================================================
The teacher for `splat3d`. The motion-residual organ (`the_splat_3d.py`) could
only read depth WHERE the camera moved, and it flickered (see the webcam frame:
depth appears on motion, vanishes on stillness). To see depth from a SINGLE frame
the way you see a face the instant you look, the model has to learn a prior — and
the cheapest possible teacher is a 3D object that turns in front of a renderer,
because then the depth label is FREE: it is the z-buffer.

This module renders meshes (built-in procedural primitives, or any .obj you load)
at random orientations and hands back:
    rgb   (H,W,3) float [0,1]   — what a camera would see
    depth (H,W)   float [0,1]   — NORMALIZED INVERSE depth (1 = nearest, 0 = far)
    mask  (H,W)   bool          — object pixels (vs background plane)

Pure numpy z-buffer rasterizer — no GPU, no GL — so it runs anywhere and the depth
is exact, not estimated. Domain randomization (albedo, light, background texture,
scale, orientation) is on by default to give a fighting chance at the sim-to-real
gap (see the honest note in the README — synthetic teapots do NOT automatically
become your living room).

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""

import numpy as np

try:
    import trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False


# ----------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------
def _normalize_mesh(v):
    """center at origin, scale so the longest axis spans ~1.6 units."""
    v = v - v.mean(0, keepdims=True)
    scale = 1.6 / (np.abs(v).max() + 1e-9)
    return v * scale


def builtin_mesh(name, rng):
    """A few procedural meshes via trimesh primitives. Returns (verts, faces)."""
    if not _HAS_TRIMESH:
        # fallback: a hand-built icosphere-ish via subdivided octahedron
        return _octa_sphere(2)
    if name == "sphere":
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    elif name == "box":
        m = trimesh.creation.box(extents=(1.4, 1.0, 1.2))
    elif name == "torus":
        m = trimesh.creation.torus(major_radius=0.9, minor_radius=0.35)
    elif name == "cylinder":
        m = trimesh.creation.cylinder(radius=0.6, height=1.6, sections=24)
    elif name == "cone":
        m = trimesh.creation.cone(radius=0.7, height=1.6, sections=24)
    elif name == "capsule":
        m = trimesh.creation.capsule(height=1.0, radius=0.5)
    else:  # "random" — pick one
        return builtin_mesh(rng.choice(
            ["sphere", "box", "torus", "cylinder", "cone", "capsule"]), rng)
    return _normalize_mesh(np.asarray(m.vertices, np.float64)), np.asarray(m.faces, np.int64)


def load_obj(path):
    """Load any .obj (or other trimesh-readable mesh). Returns (verts, faces)."""
    if not _HAS_TRIMESH:
        raise RuntimeError("trimesh not installed; pip install trimesh")
    m = trimesh.load(path, force="mesh")
    return _normalize_mesh(np.asarray(m.vertices, np.float64)), np.asarray(m.faces, np.int64)


def _octa_sphere(subdiv):
    """tiny dependency-free fallback sphere (subdivided octahedron)."""
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], int)
    for _ in range(subdiv):
        nv = list(v); nf = []; mid = {}
        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = (v[a] + v[b]); p = p / np.linalg.norm(p)
                mid[k] = len(nv); nv.append(p)
            return mid[k]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(nv); f = np.array(nf)
    return _normalize_mesh(v), f


# ----------------------------------------------------------------------
# rotation
# ----------------------------------------------------------------------
def rot_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


# ----------------------------------------------------------------------
# the rasterizer (numpy z-buffer)
# ----------------------------------------------------------------------
def rasterize(verts, faces, R, size=64, light=None, albedo=(0.8, 0.7, 0.6),
              bg_tex=None, persp=0.35):
    """
    Orthographic-ish projection with a touch of perspective, exact z-buffer.
    Returns rgb(H,W,3) in [0,1], inv_depth(H,W) in [0,1] (1=near), mask(H,W) bool.
    """
    H = W = size
    Vc = verts @ R.T                                  # rotate
    # weak perspective: shrink x,y slightly with distance (z)
    z = Vc[:, 2]
    zmin, zmax = z.min(), z.max()
    denom = (zmax - zmin) + 1e-6
    persp_scale = 1.0 / (1.0 + persp * (zmax - Vc[:, 2]) / denom)
    sx = Vc[:, 0] * persp_scale
    sy = Vc[:, 1] * persp_scale
    # to pixels (object spans ~[-0.9,0.9] -> margin)
    px = (sx * 0.5 + 0.5) * (W - 1)
    py = (1 - (sy * 0.5 + 0.5)) * (H - 1)

    # face normals (in camera space) for Lambertian shading
    a, b, c = Vc[faces[:, 0]], Vc[faces[:, 1]], Vc[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    if light is None:
        light = np.array([0.3, 0.4, 1.0]); light = light / np.linalg.norm(light)
    shade = np.clip((nrm @ light), 0, 1) * 0.85 + 0.15      # ambient floor

    # z for depth: use camera z (larger z = nearer to camera at +z)
    zf = Vc[:, 2]

    # background plane at far depth
    far = zmin - 0.3
    zbuf = np.full((H, W), far, np.float64)
    rgb = np.zeros((H, W, 3), np.float64)
    mask = np.zeros((H, W), bool)
    if bg_tex is not None:
        rgb[:] = bg_tex
    else:
        rgb[:] = 0.12

    alb = np.asarray(albedo, float)

    px_f = px[faces]; py_f = py[faces]; zf_f = zf[faces]
    for i in range(faces.shape[0]):
        x0, x1, x2 = px_f[i]; y0, y1, y2 = py_f[i]; z0, z1, z2 = zf_f[i]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        xs = np.arange(minx, maxx + 1)
        ys = np.arange(miny, maxy + 1)
        gx, gy = np.meshgrid(xs, ys)
        # barycentric
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-9:
            continue
        wa = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / d
        wb = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / d
        wc = 1 - wa - wb
        inside = (wa >= 0) & (wb >= 0) & (wc >= 0)
        if not inside.any():
            continue
        zpix = wa * z0 + wb * z1 + wc * z2
        sub_zbuf = zbuf[miny:maxy + 1, minx:maxx + 1]
        win = inside & (zpix > sub_zbuf)               # nearer (larger z) wins
        if not win.any():
            continue
        col = alb * shade[i]
        sub_rgb = rgb[miny:maxy + 1, minx:maxx + 1]
        sub_rgb[win] = col
        sub_zbuf[win] = zpix[win]
        sub_mask = mask[miny:maxy + 1, minx:maxx + 1]
        sub_mask[win] = True

    # normalized inverse depth: 1 at nearest object point, 0 at far plane
    inv = (zbuf - far) / ((zmax - far) + 1e-6)
    inv = np.clip(inv, 0, 1)
    return rgb.astype(np.float32), inv.astype(np.float32), mask


def random_bg(size, rng):
    """a textured far background plane (domain randomization)."""
    t = rng.random((size // 8, size // 8, 3)).astype(np.float32)
    t = np.kron(t, np.ones((8, 8, 1)))[:size, :size]
    # smooth
    k = np.array([1, 2, 1], float); k = k / k.sum()
    for _ in range(2):
        t = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 0, t)
        t = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 1, t)
    return (t * 0.5 + 0.1).astype(np.float32)


def sample(size=64, rng=None, obj_paths=None, randomize=True):
    """One training sample: a randomly-oriented, randomly-shaded object on a
    random far background. Returns (rgb, inv_depth, mask)."""
    rng = rng or np.random.default_rng()
    if obj_paths:
        verts, faces = load_obj(rng.choice(obj_paths))
    else:
        verts, faces = builtin_mesh("random", rng)
    yaw = rng.uniform(0, 2 * np.pi)
    pitch = rng.uniform(-0.6, 0.6)
    roll = rng.uniform(-0.3, 0.3)
    R = rot_matrix(yaw, pitch, roll)
    if randomize:
        light = rng.normal(0, 1, 3); light[2] = abs(light[2]) + 0.5
        light = light / np.linalg.norm(light)
        albedo = rng.uniform(0.35, 0.95, 3)
        bg = random_bg(size, rng) if rng.random() < 0.8 else None
    else:
        light, albedo, bg = None, (0.8, 0.7, 0.6), None
    return rasterize(verts, faces, R, size, light, albedo, bg)


if __name__ == "__main__":
    # quick self-test: render a few samples, report timing + depth sanity
    import time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    t0 = time.time()
    samples = [sample(args.size, rng) for _ in range(args.n)]
    dt = (time.time() - t0) / args.n
    rgb, inv, mask = samples[0]
    print(f"rendered {args.n} @ {args.size}px in {dt*1000:.0f} ms/sample")
    print(f"depth: obj median inv-depth {np.median(inv[mask]):.3f}  "
          f"bg median {np.median(inv[~mask]):.3f}  (obj should be > bg)")
    if args.save:
        import cv2
        rows = []
        for rgb, inv, mask in samples:
            d = cv2.applyColorMap((inv * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            r = (cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            rows.append(np.hstack([r, d]))
        cv2.imwrite(args.save, np.vstack(rows))
        print("wrote", args.save)