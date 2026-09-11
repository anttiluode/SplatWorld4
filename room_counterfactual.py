#!/usr/bin/env python3
"""SplatWorld4 room/counterfactual-view experiment.

This is deliberately self-contained and uses an analytic ray-cast room rather
than any untracked historical SplatWorld4 files.

Question
--------
Can one shared address-conditioned material substrate, queried by camera
(x, z, yaw), use sparse observed routes to predict views at unvisited interior
positions better than:

  * a capacity-matched direct pose regression,
  * local k-nearest interpolation,
  * matched random smooth features?

The final test poses are never used for basis fitting, ridge selection, g
selection, or seed selection.  We additionally report error as a function of
geometric distance to a real occlusion boundary: whether a bright beacon on the
back wall is hidden by a pillar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from operator_plate import OperatorPlate

# ------------------------------- Room -------------------------------------
ROOM = 3.0
FOV = math.radians(72.0)
BEACON = np.array([0.72, 2.92], dtype=np.float64)
PILLARS = [
    (np.array([0.00, 0.35], dtype=np.float64), 0.48, 0.78),
    (np.array([-1.28, 1.25], dtype=np.float64), 0.38, 0.56),
    (np.array([1.38, 1.62], dtype=np.float64), 0.42, 0.68),
]


def wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def look_at(x: float, z: float, target=BEACON) -> float:
    # yaw=0 points +z
    return math.atan2(float(target[0] - x), float(target[1] - z))


def _ray_circle(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float):
    oc = origin - center
    b = float(np.dot(oc, direction))
    c = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - c
    if disc <= 0.0:
        return None
    s = math.sqrt(disc)
    t0, t1 = -b - s, -b + s
    if t0 > 1e-5:
        return t0
    if t1 > 1e-5:
        return t1
    return None


def _wall_hit(origin: np.ndarray, direction: np.ndarray):
    """Return nearest room-wall ray hit as (distance, wall_id, coordinate)."""
    best = None
    x0, z0 = float(origin[0]), float(origin[1])
    dx, dz = float(direction[0]), float(direction[1])

    # wall ids: 0 back(+z), 1 front(-z), 2 right(+x), 3 left(-x)
    candidates = []
    if abs(dz) > 1e-10:
        for zid, zw in ((0, ROOM), (1, -ROOM)):
            t = (zw - z0) / dz
            x = x0 + t * dx
            if t > 1e-5 and -ROOM <= x <= ROOM:
                candidates.append((t, zid, x))
    if abs(dx) > 1e-10:
        for xid, xw in ((2, ROOM), (3, -ROOM)):
            t = (xw - x0) / dx
            z = z0 + t * dz
            if t > 1e-5 and -ROOM <= z <= ROOM:
                candidates.append((t, xid, z))
    if candidates:
        best = min(candidates, key=lambda q: q[0])
    return best


def _surface_albedo(kind: int, coord: float, point: np.ndarray) -> float:
    if kind >= 10:  # pillar
        base = {10: 0.72, 11: 0.48, 12: 0.62}.get(kind, 0.6)
        stripe = 0.10 * (0.5 + 0.5 * math.sin(7.5 * math.atan2(point[1], point[0] + 1e-9)))
        return float(np.clip(base + stripe, 0.12, 0.95))

    # Room walls each carry a different low-order texture.  The back wall also
    # contains a narrow bright beacon patch whose visibility changes sharply at
    # pillar tangencies.
    if kind == 0:
        beacon = 0.98 if abs(coord - BEACON[0]) < 0.18 else 0.0
        tex = 0.34 + 0.10 * math.sin(2.2 * coord) + 0.07 * math.sin(5.1 * coord + 0.4)
        return max(tex, beacon)
    if kind == 1:
        return 0.28 + 0.08 * (0.5 + 0.5 * math.sin(3.7 * coord))
    if kind == 2:
        return 0.46 + 0.08 * (0.5 + 0.5 * math.cos(4.2 * coord))
    return 0.38 + 0.10 * (0.5 + 0.5 * math.sin(4.7 * coord + 1.1))


def render_pose(pose, h: int = 32, w: int = 64) -> np.ndarray:
    """Simple 2.5-D ray caster returning a grayscale image in [0,1]."""
    x, z, yaw = map(float, pose)
    origin = np.array([x, z], dtype=np.float64)
    img = np.zeros((h, w), dtype=np.float64)
    horizon = h // 2

    # Ceiling / floor provide weak perspective context.
    for r in range(h):
        if r < horizon:
            img[r, :] = 0.045 + 0.025 * (r / max(horizon, 1))
        else:
            q = (r - horizon) / max(h - horizon - 1, 1)
            img[r, :] = 0.10 + 0.12 * q

    for j in range(w):
        u = ((j + 0.5) / w - 0.5) * FOV
        ang = yaw + u
        d = np.array([math.sin(ang), math.cos(ang)], dtype=np.float64)
        wall = _wall_hit(origin, d)
        if wall is None:
            continue
        best_t, best_kind, coord = wall
        point = origin + best_t * d

        for pi, (c, rad, _alb) in enumerate(PILLARS):
            t = _ray_circle(origin, d, c, rad)
            if t is not None and t < best_t:
                best_t = t
                best_kind = 10 + pi
                point = origin + t * d
                coord = math.atan2(float(point[1] - c[1]), float(point[0] - c[0]))

        # fisheye correction for vertical slice height
        depth = max(0.08, best_t * math.cos(u))
        slice_h = int(np.clip(0.92 * h / depth, 2, h * 1.8))
        top = max(0, horizon - slice_h // 2)
        bot = min(h, horizon + slice_h // 2)
        alb = _surface_albedo(best_kind, float(coord), point)
        shade = 0.38 + 0.62 * math.exp(-0.18 * depth)
        val = float(np.clip(alb * shade, 0.0, 1.0))
        img[top:bot, j] = val

        # subtle distance cue on floor below the wall slice
        if bot < h:
            img[bot:, j] *= 0.88 + 0.12 * math.exp(-0.2 * depth)

    # weak checker modulation on lower half; this creates parallax without
    # introducing any learned geometry.
    rr, cc = np.indices((h - horizon, w))
    img[horizon:, :] *= 0.94 + 0.06 * (((rr // 3 + cc // 5) % 2) == 0)
    return np.clip(img, 0.0, 1.0)


# ------------------------- Poses / occlusion -------------------------------

def _safe_pose(x: float, z: float) -> bool:
    if abs(x) > ROOM - 0.3 or abs(z) > ROOM - 0.3:
        return False
    p = np.array([x, z])
    return all(np.linalg.norm(p - c) > rad + 0.28 for c, rad, _ in PILLARS)


def training_poses() -> tuple[np.ndarray, np.ndarray]:
    """Three sparse traversed routes; fold ids are spatially blocked."""
    rows, folds = [], []
    # bottom traverse
    xs = np.linspace(-2.35, 2.35, 9)
    for i, x in enumerate(xs):
        z = -2.15
        rows.append((x, z, look_at(x, z)))
        folds.append(i % 4)
    # left vertical traverse
    zs = np.linspace(-1.55, 2.25, 8)
    for i, z in enumerate(zs):
        x = -2.22
        rows.append((x, z, look_at(x, z)))
        folds.append(i % 4)
    # right vertical traverse
    zs = np.linspace(-1.45, 2.15, 8)
    for i, z in enumerate(zs):
        x = 2.20
        rows.append((x, z, look_at(x, z)))
        folds.append(i % 4)
    return np.asarray(rows, np.float64), np.asarray(folds, np.int32)


def test_poses() -> np.ndarray:
    """Interior counterfactual positions never visited by the sparse routes."""
    pts = []
    xs = [-1.72, -1.15, -0.58, 0.0, 0.58, 1.15, 1.72]
    zs = [-1.42, -0.82, -0.22, 0.42, 1.02, 1.62]
    for zi, z in enumerate(zs):
        for xi, x in enumerate(xs):
            if not _safe_pose(x, z):
                continue
            # deterministic thinning keeps the run compact while retaining all
            # sides of the central-pillar shadow.
            if (2 * xi + zi) % 3 == 0:
                continue
            yaw = look_at(x, z)
            pts.append((x, z, yaw))
    return np.asarray(pts, np.float64)


def beacon_visible_xy(x: float, z: float) -> bool:
    p = np.array([x, z], dtype=np.float64)
    v = BEACON - p
    L = float(np.linalg.norm(v))
    if L < 1e-8:
        return True
    d = v / L
    for c, rad, _ in PILLARS:
        t = _ray_circle(p, d, c, rad)
        if t is not None and t < L:
            return False
    return True


def occlusion_boundary_distance(pose, max_dist=1.2) -> float:
    """Approximate positional distance until beacon visibility flips.

    Search radially in the floor plane while recomputing the camera's look-at
    yaw.  This is a geometric diagnostic only; the model never receives it.
    """
    x, z, _ = map(float, pose)
    base = beacon_visible_xy(x, z)
    dirs = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    radii = np.linspace(0.03, max_dist, 40)
    best = max_dist
    for r in radii:
        for a in dirs:
            xx = x + r * math.cos(a)
            zz = z + r * math.sin(a)
            if not _safe_pose(xx, zz):
                continue
            if beacon_visible_xy(xx, zz) != base:
                return float(r)
    return float(best)


# ----------------------------- Models --------------------------------------

def fit_basis(images: np.ndarray, k: int):
    flat = images.reshape(len(images), -1)
    mean = flat.mean(axis=0)
    X = flat - mean
    _U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(int(k), Vt.shape[0], len(images) - 1)
    basis = Vt[:k]
    coeff = X @ basis.T
    return mean, basis, coeff, S


def decode(mean, basis, coeff, shape):
    return np.clip((mean + coeff @ basis).reshape(shape), 0.0, 1.0)


def pose_core(poses: np.ndarray) -> np.ndarray:
    p = np.asarray(poses, np.float64)
    return np.column_stack([p[:, 0] / ROOM, p[:, 1] / ROOM, p[:, 2] / math.pi])


def direct_pose_features(poses: np.ndarray) -> np.ndarray:
    p = np.asarray(poses, np.float64)
    return np.column_stack([
        p[:, 0] / ROOM,
        p[:, 1] / ROOM,
        p[:, 2] / math.pi,
        np.sin(p[:, 2]),
        np.cos(p[:, 2]),
    ])


def plate_address(pose) -> np.ndarray:
    x, z, yaw = map(float, pose)
    return np.array([x / ROOM, z / ROOM, yaw / math.pi], dtype=np.float64)


def operator_features(plate: OperatorPlate, poses: np.ndarray, feat_dim=5) -> np.ndarray:
    out = []
    for p in poses:
        r = plate.response(plate_address(p))
        out.append(r[:feat_dim])
    return np.asarray(out, np.float64)


def random_features(poses: np.ndarray, seed: int, feat_dim=5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = pose_core(poses)
    W = rng.normal(0.0, 1.25, size=(3, feat_dim))
    b = rng.uniform(-1.0, 1.0, size=feat_dim)
    return np.tanh(X @ W + b)


def _ridge_fit(F: np.ndarray, Y: np.ndarray, ridge: float):
    X = np.column_stack([F, np.ones(len(F))])
    A = X.T @ X + float(ridge) * np.eye(X.shape[1])
    A[-1, -1] -= float(ridge)  # do not penalize intercept
    return np.linalg.solve(A, X.T @ Y)


def _ridge_predict(F: np.ndarray, W: np.ndarray):
    return np.column_stack([F, np.ones(len(F))]) @ W


def cv_ridge(F: np.ndarray, Y: np.ndarray, folds: np.ndarray, ridges=(0.01, 0.1, 1.0, 10.0)):
    best = None
    for ridge in ridges:
        errs = []
        for f in sorted(set(map(int, folds))):
            va = folds == f
            tr = ~va
            W = _ridge_fit(F[tr], Y[tr], ridge)
            pred = _ridge_predict(F[va], W)
            errs.append(np.mean((pred - Y[va]) ** 2))
        score = float(np.mean(errs))
        if best is None or score < best[0]:
            best = (score, float(ridge))
    assert best is not None
    W = _ridge_fit(F, Y, best[1])
    return best[0], best[1], W


def fit_plate_cv(train_poses, train_coeff, folds, seed, feat_dim=5, steps=700):
    rng = np.random.default_rng(seed + 10000)
    plate = OperatorPlate(dim=6, seed=seed)
    total = plate.material_sum

    def score_current():
        F = operator_features(plate, train_poses, feat_dim)
        s, ridge, _W = cv_ridge(F, train_coeff, folds)
        return s, ridge

    best, best_ridge = score_current()
    initial = best
    accepted = 0
    delta = 0.045
    for step in range(int(steps)):
        old = plate.g.copy()
        if not plate.mutate(rng, delta=delta):
            continue
        val, ridge = score_current()
        if val < best:
            best, best_ridge = val, ridge
            accepted += 1
        else:
            plate.g[:] = old
        if step in (steps // 3, 2 * steps // 3):
            delta *= 0.45
    assert abs(plate.material_sum - total) < 1e-10
    F = operator_features(plate, train_poses, feat_dim)
    _s, _r, W = cv_ridge(F, train_coeff, folds, ridges=(best_ridge,))
    return plate, W, {
        "initial_cv_mse": float(initial),
        "final_cv_mse": float(best),
        "ridge": float(best_ridge),
        "accepted": int(accepted),
        "material_l2_change": float(np.linalg.norm(plate.g - OperatorPlate(dim=6, seed=seed).g)),
    }


def local_knn(train_poses, train_coeff, test_poses, k=4):
    tr = np.asarray(train_poses, np.float64)
    out = []
    for p in np.asarray(test_poses, np.float64):
        dx = (tr[:, 0] - p[0]) / ROOM
        dz = (tr[:, 1] - p[1]) / ROOM
        dy = np.array([wrap_angle(a - p[2]) for a in tr[:, 2]]) / math.pi
        d = np.sqrt(dx * dx + dz * dz + 0.45 * dy * dy)
        ix = np.argsort(d)[:k]
        w = 1.0 / (d[ix] + 0.06) ** 2
        w /= w.sum()
        out.append(np.sum(train_coeff[ix] * w[:, None], axis=0))
    return np.asarray(out)


def rel_rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-12))


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float(99.0 if mse < 1e-14 else -10.0 * math.log10(mse))


def image_errors(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    num = np.sqrt(np.mean((pred - true) ** 2, axis=(1, 2)))
    den = np.sqrt(np.mean(true ** 2, axis=(1, 2))) + 1e-12
    return num / den


def summarize_bins(boundary_dist: np.ndarray, per_method: dict[str, np.ndarray]):
    order = np.argsort(boundary_dist)
    n = len(order)
    k = max(1, n // 3)
    groups = {
        "near": order[:k],
        "middle": order[k:n-k] if n > 2 * k else order[k:2*k],
        "far": order[n-k:],
    }
    out = {}
    for name, idx in groups.items():
        out[name] = {
            "n": int(len(idx)),
            "boundary_distance_mean": float(np.mean(boundary_dist[idx])) if len(idx) else None,
            "errors": {m: float(np.mean(v[idx])) for m, v in per_method.items()} if len(idx) else {},
        }
    return out


# ----------------------------- Experiment ----------------------------------

def run(size_h=32, size_w=64, basis_dim=12, seeds=8, g_steps=700, out_dir="room_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_poses, folds = training_poses()
    test = test_poses()
    train_images = np.stack([render_pose(p, size_h, size_w) for p in train_poses])
    test_images = np.stack([render_pose(p, size_h, size_w) for p in test])

    mean, basis, train_coeff, singulars = fit_basis(train_images, basis_dim)
    test_flat = test_images.reshape(len(test), -1)
    true_test_coeff = (test_flat - mean) @ basis.T
    oracle_images = np.stack([decode(mean, basis, c, (size_h, size_w)) for c in true_test_coeff])

    # Direct pose baseline: same five feature coordinates as the five operator
    # response coordinates, with ridge selected on blocked training CV only.
    F_pose_tr = direct_pose_features(train_poses)
    F_pose_te = direct_pose_features(test)
    pose_cv, pose_ridge, pose_W = cv_ridge(F_pose_tr, train_coeff, folds)
    pose_coeff = _ridge_predict(F_pose_te, pose_W)
    pose_images = np.stack([decode(mean, basis, c, (size_h, size_w)) for c in pose_coeff])

    knn_coeff = local_knn(train_poses, train_coeff, test, k=4)
    knn_images = np.stack([decode(mean, basis, c, (size_h, size_w)) for c in knn_coeff])

    random_rows = []
    operator_fixed_rows = []
    operator_learned_rows = []
    fixed_images_for_seed0 = None
    learned_images_for_seed0 = None

    for seed in range(int(seeds)):
        # matched random smooth features
        Fr_tr = random_features(train_poses, seed, 5)
        Fr_te = random_features(test, seed, 5)
        rcv, rridge, rW = cv_ridge(Fr_tr, train_coeff, folds)
        rc = _ridge_predict(Fr_te, rW)
        ri = np.stack([decode(mean, basis, c, (size_h, size_w)) for c in rc])
        random_rows.append({
            "seed": seed,
            "cv_mse": rcv,
            "ridge": rridge,
            "rel_rmse": rel_rmse(ri, test_images),
            "psnr": psnr(ri, test_images),
        })

        # fixed untrained operator
        base = OperatorPlate(dim=6, seed=seed)
        Fo_tr = operator_features(base, train_poses, 5)
        Fo_te = operator_features(base, test, 5)
        ocv, oridge, oW = cv_ridge(Fo_tr, train_coeff, folds)
        oc = _ridge_predict(Fo_te, oW)
        oi = np.stack([decode(mean, basis, c, (size_h, size_w)) for c in oc])
        operator_fixed_rows.append({
            "seed": seed,
            "cv_mse": ocv,
            "ridge": oridge,
            "rel_rmse": rel_rmse(oi, test_images),
            "psnr": psnr(oi, test_images),
        })

        # g is selected only by blocked CV inside the training routes.
        plate, lW, info = fit_plate_cv(train_poses, train_coeff, folds, seed, 5, g_steps)
        Fl_te = operator_features(plate, test, 5)
        lc = _ridge_predict(Fl_te, lW)
        li = np.stack([decode(mean, basis, c, (size_h, size_w)) for c in lc])
        operator_learned_rows.append({
            "seed": seed,
            "rel_rmse": rel_rmse(li, test_images),
            "psnr": psnr(li, test_images),
            **info,
        })
        if seed == 0:
            fixed_images_for_seed0 = oi
            learned_images_for_seed0 = li

    def dist(rows, key="rel_rmse"):
        a = np.asarray([r[key] for r in rows], np.float64)
        return {
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "min_diagnostic_only": float(np.min(a)),
            "max": float(np.max(a)),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
        }

    # Bin a fixed preregistered seed (0) plus seed-invariant baselines by true
    # geometric distance to a pillar-induced beacon visibility boundary.
    boundary = np.asarray([occlusion_boundary_distance(p) for p in test], np.float64)
    per_method = {
        "direct_pose": image_errors(pose_images, test_images),
        "local_knn": image_errors(knn_images, test_images),
        "operator_fixed_seed0": image_errors(fixed_images_for_seed0, test_images),
        "operator_learned_seed0": image_errors(learned_images_for_seed0, test_images),
        "basis_oracle": image_errors(oracle_images, test_images),
    }
    bins = summarize_bins(boundary, per_method)

    learned_arr = np.asarray([r["rel_rmse"] for r in operator_learned_rows])
    fixed_arr = np.asarray([r["rel_rmse"] for r in operator_fixed_rows])
    metrics = {
        "question": "Can learned shared g predict counterfactual unvisited room views beyond direct pose, local interpolation, and matched random features?",
        "n_train": int(len(train_poses)),
        "n_test": int(len(test)),
        "basis_dim": int(basis.shape[0]),
        "seeds": int(seeds),
        "g_steps_per_seed": int(g_steps),
        "direct_pose": {
            "rel_rmse": rel_rmse(pose_images, test_images),
            "psnr": psnr(pose_images, test_images),
            "train_cv_mse": float(pose_cv),
            "ridge": float(pose_ridge),
        },
        "local_knn": {
            "rel_rmse": rel_rmse(knn_images, test_images),
            "psnr": psnr(knn_images, test_images),
        },
        "basis_oracle": {
            "rel_rmse": rel_rmse(oracle_images, test_images),
            "psnr": psnr(oracle_images, test_images),
        },
        "random5_distribution": dist(random_rows),
        "operator_fixed_distribution": dist(operator_fixed_rows),
        "operator_learned_distribution": dist(operator_learned_rows),
        "paired_learned_beats_fixed_count": int(np.sum(learned_arr < fixed_arr)),
        "paired_seed_count": int(len(fixed_arr)),
        "occlusion_distance_bins_seed0": bins,
        "basis_singular_values": [float(v) for v in singulars[:basis.shape[0]]],
        "random_rows": random_rows,
        "operator_fixed_rows": operator_fixed_rows,
        "operator_learned_rows": operator_learned_rows,
        "interpretation_guard": "No seed is selected using test error. Seed minima are diagnostic only.",
    }

    # A conservative verdict: learned g must improve the paired median over fixed
    # operator coordinates AND beat direct pose and local interpolation.
    learned_med = metrics["operator_learned_distribution"]["median"]
    fixed_med = metrics["operator_fixed_distribution"]["median"]
    metrics["learned_g_beats_fixed_median"] = bool(learned_med < fixed_med)
    metrics["learned_g_beats_direct_pose"] = bool(learned_med < metrics["direct_pose"]["rel_rmse"])
    metrics["learned_g_beats_local_knn"] = bool(learned_med < metrics["local_knn"]["rel_rmse"])
    metrics["verdict"] = (
        "COUNTERFACTUAL_ROOM_SIGNAL_SHOWN"
        if metrics["learned_g_beats_fixed_median"]
        and metrics["learned_g_beats_direct_pose"]
        and metrics["learned_g_beats_local_knn"]
        else "COUNTERFACTUAL_ROOM_SIGNAL_NOT_YET_SHOWN"
    )

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "per_pose.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["x", "z", "yaw", "beacon_visible", "boundary_distance", *per_method.keys()])
        for i, p in enumerate(test):
            wr.writerow([
                float(p[0]), float(p[1]), float(p[2]), int(beacon_visible_xy(p[0], p[1])), float(boundary[i]),
                *[float(per_method[m][i]) for m in per_method],
            ])
    np.savez_compressed(
        out / "room_data.npz",
        train_poses=train_poses,
        test_poses=test,
        train_images=train_images,
        test_images=test_images,
        direct_pose_images=pose_images,
        knn_images=knn_images,
        operator_fixed_seed0=fixed_images_for_seed0,
        operator_learned_seed0=learned_images_for_seed0,
        oracle_images=oracle_images,
        boundary_distance=boundary,
    )
    render_sheet(out / "comparison.png", test, test_images, pose_images, knn_images, fixed_images_for_seed0, learned_images_for_seed0)
    print(json.dumps(metrics, indent=2))
    return metrics


def render_sheet(path: Path, poses, truth, pose_img, knn_img, fixed_img, learned_img):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    # show the 10 poses nearest an occlusion boundary
    bd = np.asarray([occlusion_boundary_distance(p) for p in poses])
    ix = np.argsort(bd)[: min(10, len(poses))]
    rows = [
        ("TRUTH", truth),
        ("DIRECT POSE", pose_img),
        ("LOCAL KNN", knn_img),
        ("OP FIXED s0", fixed_img),
        ("OP LEARNED s0", learned_img),
    ]
    h, w = truth.shape[1:]
    scale = 2
    tw, th = w * scale, h * scale
    left, top = 150, 28
    can = Image.new("RGB", (left + tw * len(ix), top + th * len(rows)), (18, 18, 18))
    dr = ImageDraw.Draw(can)
    for j, k in enumerate(ix):
        x, z, _ = poses[k]
        dr.text((left + j * tw + 2, 4), f"{x:+.1f},{z:+.1f}", fill=(150, 255, 150))
    for r, (name, imgs) in enumerate(rows):
        y = top + r * th
        dr.text((5, y + 5), name, fill=(150, 255, 150))
        for j, k in enumerate(ix):
            im = Image.fromarray(np.uint8(np.clip(imgs[k], 0, 1) * 255), mode="L").convert("RGB")
            im = im.resize((tw, th), Image.Resampling.NEAREST)
            can.paste(im, (left + j * tw, y))
    can.save(path)


def selftest():
    tr, folds = training_poses()
    te = test_poses()
    assert tr.shape[1] == 3 and te.shape[1] == 3
    assert len(set(folds.tolist())) == 4
    a = render_pose(tr[0], 20, 36)
    b = render_pose(tr[len(tr)//2], 20, 36)
    assert a.shape == (20, 36) and np.isfinite(a).all()
    assert np.mean(np.abs(a - b)) > 1e-3
    # Occlusion boundary diagnostic must find both visible and hidden beacon views.
    vis = [beacon_visible_xy(p[0], p[1]) for p in te]
    assert any(vis) and not all(vis)
    # Model utilities smoke test.
    imgs = np.stack([render_pose(p, 16, 24) for p in tr[:10]])
    mean, basis, coeff, _ = fit_basis(imgs, 5)
    F = direct_pose_features(tr[:10])
    s, ridge, W = cv_ridge(F, coeff, np.arange(10) % 2)
    assert np.isfinite(s) and ridge > 0 and np.isfinite(W).all()
    p = OperatorPlate(dim=6, seed=0)
    assert operator_features(p, tr[:4], 5).shape == (4, 5)
    print(f"room_counterfactual selftest: PASS ({len(tr)} train, {len(te)} counterfactual test poses)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--basis-dim", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--g-steps", type=int, default=700)
    ap.add_argument("--out-dir", default="room_out")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.height, args.width, args.basis_dim, args.seeds, args.g_steps, args.out_dir)


if __name__ == "__main__":
    main()
