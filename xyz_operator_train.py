#!/usr/bin/env python3
"""SplatWorld4 XYZ/depth experiment.

Phase A gives the third operator address an unambiguous physical meaning:
camera depth. Downloaded OBJ meshes are placed at different world depths and
rendered from a sparse 3-D camera lattice. Entire intermediate camera-z planes
are withheld from training.

The learned object is still one small material vector g. It is queried at

    M_g(x,y,z) = [K(g) + D(x,y,z) - omega(z)^2 I + i gamma omega(z) I]^-1

and a small linear readout maps its response to train-only image-PCA
coefficients.

Critical controls:
  * fixed_xyz   : same operator coordinates, g not learned
  * xy_only     : operator never receives camera z
  * wrong_z     : training depths are cyclically relabeled; test uses real z
  * direct_xyz  : small pose MLP
  * direct_xy   : same MLP without z
  * basis oracle: best reconstruction available in the train-only image basis

This does NOT ask depth to emerge from monocular video. It asks the cleaner
preceding question: can the third operator coordinate be taught a real 3-D
camera-depth semantics, and does that semantics matter on unseen z planes?
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise SystemExit("PyTorch is required. Install it with: python -m pip install torch") from exc

import obj_xyz_world as ow
from operator_plate import OperatorPlate


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_pca(images: np.ndarray, train_idx: np.ndarray, k: int):
    X = images.reshape(len(images), -1).astype(np.float64)
    mean = X[train_idx].mean(axis=0)
    A = X[train_idx] - mean
    _u, s, vt = np.linalg.svd(A, full_matrices=False)
    k = min(int(k), len(vt))
    basis = vt[:k]
    coeff = (X - mean) @ basis.T
    return mean.astype(np.float32), basis.astype(np.float32), coeff.astype(np.float32), s[:k]


def decode_images(mean, basis, coeff, shape):
    X = np.asarray(coeff) @ np.asarray(basis) + np.asarray(mean)
    return np.clip(X.reshape((len(X),) + tuple(shape)), 0.0, 1.0).astype(np.float32)


def image_rel_rmse(pred: np.ndarray, true: np.ndarray):
    p = np.asarray(pred, np.float64).reshape(len(pred), -1)
    t = np.asarray(true, np.float64).reshape(len(true), -1)
    num = np.sqrt(np.mean((p - t) ** 2, axis=1))
    den = np.sqrt(np.mean(t ** 2, axis=1)) + 1e-9
    return num / den


def _incidence(edges: np.ndarray, dim: int) -> np.ndarray:
    E = np.zeros((len(edges), dim), np.float64)
    for q, (i, j) in enumerate(edges):
        E[q, i] = +1.0
        E[q, j] = -1.0
    return E


class TorchOperatorPlate(nn.Module):
    """Differentiable, material-conserving version of OperatorPlate."""

    def __init__(self, dim=10, seed=17, learn_material=True, dtype=torch.float32):
        super().__init__()
        p = OperatorPlate(dim=dim, seed=seed)
        self.dim = int(dim)
        self.damping = float(p.damping)
        self.address_gain = float(p.address_gain)
        self.onsite = float(p.onsite)
        self.carrier = float(p.carrier)
        self.min_g = 0.035
        g0 = np.asarray(p.g, np.float64)
        free = np.maximum(g0 - self.min_g, 1e-6)
        self.free_budget = float(free.sum())
        self.theta = nn.Parameter(torch.tensor(np.log(free), dtype=dtype), requires_grad=bool(learn_material))
        self.register_buffer("coords", torch.tensor(p.coords, dtype=dtype))
        self.register_buffer("incidence", torch.tensor(_incidence(p.edges, dim), dtype=dtype))
        self.register_buffer("eye", torch.eye(dim, dtype=dtype))
        self.register_buffer("drive_real", torch.tensor(np.real(p.drive), dtype=dtype))
        self.register_buffer("drive_imag", torch.tensor(np.imag(p.drive), dtype=dtype))

    def material(self):
        return self.min_g + self.free_budget * torch.softmax(self.theta, dim=0)

    def stiffness(self):
        g = self.material()
        weighted = g[:, None] * self.incidence
        return self.onsite * self.eye + self.incidence.T @ weighted

    def _raw(self, address):
        detune = self.address_gain * (address @ self.coords.T)
        omega = self.carrier * (1.0 + 0.16 * torch.tanh(address[:, 2]))
        diag_real = detune - omega[:, None] ** 2
        diag_imag = self.damping * omega[:, None].expand_as(diag_real)
        K = self.stiffness()
        A = torch.complex(
            K[None, :, :].expand(len(address), -1, -1) + torch.diag_embed(diag_real),
            torch.diag_embed(diag_imag),
        )
        drive = torch.complex(self.drive_real, self.drive_imag)
        drive = drive[None, :, None].expand(len(address), -1, -1)
        return torch.linalg.solve(A, drive)[..., 0]

    def forward(self, address):
        r = self._raw(address)
        zero = torch.zeros((1, 3), dtype=address.dtype, device=address.device)
        r0 = self._raw(zero)
        return torch.tanh(3.5 * torch.real(r - r0))


class OperatorPredictor(nn.Module):
    def __init__(self, out_dim: int, dim=10, seed=17, learn_material=True):
        super().__init__()
        self.plate = TorchOperatorPlate(dim, seed, learn_material)
        self.readout = nn.Linear(dim, out_dim)

    def forward(self, x):
        return self.readout(self.plate(x))


class PoseMLP(nn.Module):
    def __init__(self, out_dim: int, use_z=True, hidden=24):
        super().__init__()
        self.use_z = bool(use_z)
        d = 3 if use_z else 2
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x if self.use_z else x[:, :2])


def wrong_z_map(addresses: np.ndarray) -> np.ndarray:
    """Cycle training depths -1 -> 0 -> +1 -> -1, preserving z histogram."""
    a = np.asarray(addresses, np.float32).copy()
    z = a[:, 2]
    out = z.copy()
    out[np.isclose(z, -1.0)] = 0.0
    out[np.isclose(z, 0.0)] = +1.0
    out[np.isclose(z, +1.0)] = -1.0
    a[:, 2] = out
    return a


def transform_addresses(addresses: np.ndarray, mode: str, for_training: bool):
    a = np.asarray(addresses, np.float32).copy()
    if mode in ("xy_only", "direct_xy"):
        a[:, 2] = 0.0
    elif mode == "wrong_z" and for_training:
        a = wrong_z_map(a)
    return a


def train_one(mode: str, addresses: np.ndarray, coeff: np.ndarray,
              train_idx: np.ndarray, val_idx: np.ndarray, seed: int,
              steps: int, device: str, dim: int, lr: float = 0.015):
    seed_all(seed)
    device = torch.device(device)
    ymean = coeff[train_idx].mean(axis=0)
    ystd = coeff[train_idx].std(axis=0) + 1e-4
    yn = (coeff - ymean) / ystd
    operator_mode = mode in ("full_xyz", "fixed_xyz", "xy_only", "wrong_z")
    if operator_mode:
        model = OperatorPredictor(coeff.shape[1], dim, seed, learn_material=(mode != "fixed_xyz"))
    elif mode == "direct_xyz":
        model = PoseMLP(coeff.shape[1], use_z=True)
    elif mode == "direct_xy":
        model = PoseMLP(coeff.shape[1], use_z=False)
    else:
        raise KeyError(mode)
    model.to(device)
    train_a = transform_addresses(addresses, mode, True)
    eval_a = transform_addresses(addresses, mode, False)
    tx = torch.tensor(train_a[train_idx], dtype=torch.float32, device=device)
    ty = torch.tensor(yn[train_idx], dtype=torch.float32, device=device)
    vx = torch.tensor(train_a[val_idx], dtype=torch.float32, device=device)
    vy = torch.tensor(yn[val_idx], dtype=torch.float32, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best, best_state, best_step = float("inf"), None, 0
    patience = max(250, int(steps * 0.30))
    for step in range(int(steps)):
        model.train()
        loss = torch.mean((model(tx) - ty) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                v = torch.mean((model(vx) - vy) ** 2).item()
            if v < best - 1e-7:
                best, best_step, best_state = v, step, copy.deepcopy(model.state_dict())
            if step - best_step > patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ex = torch.tensor(eval_a, dtype=torch.float32, device=device)
    with torch.no_grad():
        predn = model(ex).detach().cpu().numpy()
    pred_coeff = predn * ystd + ymean
    receipt = {"mode": mode, "seed": int(seed), "best_val_mse": float(best), "best_step": int(best_step), "steps_requested": int(steps)}
    if operator_mode:
        with torch.no_grad():
            g = model.plate.material().detach().cpu().numpy()
        receipt.update({
            "material_sum": float(g.sum()),
            "material_min": float(g.min()),
            "material_max": float(g.max()),
            "material_entropy": float(-np.sum((g / g.sum()) * np.log(g / g.sum() + 1e-12))),
        })
    return pred_coeff.astype(np.float32), receipt


def summarize(a):
    a = np.asarray(a, np.float64)
    return {"mean": float(np.mean(a)), "median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75)), "max": float(np.max(a))}


def comparison_sheet(path: Path, images: np.ndarray, ijk: np.ndarray,
                     test_idx: np.ndarray, pred_images: dict[str, np.ndarray], axes):
    xs, ys, _zs = axes
    mid_y = len(ys) // 2
    chosen = [i for i in test_idx if int(ijk[i, 1]) == mid_y]
    chosen = sorted(chosen, key=lambda i: (int(ijk[i, 2]), int(ijk[i, 0])))
    names = ["ground_truth", "full_xyz", "fixed_xyz", "xy_only", "wrong_z", "direct_xyz", "direct_xy"]
    available = [n for n in names if n == "ground_truth" or n in pred_images]
    H, W = images.shape[1:3]
    scale, label_h = 2, 16
    cell_w, cell_h = W * scale, H * scale
    canvas = Image.new("RGB", (max(1, len(chosen)) * cell_w, len(available) * (cell_h + label_h)), (14, 14, 14))
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(available):
        y0 = row * (cell_h + label_h)
        draw.text((3, y0 + 2), name, fill=(235, 235, 235))
        for col, i in enumerate(chosen):
            im = images[i] if name == "ground_truth" else pred_images[name][i]
            tile = Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255)).resize((cell_w, cell_h), Image.Resampling.NEAREST)
            canvas.paste(tile, (col * cell_w, y0 + label_h))
            if row == 0:
                ix, _iy, iz = ijk[i]
                draw.text((col * cell_w + 2, y0 + label_h + 2), f"x{ix} z{iz}", fill=(255, 255, 255))
    canvas.save(path)


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.procedural:
        meshes = [ow.tiny_cube_mesh(), ow.tiny_cube_mesh(), ow.tiny_cube_mesh()]
        model_names = ["cubeA", "cubeB", "cubeC"]
    elif args.obj:
        paths = [Path(x) for x in args.obj]
        meshes = [ow.load_obj(p, args.max_faces) for p in paths]
        model_names = [p.stem for p in paths]
    else:
        paths = ow.download_assets(args.models, args.asset_dir)
        meshes = [ow.load_obj(p, args.max_faces) for p in paths]
        model_names = list(args.models)
    scene = ow.build_scene(meshes)
    poses, ijk, axes = ow.camera_grid(args.nx, args.ny, args.nz, args.x_span, args.y_span, args.z_near, args.z_far)
    if args.nz != 5:
        raise ValueError("this first z-interpolation gate currently requires --nz 5")
    addr = ow.normalized_addresses(poses, axes)
    train_idx, val_idx, test_idx = ow.split_indices(ijk)
    if args.procedural:
        cache_name = "procedural"
    elif args.obj:
        cache_name = "custom-" + "-".join(Path(x).stem for x in args.obj)
    else:
        cache_name = "-".join(args.models)
    cache = out / f"render_cache_{cache_name}_{args.height}x{args.width}.npz"
    images = ow.render_dataset(scene, poses, args.height, args.width, cache)
    ow.make_atlas(images, ijk, axes, scale=2).save(out / "ground_truth_xyz_atlas.png")
    mean, basis, coeff, singular = fit_pca(images, train_idx, args.basis_dim)
    oracle = decode_images(mean, basis, coeff, images.shape[1:])
    oracle_err = image_rel_rmse(oracle[test_idx], images[test_idx])
    modes = ["full_xyz", "fixed_xyz", "xy_only", "wrong_z", "direct_xyz", "direct_xy"]
    pred_coeff_by_mode, receipts, per_seed_metrics = {}, {}, {}
    for mode in modes:
        preds, recs, seed_errs = [], [], []
        print(f"\n=== {mode} ===")
        for seed in range(args.seeds):
            pc, rec = train_one(mode, addr, coeff, train_idx, val_idx, seed, args.steps, args.device, args.operator_dim, args.lr)
            pi = decode_images(mean, basis, pc, images.shape[1:])
            err = image_rel_rmse(pi[test_idx], images[test_idx])
            rec["test_rel_rmse_mean"] = float(np.mean(err))
            rec["test_rel_rmse_median"] = float(np.median(err))
            print(f"seed {seed}: val={rec['best_val_mse']:.5f} test={np.mean(err):.5f} step={rec['best_step']}")
            preds.append(pc); recs.append(rec); seed_errs.append(float(np.mean(err)))
        pred_coeff_by_mode[mode] = np.median(np.stack(preds), axis=0)
        receipts[mode] = recs
        per_seed_metrics[mode] = seed_errs
    pred_images = {mode: decode_images(mean, basis, pc, images.shape[1:]) for mode, pc in pred_coeff_by_mode.items()}
    metrics = {}
    for mode, pim in pred_images.items():
        e = image_rel_rmse(pim[test_idx], images[test_idx])
        by_z = {}
        for kz in sorted(set(ijk[test_idx, 2].tolist())):
            ii = test_idx[ijk[test_idx, 2] == kz]
            by_z[str(int(kz))] = summarize(image_rel_rmse(pim[ii], images[ii]))
        metrics[mode] = {"test": summarize(e), "by_unseen_z_plane": by_z, "per_seed_mean_test_rmse": per_seed_metrics[mode]}
    metrics["basis_oracle"] = {"test": summarize(oracle_err)}
    verdict = {
        "full_beats_xy": bool(metrics["full_xyz"]["test"]["mean"] < metrics["xy_only"]["test"]["mean"]),
        "full_beats_wrong_z": bool(metrics["full_xyz"]["test"]["mean"] < metrics["wrong_z"]["test"]["mean"]),
        "full_beats_fixed": bool(metrics["full_xyz"]["test"]["mean"] < metrics["fixed_xyz"]["test"]["mean"]),
        "full_beats_direct_xyz": bool(metrics["full_xyz"]["test"]["mean"] < metrics["direct_xyz"]["test"]["mean"]),
        "gain_vs_xy": float(metrics["xy_only"]["test"]["mean"] - metrics["full_xyz"]["test"]["mean"]),
        "gain_vs_wrong_z": float(metrics["wrong_z"]["test"]["mean"] - metrics["full_xyz"]["test"]["mean"]),
        "gain_vs_fixed": float(metrics["fixed_xyz"]["test"]["mean"] - metrics["full_xyz"]["test"]["mean"]),
    }
    verdict["z_semantics"] = "Z_COORDINATE_MATTERS_ON_UNSEEN_DEPTH_PLANES" if verdict["full_beats_xy"] and verdict["full_beats_wrong_z"] else "Z_COORDINATE_NOT_YET_EARNED"
    result = {
        "question": "Can one learned operator plate use physical camera (x,y,z) to predict entire unseen intermediate z planes of a real 3-D mesh scene?",
        "models": model_names,
        "asset_note": "download presets are CC0 entries from adobe/lagrange-test-data; renderer ignores OBJ textures/materials",
        "camera_axes": {"x": [float(x) for x in axes[0]], "y": [float(x) for x in axes[1]], "z": [float(x) for x in axes[2]]},
        "split": {
            "train_count": int(len(train_idx)), "validation_count": int(len(val_idx)), "test_count": int(len(test_idx)),
            "train_z_indices": sorted(set(map(int, ijk[train_idx, 2]))),
            "validation_z_indices": sorted(set(map(int, ijk[val_idx, 2]))),
            "test_z_indices": sorted(set(map(int, ijk[test_idx, 2]))),
            "guard": "image PCA basis and all learning use only z planes 0,2,4; entire z planes 1,3 are untouched until final evaluation",
        },
        "operator": {"dim": int(args.operator_dim), "material_parameterization": "g = min_g + fixed_free_budget * softmax(theta), so total material is conserved during gradient training"},
        "training": {"device": str(args.device), "steps": int(args.steps), "seeds": int(args.seeds), "basis_dim": int(basis.shape[0]), "lr": float(args.lr)},
        "pca_train_singular_values": [float(x) for x in singular],
        "metrics": metrics, "verdict": verdict, "receipts": receipts,
    }
    (out / "xyz_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    comparison_sheet(out / "xyz_comparison.png", images, ijk, test_idx, pred_images, axes)
    print("\n=== FINAL ===")
    for mode in modes:
        print(f"{mode:12s} test mean rel-RMSE {metrics[mode]['test']['mean']:.6f}")
    print(f"{'basis_oracle':12s} test mean rel-RMSE {metrics['basis_oracle']['test']['mean']:.6f}")
    print(json.dumps(verdict, indent=2))
    print(f"wrote {out / 'xyz_metrics.json'}")
    print(f"wrote {out / 'xyz_comparison.png'}")
    return result


def selftest():
    seed_all(1)
    scene = ow.build_scene([ow.tiny_cube_mesh()] * 3)
    poses, ijk, axes = ow.camera_grid(3, 2, 5)
    images = ow.render_dataset(scene, poses, 20, 20, None)
    tr, va, te = ow.split_indices(ijk)
    mean, basis, coeff, _ = fit_pca(images, tr, 6)
    addr = ow.normalized_addresses(poses, axes)
    pc, rec = train_one("full_xyz", addr, coeff, tr, va, seed=0, steps=35, device="cpu", dim=6, lr=0.02)
    assert pc.shape == coeff.shape and np.isfinite(pc).all() and rec["material_sum"] > 0
    w = wrong_z_map(addr[tr])
    assert np.any(np.abs(w[:, 2] - addr[tr, 2]) > 0.1)
    oracle = decode_images(mean, basis, coeff, images.shape[1:])
    assert np.isfinite(oracle).all()
    print(f"xyz_operator_train selftest: PASS (train={len(tr)} val={len(va)} held-out-z={len(te)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--procedural", action="store_true", help="use 3 cubes; no downloads")
    ap.add_argument("--models", nargs="+", default=["spot", "rounded_cube", "avocado"], choices=sorted(ow.ASSETS))
    ap.add_argument("--obj", nargs="*", default=[], help="local OBJ path(s); overrides --models")
    ap.add_argument("--asset-dir", default="assets/obj")
    ap.add_argument("--max-faces", type=int, default=5000)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--nz", type=int, default=5)
    ap.add_argument("--x-span", type=float, default=0.95)
    ap.add_argument("--y-span", type=float, default=0.58)
    ap.add_argument("--z-near", type=float, default=2.45)
    ap.add_argument("--z-far", type=float, default=4.05)
    ap.add_argument("--basis-dim", type=int, default=16)
    ap.add_argument("--operator-dim", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="xyz_depth_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.run:
        ap.error("use --run or --selftest")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    print(f"torch {torch.__version__}; device={args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run(args)


if __name__ == "__main__":
    main()
