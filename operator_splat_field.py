#!/usr/bin/env python3
"""Gate 6: operator-addressed 3-D splat field.

This gate removes the camera->image shortcut used by the earlier XYZ experiment.
The operator is queried at *world-space splat anchors*, not camera poses:

    q=(X,Y,Z) -> h_g(q) -> (rgb, opacity)
                       -> differentiable perspective splat renderer -> image

The camera is therefore only a renderer. It cannot directly ask the model what
image belongs to a pose. A single shared material vector g must generate a
view-independent 3-D field that is rendered from sparse training views and
judged on wholly unseen camera-depth planes.

Controls share exactly the same anchors and renderer:
  operator       learned material g + linear splat-property head
  fixed_operator frozen random g + learned linear head
  mlp            ordinary coordinate MLP q -> splat properties
  free           one unconstrained rgb/opacity tuple per anchor
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise SystemExit("PyTorch is required. Install it with: python -m pip install torch") from exc

import obj_xyz_world as ow
from xyz_operator_train import TorchOperatorPlate, image_rel_rmse, seed_all


def make_anchor_grid(nx, ny, nz, x_span=1.05, y_span=0.78, z_min=-1.18, z_max=0.78):
    xs = np.linspace(-x_span, x_span, int(nx), dtype=np.float32)
    ys = np.linspace(-y_span, y_span, int(ny), dtype=np.float32)
    zs = np.linspace(z_min, z_max, int(nz), dtype=np.float32)
    q, ijk = [], []
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                q.append([x, y, z])
                ijk.append([ix, iy, iz])
    q = np.asarray(q, np.float32)
    scale = np.asarray([
        max(abs(xs[0]), abs(xs[-1]), 1e-6),
        max(abs(ys[0]), abs(ys[-1]), 1e-6),
        max(abs(zs[0]), abs(zs[-1]), 1e-6),
    ], np.float32)
    return q, q / scale[None, :], np.asarray(ijk, np.int32), (xs, ys, zs)


def sparse_split(ijk, sparse_views, seed=123):
    kz = np.asarray(ijk)[:, 2]
    candidate = np.flatnonzero((kz % 2) == 0)
    test = np.flatnonzero((kz % 2) == 1)
    if len(candidate) < 3:
        raise ValueError("need at least three non-test views")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(candidate)
    ntrain = min(max(2, int(sparse_views)), max(2, len(candidate) - 1))
    train = np.sort(order[:ntrain])
    val_pool = order[ntrain:]
    if len(val_pool) == 0:
        val = train[-1:]
        train = train[:-1]
    else:
        nval = min(max(1, len(train) // 4), len(val_pool))
        val = np.sort(val_pool[:nval])
    return train, val, np.sort(test)


def _camera_projection(anchors, camera, height, width, radius_world, fov_deg=48.0):
    pos = np.asarray(camera, np.float64)
    right, up, forward = ow.camera_frame(pos)
    rel = np.asarray(anchors, np.float64) - pos[None, :]
    xc = rel @ right
    yc = rel @ up
    zc = rel @ forward
    focal = 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)
    safe_z = np.maximum(zc, 1e-4)
    u = focal * xc / safe_z + (width - 1) * 0.5
    v = -focal * yc / safe_z + (height - 1) * 0.5
    sigma = np.clip(float(radius_world) * focal / safe_z, 0.45, 5.0)

    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    du = xx[None, :, :] - u[:, None, None]
    dv = yy[None, :, :] - v[:, None, None]
    kernel = np.exp(-0.5 * (du * du + dv * dv) / (sigma[:, None, None] ** 2 + 1e-9))
    kernel *= (zc > 0.08)[:, None, None]
    order = np.argsort(zc).astype(np.int64)
    return kernel.astype(np.float32), order, zc.astype(np.float32)


def precompute_renderer(anchors, poses, height, width, radius_world, device):
    kernels, orders, depths = [], [], []
    for p in poses:
        k, o, z = _camera_projection(anchors, p, height, width, radius_world)
        kernels.append(k); orders.append(o); depths.append(z)
    dev = torch.device(device)
    return (
        torch.tensor(np.stack(kernels), dtype=torch.float32, device=dev),
        torch.tensor(np.stack(orders), dtype=torch.long, device=dev),
        torch.tensor(np.stack(depths), dtype=torch.float32, device=dev),
    )


def background(batch, height, width, device):
    y = torch.linspace(0.06, 0.12, height, device=device)[:, None, None]
    bg = y.expand(height, width, 3)
    ramp = torch.linspace(1.0, 0.78, height, device=device)[:, None, None]
    return (bg * ramp).unsqueeze(0).expand(batch, -1, -1, -1)


def render_properties(rgb, opacity, kernels, order, view_idx):
    """Front-to-back differentiable alpha compositing of fixed 3-D anchors."""
    k = kernels[view_idx]
    o = order[view_idx]
    b, n, h, w = k.shape
    ks = torch.gather(k, 1, o[:, :, None, None].expand(-1, -1, h, w))
    a = opacity[o][:, :, None, None] * ks
    a = torch.clamp(a, 0.0, 0.985)
    rgb_s = rgb[o]
    survive = torch.clamp(1.0 - a, min=1e-5)
    trans_after = torch.cumprod(survive, dim=1)
    trans_before = torch.cat([torch.ones_like(a[:, :1]), trans_after[:, :-1]], dim=1)
    weights = trans_before * a
    color = torch.sum(weights[..., None] * rgb_s[:, :, None, None, :], dim=1)
    bg = background(b, h, w, k.device)
    return color + trans_after[:, -1, :, :, None] * bg


def decode_raw(raw):
    rgb = torch.sigmoid(raw[:, :3])
    opacity = 0.98 * torch.sigmoid(raw[:, 3] - 2.5)
    return rgb, opacity


class OperatorField(nn.Module):
    def __init__(self, q_norm, dim, seed, learn_material):
        super().__init__()
        self.plate = TorchOperatorPlate(dim=dim, seed=seed, learn_material=learn_material)
        self.head = nn.Linear(dim, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("q", torch.tensor(q_norm, dtype=torch.float32))

    def forward(self):
        return decode_raw(self.head(self.plate(self.q)))


class MLPField(nn.Module):
    def __init__(self, q_norm, hidden=24):
        super().__init__()
        self.register_buffer("q", torch.tensor(q_norm, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self):
        return decode_raw(self.net(self.q))


class FreeField(nn.Module):
    def __init__(self, q_norm):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros((len(q_norm), 4), dtype=torch.float32))

    def forward(self):
        return decode_raw(self.raw)


def make_model(mode, q_norm, operator_dim, seed):
    if mode == "operator":
        return OperatorField(q_norm, operator_dim, seed, True)
    if mode == "fixed_operator":
        return OperatorField(q_norm, operator_dim, seed, False)
    if mode == "mlp":
        return MLPField(q_norm)
    if mode == "free":
        return FreeField(q_norm)
    raise KeyError(mode)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def train_field(mode, q_norm, target, train_idx, val_idx, kernels, order,
                steps, lr, operator_dim, seed, device):
    seed_all(seed)
    dev = torch.device(device)
    model = make_model(mode, q_norm, operator_dim, seed).to(dev)
    total_params, trainable_params = count_params(model)
    target_t = torch.tensor(target, dtype=torch.float32, device=dev)
    tr = torch.tensor(train_idx, dtype=torch.long, device=dev)
    va = torch.tensor(val_idx, dtype=torch.long, device=dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=1e-5)
    best, best_state, best_step = float("inf"), None, 0
    patience = max(120, int(steps * 0.30))

    for step in range(int(steps)):
        model.train()
        rgb, alpha = model()
        pred = render_properties(rgb, alpha, kernels, order, tr)
        loss = torch.mean((pred - target_t[tr]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vrgb, vaa = model()
                vp = render_properties(vrgb, vaa, kernels, order, va)
                v = torch.mean((vp - target_t[va]) ** 2).item()
            if v < best - 1e-8:
                best, best_step, best_state = v, step, copy.deepcopy(model.state_dict())
            if step - best_step > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_idx = torch.arange(len(target), dtype=torch.long, device=dev)
    with torch.no_grad():
        rgb, alpha = model()
        pred = render_properties(rgb, alpha, kernels, order, all_idx).cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        rgb_np = rgb.cpu().numpy()
    rec = {
        "mode": mode,
        "seed": int(seed),
        "best_val_mse": float(best),
        "best_step": int(best_step),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "mean_opacity": float(np.mean(alpha_np)),
        "active_splats_alpha_gt_0_10": int(np.sum(alpha_np > 0.10)),
    }
    if hasattr(model, "plate"):
        with torch.no_grad():
            g = model.plate.material().detach().cpu().numpy()
        rec["material_sum"] = float(g.sum())
        rec["material_entropy"] = float(-np.sum((g / g.sum()) * np.log(g / g.sum() + 1e-12)))
    return pred.astype(np.float32), rgb_np.astype(np.float32), alpha_np.astype(np.float32), rec


def comparison_sheet(path, target, pred, ijk, test_idx):
    chosen = list(test_idx)
    if len(chosen) > 10:
        take = np.linspace(0, len(chosen) - 1, 10, dtype=int)
        chosen = [chosen[i] for i in take]
    names = ["ground_truth"] + [n for n in ("operator", "fixed_operator", "mlp", "free") if n in pred]
    h, w = target.shape[1:3]
    scale = max(1, 64 // max(h, w))
    th, tw, label_h = h * scale, w * scale, 15
    canvas = Image.new("RGB", (max(1, len(chosen)) * tw, len(names) * (th + label_h)), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for r, name in enumerate(names):
        y0 = r * (th + label_h)
        draw.text((2, y0 + 1), name, fill=(240, 240, 240))
        for c, i in enumerate(chosen):
            im = target[i] if name == "ground_truth" else pred[name][i]
            tile = Image.fromarray(np.uint8(np.clip(im, 0, 1) * 255)).resize((tw, th), Image.Resampling.NEAREST)
            canvas.paste(tile, (c * tw, y0 + label_h))
            if r == 0:
                ix, iy, iz = ijk[i]
                draw.text((c * tw + 2, y0 + label_h + 2), f"{ix},{iy},z{iz}", fill=(255, 255, 255))
    canvas.save(path)


def field_slice_sheet(path, field, anchor_ijk, anchor_axes):
    xs, ys, zs = anchor_axes
    cell = 14
    names = list(field)
    canvas = Image.new("RGB", (len(zs) * len(xs) * cell, len(names) * len(ys) * cell), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(names):
        rgb, alpha = field[name]
        for k in range(len(zs)):
            for i, (ix, iy, iz) in enumerate(anchor_ijk):
                if iz != k:
                    continue
                col = np.clip(rgb[i] * alpha[i], 0.0, 1.0)
                color = tuple(np.uint8(col * 255).tolist())
                x0 = (k * len(xs) + ix) * cell
                y0 = (row * len(ys) + (len(ys) - 1 - iy)) * cell
                draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=color)
    canvas.save(path)


def build_target_scene(args):
    if args.procedural:
        meshes = [ow.tiny_cube_mesh(), ow.tiny_cube_mesh(), ow.tiny_cube_mesh()]
        names = ["cubeA", "cubeB", "cubeC"]
    elif args.obj:
        paths = [Path(x) for x in args.obj]
        meshes = [ow.load_obj(p, args.max_faces) for p in paths]
        names = [p.stem for p in paths]
    else:
        paths = ow.download_assets(args.models, args.asset_dir)
        meshes = [ow.load_obj(p, args.max_faces) for p in paths]
        names = list(args.models)
    return ow.build_scene(meshes), names


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using cpu")
        dev = "cpu"

    scene, scene_names = build_target_scene(args)
    poses, cam_ijk, _cam_axes = ow.camera_grid(args.nx, args.ny, 5, args.x_span, args.y_span, args.z_near, args.z_far)
    cache_tag = "procedural" if args.procedural else "-".join(scene_names)
    cache = out / f"splat_target_{cache_tag}_{args.height}x{args.width}.npz"
    target = ow.render_dataset(scene, poses, args.height, args.width, cache)
    train_idx, val_idx, test_idx = sparse_split(cam_ijk, args.sparse_views, args.split_seed)

    anchors, q_norm, anchor_ijk, anchor_axes = make_anchor_grid(
        args.grid_x, args.grid_y, args.grid_z,
        args.anchor_x_span, args.anchor_y_span, args.anchor_z_min, args.anchor_z_max,
    )
    kernels, order, _depth = precompute_renderer(
        anchors, poses, args.height, args.width, args.splat_radius, dev,
    )

    modes = list(args.modes)
    per_mode, pred_mean, field_mean, receipts = {}, {}, {}, {}
    for mode in modes:
        preds, rgbs, alphas, recs = [], [], [], []
        print(f"\n=== {mode} ===")
        for seed in range(args.seeds):
            pred, rgb, alpha, rec = train_field(
                mode, q_norm, target, train_idx, val_idx, kernels, order,
                args.steps, args.lr, args.operator_dim, seed, dev,
            )
            err = image_rel_rmse(pred[test_idx], target[test_idx])
            rec["test_rel_rmse_mean"] = float(np.mean(err))
            rec["test_rel_rmse_median"] = float(np.median(err))
            print(
                f"seed {seed}: test={rec['test_rel_rmse_mean']:.5f} "
                f"val={rec['best_val_mse']:.6f} params={rec['trainable_params']} "
                f"active={rec['active_splats_alpha_gt_0_10']}"
            )
            preds.append(pred); rgbs.append(rgb); alphas.append(alpha); recs.append(rec)
        pred_mean[mode] = np.mean(preds, axis=0)
        field_mean[mode] = (np.mean(rgbs, axis=0), np.mean(alphas, axis=0))
        receipts[mode] = recs
        vals = [r["test_rel_rmse_mean"] for r in recs]
        per_mode[mode] = {
            "test_rel_rmse_mean": float(np.mean(vals)),
            "test_rel_rmse_std": float(np.std(vals)),
            "trainable_params_mean": float(np.mean([r["trainable_params"] for r in recs])),
        }

    comparison_sheet(out / "operator_splat_comparison.png", target, pred_mean, cam_ijk, test_idx)
    field_slice_sheet(out / "operator_splat_field_slices.png", field_mean, anchor_ijk, anchor_axes)
    metrics = {
        "experiment": "world-space operator splat field",
        "scene": scene_names,
        "camera_views": int(len(poses)),
        "train_views": int(len(train_idx)),
        "validation_views": int(len(val_idx)),
        "test_views": int(len(test_idx)),
        "test_z_planes": sorted(set(int(x) for x in cam_ijk[test_idx, 2])),
        "anchors": int(len(anchors)),
        "anchor_grid": [int(args.grid_x), int(args.grid_y), int(args.grid_z)],
        "sparse_views_requested": int(args.sparse_views),
        "summary": per_mode,
        "receipts": receipts,
    }
    if "operator" in per_mode and "fixed_operator" in per_mode:
        metrics["operator_beats_fixed"] = bool(
            per_mode["operator"]["test_rel_rmse_mean"] < per_mode["fixed_operator"]["test_rel_rmse_mean"]
        )
    if "operator" in per_mode and "mlp" in per_mode:
        metrics["operator_beats_mlp"] = bool(
            per_mode["operator"]["test_rel_rmse_mean"] < per_mode["mlp"]["test_rel_rmse_mean"]
        )
        metrics["operator_param_ratio_vs_mlp"] = float(
            per_mode["operator"]["trainable_params_mean"] /
            max(1.0, per_mode["mlp"]["trainable_params_mean"])
        )

    (out / "operator_splat_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n=== FINAL ===")
    for mode in modes:
        s = per_mode[mode]
        print(
            f"{mode:14s} test={s['test_rel_rmse_mean']:.6f} +/- {s['test_rel_rmse_std']:.6f} "
            f"trainable_params={s['trainable_params_mean']:.0f}"
        )
    print(f"wrote {out / 'operator_splat_metrics.json'}")
    print(f"wrote {out / 'operator_splat_comparison.png'}")
    print(f"wrote {out / 'operator_splat_field_slices.png'}")


def selftest():
    seed_all(0)
    meshes = [ow.tiny_cube_mesh(), ow.tiny_cube_mesh(), ow.tiny_cube_mesh()]
    scene = ow.build_scene(meshes)
    poses, ijk, _axes = ow.camera_grid(3, 2, 5)
    target = np.stack([ow.render(scene, p, 12, 12) for p in poses]).astype(np.float32)
    tr, va, te = sparse_split(ijk, 6, 3)
    anchors, qn, _aijk, _aaxes = make_anchor_grid(3, 2, 3)
    kernels, order, depth = precompute_renderer(anchors, poses, 12, 12, 0.22, "cpu")
    assert kernels.shape == (len(poses), len(anchors), 12, 12)
    assert depth.shape == (len(poses), len(anchors))
    model = OperatorField(qn, dim=6, seed=0, learn_material=True)
    rgb, alpha = model()
    pred = render_properties(rgb, alpha, kernels, order, torch.tensor(te[:2], dtype=torch.long))
    assert pred.shape == (2, 12, 12, 3)
    assert torch.isfinite(pred).all()
    _pred, _rgb, _alpha, rec = train_field(
        "operator", qn, target, tr, va, kernels, order,
        steps=12, lr=0.02, operator_dim=6, seed=0, device="cpu",
    )
    assert np.isfinite(rec["best_val_mse"])
    print(
        f"operator_splat_field selftest: PASS "
        f"(train={len(tr)} val={len(va)} test={len(te)} anchors={len(anchors)})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--procedural", action="store_true")
    ap.add_argument("--models", nargs="*", default=["spot", "rounded_cube", "avocado"])
    ap.add_argument("--obj", nargs="*", default=[])
    ap.add_argument("--asset-dir", default="assets/obj")
    ap.add_argument("--max-faces", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--nx", type=int, default=5)
    ap.add_argument("--ny", type=int, default=3)
    ap.add_argument("--x-span", type=float, default=0.95)
    ap.add_argument("--y-span", type=float, default=0.58)
    ap.add_argument("--z-near", type=float, default=2.45)
    ap.add_argument("--z-far", type=float, default=4.05)
    ap.add_argument("--grid-x", type=int, default=6)
    ap.add_argument("--grid-y", type=int, default=4)
    ap.add_argument("--grid-z", type=int, default=6)
    ap.add_argument("--anchor-x-span", type=float, default=1.05)
    ap.add_argument("--anchor-y-span", type=float, default=0.78)
    ap.add_argument("--anchor-z-min", type=float, default=-1.18)
    ap.add_argument("--anchor-z-max", type=float, default=0.78)
    ap.add_argument("--splat-radius", type=float, default=0.18)
    ap.add_argument("--sparse-views", type=int, default=12)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--operator-dim", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument(
        "--modes", nargs="*",
        choices=["operator", "fixed_operator", "mlp", "free"],
        default=["operator", "fixed_operator", "mlp", "free"],
    )
    ap.add_argument("--out-dir", default="operator_splat_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.run:
        run(args); return
    ap.print_help()


if __name__ == "__main__":
    main()
