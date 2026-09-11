#!/usr/bin/env python3
"""Gate 6b: parameter-matched controls for the world-space operator splat field.

Gate 6 found that a learned operator field (64 trainable parameters at the
default operator_dim=10) beats the same field with frozen material (44
trainable parameters).  The missing control is whether those 20 extra material
parameters are a useful *substrate* inductive bias, or simply 20 extra degrees
of freedom.

This script attacks that confound with:

  operator64          learned material + linear head (Gate 6 model)
  fixed_deep64        frozen material + nonlinear 10->4->4 head; exactly 64
                      trainable parameters when operator_dim=10
  tiny_mlp56          coordinate MLP 3->4->4->4; 56 parameters
  tiny_mlp74          coordinate MLP 3->5->5->4; 74 parameters

All models use the same world-space anchors, sparse camera views, renderer,
validation split, held-out intermediate camera-depth planes, optimizer, and
seed pairing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import obj_xyz_world as ow
import operator_splat_field as osf
from xyz_operator_train import image_rel_rmse, seed_all


class FixedDeepOperatorField(nn.Module):
    """Frozen plate with a capacity-matched nonlinear readout."""

    def __init__(self, q_norm, dim=10, seed=0, hidden=4):
        super().__init__()
        self.plate = osf.TorchOperatorPlate(dim=dim, seed=seed, learn_material=False)
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        self.register_buffer("q", torch.tensor(q_norm, dtype=torch.float32))

    def forward(self):
        return osf.decode_raw(self.head(self.plate(self.q)))


class TinyMLPField(nn.Module):
    def __init__(self, q_norm, hidden):
        super().__init__()
        self.register_buffer("q", torch.tensor(q_norm, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self):
        return osf.decode_raw(self.net(self.q))


def make_model(mode, q_norm, operator_dim, seed):
    if mode == "operator64":
        return osf.OperatorField(q_norm, operator_dim, seed, True)
    if mode == "fixed_deep64":
        return FixedDeepOperatorField(q_norm, operator_dim, seed, hidden=4)
    if mode == "tiny_mlp56":
        return TinyMLPField(q_norm, hidden=4)
    if mode == "tiny_mlp74":
        return TinyMLPField(q_norm, hidden=5)
    if mode == "fixed_operator44":
        return osf.OperatorField(q_norm, operator_dim, seed, False)
    if mode == "mlp796":
        return osf.MLPField(q_norm, hidden=24)
    if mode == "free576":
        return osf.FreeField(q_norm)
    raise KeyError(mode)


def train_one(mode, q_norm, target, train_idx, val_idx, kernels, order,
              steps, lr, operator_dim, seed, device):
    seed_all(seed)
    dev = torch.device(device)
    model = make_model(mode, q_norm, operator_dim, seed).to(dev)
    total_params, trainable_params = osf.count_params(model)
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
        pred = osf.render_properties(rgb, alpha, kernels, order, tr)
        loss = torch.mean((pred - target_t[tr]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vrgb, vaa = model()
                vp = osf.render_properties(vrgb, vaa, kernels, order, va)
                v = torch.mean((vp - target_t[va]) ** 2).item()
            if v < best - 1e-8:
                best = v
                best_step = step
                best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
            if step - best_step > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_idx = torch.arange(len(target), dtype=torch.long, device=dev)
    with torch.no_grad():
        rgb, alpha = model()
        pred = osf.render_properties(rgb, alpha, kernels, order, all_idx).cpu().numpy()
        alpha_np = alpha.cpu().numpy()

    rec = {
        "mode": mode,
        "seed": int(seed),
        "best_val_mse": float(best),
        "best_step": int(best_step),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "mean_opacity": float(np.mean(alpha_np)),
        "active_splats_alpha_gt_0_10": int(np.sum(alpha_np > 0.10)),
    }
    if hasattr(model, "plate"):
        with torch.no_grad():
            g = model.plate.material().detach().cpu().numpy()
        rec["material_sum"] = float(g.sum())
        rec["material_entropy"] = float(-np.sum((g / g.sum()) * np.log(g / g.sum() + 1e-12)))
    return pred.astype(np.float32), rec


def build_target(args):
    scene, names = osf.build_target_scene(args)
    poses, cam_ijk, _ = ow.camera_grid(
        args.nx, args.ny, 5,
        args.x_span, args.y_span, args.z_near, args.z_far,
    )
    cache_tag = "procedural" if args.procedural else "-".join(names)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"capacity_target_{cache_tag}_{args.height}x{args.width}.npz"
    target = ow.render_dataset(scene, poses, args.height, args.width, cache)
    return target, poses, cam_ijk, names


def run(args):
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using cpu")
        dev = "cpu"

    target, poses, cam_ijk, names = build_target(args)
    train_idx, val_idx, test_idx = osf.sparse_split(cam_ijk, args.sparse_views, args.split_seed)
    anchors, q_norm, _anchor_ijk, _anchor_axes = osf.make_anchor_grid(
        args.grid_x, args.grid_y, args.grid_z,
        args.anchor_x_span, args.anchor_y_span,
        args.anchor_z_min, args.anchor_z_max,
    )
    kernels, order, _ = osf.precompute_renderer(
        anchors, poses, args.height, args.width, args.splat_radius, dev,
    )

    modes = list(args.modes)
    receipts = {}
    summary = {}
    for mode in modes:
        recs = []
        print(f"\n=== {mode} ===")
        for seed in range(args.seeds):
            pred, rec = train_one(
                mode, q_norm, target, train_idx, val_idx, kernels, order,
                args.steps, args.lr, args.operator_dim, seed, dev,
            )
            err = image_rel_rmse(pred[test_idx], target[test_idx])
            rec["test_rel_rmse_mean"] = float(np.mean(err))
            rec["test_rel_rmse_median"] = float(np.median(err))
            print(
                f"seed {seed}: test={rec['test_rel_rmse_mean']:.5f} "
                f"val={rec['best_val_mse']:.6f} "
                f"params={rec['trainable_params']} "
                f"active={rec['active_splats_alpha_gt_0_10']}"
            )
            recs.append(rec)
        receipts[mode] = recs
        vals = np.asarray([r["test_rel_rmse_mean"] for r in recs], np.float64)
        summary[mode] = {
            "test_rel_rmse_mean": float(vals.mean()),
            "test_rel_rmse_std": float(vals.std()),
            "trainable_params": int(recs[0]["trainable_params"]),
        }

    paired = {}
    if "operator64" in receipts:
        op = np.asarray([r["test_rel_rmse_mean"] for r in receipts["operator64"]])
        for other in ("fixed_deep64", "tiny_mlp56", "tiny_mlp74", "fixed_operator44"):
            if other not in receipts:
                continue
            x = np.asarray([r["test_rel_rmse_mean"] for r in receipts[other]])
            d = x - op
            paired[f"operator64_vs_{other}"] = {
                "operator_wins": int(np.sum(d > 0)),
                "pairs": int(len(d)),
                "mean_other_minus_operator": float(d.mean()),
                "median_other_minus_operator": float(np.median(d)),
            }

    metrics = {
        "experiment": "Gate 6b parameter-matched controls",
        "scene": names,
        "train_views": int(len(train_idx)),
        "validation_views": int(len(val_idx)),
        "test_views": int(len(test_idx)),
        "test_z_planes": sorted(set(int(x) for x in cam_ijk[test_idx, 2])),
        "anchors": int(len(anchors)),
        "operator_dim": int(args.operator_dim),
        "summary": summary,
        "paired": paired,
        "receipts": receipts,
    }
    out = Path(args.out_dir)
    path = out / "operator_splat_capacity_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== FINAL ===")
    for mode in modes:
        s = summary[mode]
        print(
            f"{mode:18s} test={s['test_rel_rmse_mean']:.6f} +/- "
            f"{s['test_rel_rmse_std']:.6f} params={s['trainable_params']}"
        )
    for name, p in paired.items():
        print(
            f"{name}: wins={p['operator_wins']}/{p['pairs']} "
            f"other-op={p['mean_other_minus_operator']:+.6f}"
        )
    print(f"wrote {path}")


def selftest():
    q, qn, _, _ = osf.make_anchor_grid(3, 2, 3)
    expected = {
        "operator64": 44 + 12,  # dim=6 -> 4*(6+1)=28 head + 12 material = 40; checked below directly
        "fixed_deep64": None,
        "tiny_mlp56": 56,
        "tiny_mlp74": 74,
    }
    # Exact default-dim parameter controls are the important invariant.
    op10 = make_model("operator64", qn, 10, 0)
    deep10 = make_model("fixed_deep64", qn, 10, 0)
    tiny4 = make_model("tiny_mlp56", qn, 10, 0)
    tiny5 = make_model("tiny_mlp74", qn, 10, 0)
    assert osf.count_params(op10)[1] == 64
    assert osf.count_params(deep10)[1] == 64
    assert osf.count_params(tiny4)[1] == 56
    assert osf.count_params(tiny5)[1] == 74
    for model in (op10, deep10, tiny4, tiny5):
        rgb, alpha = model()
        assert rgb.shape == (len(q), 3)
        assert alpha.shape == (len(q),)
        assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
    print("operator_splat_capacity selftest: PASS (64 vs 64; tiny MLP 56/74)")


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
        choices=[
            "operator64", "fixed_deep64", "tiny_mlp56", "tiny_mlp74",
            "fixed_operator44", "mlp796", "free576",
        ],
        default=["operator64", "fixed_deep64", "tiny_mlp56", "tiny_mlp74"],
    )
    ap.add_argument("--out-dir", default="operator_splat_capacity_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.run:
        run(args); return
    ap.print_help()


if __name__ == "__main__":
    main()
