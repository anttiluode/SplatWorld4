#!/usr/bin/env python3
"""SplatWorld4 Gate 3: learn g from visual change along observed routes.

Static edge training repaired most of the visibility damage caused by a global
image objective but still failed to beat the fixed substrate at the true
occlusion boundary.  This gate uses a different visual fact:

    occlusion is not merely an edge in one frame;
    it is structure that appears/disappears as viewpoint changes.

No geometry or visibility labels are given to g.  The material-learning target
is built only from the ordered TRAINING-route images and contains:

  intensity, spatial dx, spatial dy,
  signed route-view derivative dt,
  absolute route-view change |dt|.

Every channel is centered and RMS-normalized from training data only.  Weights
are fixed before test evaluation.  After material learning, g is frozen and
scored on ordinary counterfactual image reconstruction and the unchanged
beacon-visibility probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import room_counterfactual as rw
import visibility_topology as vt
import edge_aware_learning as ea
from operator_plate import OperatorPlate

# Fixed design, not selected on test data.
WEIGHTS = {
    "intensity": 0.20,
    "dx": 0.50,
    "dy": 0.50,
    "dt": 1.00,
    "abs_dt": 1.00,
}
ROUTES = ((0, 9), (9, 17), (17, 25))


def route_derivative(images: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Finite derivative of image wrt traveled floor-plane distance on each route."""
    images = np.asarray(images, np.float64)
    poses = np.asarray(poses, np.float64)
    out = np.zeros_like(images)
    for lo, hi in ROUTES:
        assert hi <= len(images)
        for i in range(lo, hi):
            if i == lo:
                j0, j1 = i, i + 1
            elif i == hi - 1:
                j0, j1 = i - 1, i
            else:
                j0, j1 = i - 1, i + 1
            ds = float(np.linalg.norm(poses[j1, :2] - poses[j0, :2])) + 1e-8
            out[i] = (images[j1] - images[j0]) / ds
    return out


def change_augmented_matrix(train_images: np.ndarray, train_poses: np.ndarray):
    imgs = np.asarray(train_images, np.float64)
    dx, dy = ea.gradient_channels(imgs)
    dt = route_derivative(imgs, train_poses)
    channels = (
        ("intensity", imgs),
        ("dx", dx),
        ("dy", dy),
        ("dt", dt),
        ("abs_dt", np.abs(dt)),
    )
    blocks = []
    receipt = {}
    for name, arr in channels:
        centered = arr - arr.mean(axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(centered * centered)) + 1e-8)
        weight = float(WEIGHTS[name])
        receipt[name] = {"rms": rms, "weight": weight}
        blocks.append(weight * centered.reshape(len(imgs), -1) / rms)
    return np.concatenate(blocks, axis=1), receipt


def fit_change_targets(images: np.ndarray, poses: np.ndarray, k: int):
    X, receipt = change_augmented_matrix(images, poses)
    _U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(int(k), len(images) - 1, Vt.shape[0])
    basis = Vt[:k]
    return X @ basis.T, S, receipt


def frozen_eval(plate, train, folds, test, image_coeff, mean, image_basis, shape, ytr, yte, near, far):
    pred_coeff, image_info = ea.fit_readout_for_frozen_plate(
        plate, train, image_coeff, folds, test, feat_dim=5
    )
    pred_img = ea.decode_coeffs(mean, image_basis, pred_coeff, shape)
    vis_pred, vis_info = ea.visibility_from_plate(plate, train, folds, test, ytr)
    vis = vt.classification_metrics(vis_pred, yte)
    return {
        "image_rel_rmse": float(rw.rel_rmse(pred_img, test_images=frozen_eval.test_images) if False else rw.rel_rmse(pred_img, frozen_eval.test_images)),
        "image_per_pose": rw.image_errors(pred_img, frozen_eval.test_images),
        "visibility_prediction": vis_pred,
        "visibility_brier": float(vis["brier"]),
        "visibility_balanced_accuracy": float(vis["balanced_accuracy"]),
        "near_visibility_brier": float(vt.eval_subset(vis_pred, yte, near)["brier"]),
        "far_visibility_brier": float(vt.eval_subset(vis_pred, yte, far)["brier"]),
        "image_readout_cv_mse": float(image_info["train_cv_mse"]),
        "visibility_readout_cv_mse": float(vis_info["train_cv_mse"]),
    }


# `frozen_eval` receives test images through this function attribute to keep its
# call signature compact without ever making test images available to training.
frozen_eval.test_images = None


def summarize(vals):
    a = np.asarray(vals, np.float64)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "min_diagnostic_only": float(np.min(a)),
        "max": float(np.max(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def run(height=32, width=64, image_basis_dim=12, target_dim=12, seeds=8, g_steps=700, out_dir="view_change_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, folds = rw.training_poses()
    test = rw.test_poses()
    train_images = np.stack([rw.render_pose(p, height, width) for p in train])
    test_images = np.stack([rw.render_pose(p, height, width) for p in test])
    frozen_eval.test_images = test_images

    mean, image_basis, image_coeff, _ = rw.fit_basis(train_images, image_basis_dim)
    edge_coeff, _edgeS, edge_receipt = ea.fit_edge_targets(train_images, target_dim)
    change_coeff, changeS, change_receipt = fit_change_targets(train_images, train, target_dim)

    ytr, yte = vt.labels(train), vt.labels(test)
    boundary = np.asarray([rw.occlusion_boundary_distance(p) for p in test], np.float64)
    order = np.argsort(boundary)
    k = max(1, len(test) // 3)
    near, far = order[:k], order[-k:]

    methods = {name: [] for name in ("fixed", "global", "edge", "change")}
    preds = {name: [] for name in methods}
    per_pose_image = {name: [] for name in methods}

    for seed in range(int(seeds)):
        fixed = OperatorPlate(dim=6, seed=seed)
        global_plate, global_info = ea.train_material(train, image_coeff, folds, seed, g_steps)
        edge_plate, edge_info = ea.train_material(train, edge_coeff, folds, seed, g_steps)
        change_plate, change_info = ea.train_material(train, change_coeff, folds, seed, g_steps)

        for name, plate, info in (
            ("fixed", fixed, {"accepted": 0, "material_l2_change": 0.0, "initial_cv_mse": None, "final_cv_mse": None}),
            ("global", global_plate, global_info),
            ("edge", edge_plate, edge_info),
            ("change", change_plate, change_info),
        ):
            ev = frozen_eval(
                plate, train, folds, test, image_coeff, mean, image_basis,
                (height, width), ytr, yte, near, far
            )
            row = {
                "seed": int(seed),
                "image_rel_rmse": ev["image_rel_rmse"],
                "visibility_brier": ev["visibility_brier"],
                "visibility_balanced_accuracy": ev["visibility_balanced_accuracy"],
                "near_visibility_brier": ev["near_visibility_brier"],
                "far_visibility_brier": ev["far_visibility_brier"],
                "accepted": int(info.get("accepted", 0)),
                "material_l2_change": float(info.get("material_l2_change", 0.0)),
                "material_initial_cv_mse": info.get("initial_cv_mse"),
                "material_final_cv_mse": info.get("final_cv_mse"),
            }
            methods[name].append(row)
            preds[name].append(ev["visibility_prediction"])
            per_pose_image[name].append(ev["image_per_pose"])

    def d(name, key):
        return summarize([r[key] for r in methods[name]])

    for name in preds:
        preds[name] = np.asarray(preds[name])
        per_pose_image[name] = np.asarray(per_pose_image[name])

    fixed_b = np.asarray([r["visibility_brier"] for r in methods["fixed"]])
    edge_b = np.asarray([r["visibility_brier"] for r in methods["edge"]])
    change_b = np.asarray([r["visibility_brier"] for r in methods["change"]])
    fixed_nb = np.asarray([r["near_visibility_brier"] for r in methods["fixed"]])
    edge_nb = np.asarray([r["near_visibility_brier"] for r in methods["edge"]])
    change_nb = np.asarray([r["near_visibility_brier"] for r in methods["change"]])
    fixed_r = np.asarray([r["image_rel_rmse"] for r in methods["fixed"]])
    edge_r = np.asarray([r["image_rel_rmse"] for r in methods["edge"]])
    change_r = np.asarray([r["image_rel_rmse"] for r in methods["change"]])

    median_pred = {name: np.median(preds[name], axis=0) for name in preds}
    med_vis = {name: vt.classification_metrics(median_pred[name], yte) for name in preds}
    med_near = {name: vt.eval_subset(median_pred[name], yte, near) for name in preds}
    med_far = {name: vt.eval_subset(median_pred[name], yte, far) for name in preds}

    result = {
        "question": "Can route-view change, learned from images only, sharpen visibility topology at a held-out occlusion boundary?",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seeds": int(seeds),
        "g_steps": int(g_steps),
        "target_dim": int(target_dim),
        "change_target": {
            "weights": WEIGHTS,
            "train_channel_scales": change_receipt,
            "route_slices": [list(x) for x in ROUTES],
            "guard": "route ordering and channel weights are fixed before test evaluation; only training images create dt",
        },
        "edge_target_receipt": edge_receipt,
        "distributions": {
            name: {
                "image_rel_rmse": d(name, "image_rel_rmse"),
                "visibility_brier": d(name, "visibility_brier"),
                "visibility_balanced_accuracy": d(name, "visibility_balanced_accuracy"),
                "near_visibility_brier": d(name, "near_visibility_brier"),
                "far_visibility_brier": d(name, "far_visibility_brier"),
            }
            for name in methods
        },
        "paired_change_vs_edge": {
            "global_visibility_wins": int(np.sum(change_b < edge_b)),
            "near_boundary_wins": int(np.sum(change_nb < edge_nb)),
            "image_rmse_wins": int(np.sum(change_r < edge_r)),
            "n": int(seeds),
            "median_visibility_brier_improvement": float(np.median(edge_b - change_b)),
            "median_near_brier_improvement": float(np.median(edge_nb - change_nb)),
            "median_image_rmse_improvement": float(np.median(edge_r - change_r)),
        },
        "paired_change_vs_fixed": {
            "global_visibility_wins": int(np.sum(change_b < fixed_b)),
            "near_boundary_wins": int(np.sum(change_nb < fixed_nb)),
            "image_rmse_wins": int(np.sum(change_r < fixed_r)),
            "n": int(seeds),
            "median_visibility_brier_improvement": float(np.median(fixed_b - change_b)),
            "median_near_brier_improvement": float(np.median(fixed_nb - change_nb)),
            "median_image_rmse_improvement": float(np.median(fixed_r - change_r)),
        },
        "median_prediction": {
            "global": med_vis,
            "near": med_near,
            "far": med_far,
        },
        "boundary_distance": {
            "near_mean": float(np.mean(boundary[near])),
            "far_mean": float(np.mean(boundary[far])),
        },
        "change_singular_values": [float(x) for x in changeS[:target_dim]],
        "rows": methods,
        "guard": "g never receives visibility labels or test images; visibility is a frozen downstream probe",
    }

    result["change_beats_edge_near"] = bool(
        result["paired_change_vs_edge"]["near_boundary_wins"] >= int(seeds) - 1
        and np.median(change_nb) < np.median(edge_nb)
    )
    result["change_beats_fixed_near"] = bool(
        result["paired_change_vs_fixed"]["near_boundary_wins"] >= int(seeds) - 1
        and np.median(change_nb) < np.median(fixed_nb)
    )
    result["change_improves_image_vs_fixed"] = bool(np.median(change_r) < np.median(fixed_r))
    result["change_beats_edge_global_visibility"] = bool(np.median(change_b) < np.median(edge_b))

    if result["change_beats_edge_near"] and result["change_beats_fixed_near"] and result["change_improves_image_vs_fixed"]:
        verdict = "VIEW_CHANGE_OBJECTIVE_SHARPENS_HELDOUT_OCCLUSION_TOPOLOGY"
    elif result["change_beats_edge_near"]:
        verdict = "VIEW_CHANGE_OBJECTIVE_IMPROVES_BOUNDARY_BUT_NOT_BEYOND_FIXED"
    else:
        verdict = "VIEW_CHANGE_OBJECTIVE_DOES_NOT_IMPROVE_BOUNDARY"
    result["verdict"] = verdict

    (out / "view_change_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "view_change_data.npz",
        test_poses=test,
        test_visibility=yte,
        boundary_distance=boundary,
        **{f"median_{name}_visibility": median_pred[name] for name in median_pred},
        **{f"median_{name}_image_error": np.median(per_pose_image[name], axis=0) for name in per_pose_image},
    )
    print(json.dumps(result, indent=2))
    return result


def selftest():
    train, _folds = rw.training_poses()
    imgs = np.stack([rw.render_pose(p, 16, 24) for p in train])
    dt = route_derivative(imgs, train)
    assert dt.shape == imgs.shape and np.isfinite(dt).all()
    assert np.mean(np.abs(dt)) > 1e-4
    coeff, _s, receipt = fit_change_targets(imgs, train, 6)
    assert coeff.shape == (len(train), 6)
    assert receipt["dt"]["rms"] > 0 and receipt["abs_dt"]["rms"] > 0
    print("view_change_learning selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--image-basis-dim", type=int, default=12)
    ap.add_argument("--target-dim", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--g-steps", type=int, default=700)
    ap.add_argument("--out-dir", default="view_change_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.height, args.width, args.image_basis_dim, args.target_dim, args.seeds, args.g_steps, args.out_dir)


if __name__ == "__main__":
    main()
