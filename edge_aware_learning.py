#!/usr/bin/env python3
"""SplatWorld4: image-only edge-aware material-learning gate.

Why this exists
---------------
The first room experiment found two facts at once:

1. changing g with a global image-PCA objective slightly improved held-out image
   reconstruction in 8/8 seeds;
2. the same image training *damaged* a beacon/pillar visibility boundary that was
   already readable from the fixed operator coordinates.

This gate changes only the visual training target.  It gives g NO geometry,
visibility, depth, hit-id, or test-view labels.  Instead of optimizing only
low-order image intensity coefficients, it optimizes a train-only PCA of an
augmented visual field containing intensity plus horizontal/vertical image
finite differences.  Gradient channels are normalized from training data only
so that edges are not numerically drowned by smooth intensity energy.

After material learning, g is frozen and tested two ways:
  * ordinary held-out image reconstruction, using the same normal image-PCA
    decoder target as before;
  * the same held-out beacon-visibility probe as VISIBILITY_RESULTS.md.

No test data selects any material, ridge, seed, edge weight, or PCA dimension.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import room_counterfactual as rw
import visibility_topology as vt
from operator_plate import OperatorPlate

# Fixed before the run.  Each channel is first normalized from training data,
# then smooth intensity is deliberately downweighted relative to derivatives.
INTENSITY_WEIGHT = 0.35
DX_WEIGHT = 1.0
DY_WEIGHT = 1.0


def gradient_channels(images: np.ndarray):
    images = np.asarray(images, np.float64)
    dx = np.zeros_like(images)
    dy = np.zeros_like(images)
    dx[:, :, 1:] = images[:, :, 1:] - images[:, :, :-1]
    dy[:, 1:, :] = images[:, 1:, :] - images[:, :-1, :]
    return dx, dy


def edge_augmented_matrix(train_images: np.ndarray):
    """Return train-only normalized [intensity, dx, dy] vectors + scale receipt."""
    imgs = np.asarray(train_images, np.float64)
    dx, dy = gradient_channels(imgs)
    channels = []
    scales = {}
    for name, arr, weight in (
        ("intensity", imgs, INTENSITY_WEIGHT),
        ("dx", dx, DX_WEIGHT),
        ("dy", dy, DY_WEIGHT),
    ):
        centered = arr - arr.mean(axis=0, keepdims=True)
        rms = float(np.sqrt(np.mean(centered * centered)) + 1e-8)
        scales[name] = {"rms": rms, "weight": float(weight)}
        channels.append(weight * centered.reshape(len(imgs), -1) / rms)
    return np.concatenate(channels, axis=1), scales


def fit_edge_targets(train_images: np.ndarray, k: int):
    X, scales = edge_augmented_matrix(train_images)
    # X is already centered channel-wise; SVD is train-only.
    _U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(int(k), len(train_images) - 1, Vt.shape[0])
    basis = Vt[:k]
    coeff = X @ basis.T
    return coeff, S, scales


def fit_readout_for_frozen_plate(plate, train_poses, target, folds, test_poses, feat_dim=5):
    Ftr = rw.operator_features(plate, train_poses, feat_dim)
    Fte = rw.operator_features(plate, test_poses, feat_dim)
    cv, ridge, W = rw.cv_ridge(Ftr, target, folds)
    return rw._ridge_predict(Fte, W), {"train_cv_mse": float(cv), "ridge": float(ridge)}


def decode_coeffs(mean, basis, coeff, shape):
    return np.stack([rw.decode(mean, basis, c, shape) for c in coeff])


def train_material(train_poses, target, folds, seed, steps):
    plate, _W, info = rw.fit_plate_cv(train_poses, target, folds, seed, feat_dim=5, steps=steps)
    return plate, info


def visibility_from_plate(plate, train, folds, test, y_train):
    pred, info = vt.classify_features(
        rw.operator_features(plate, train, 5), y_train, folds,
        rw.operator_features(plate, test, 5),
    )
    return pred, info


def summarize(a):
    a = np.asarray(a, np.float64)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "min_diagnostic_only": float(np.min(a)),
        "max": float(np.max(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
    }


def run(height=32, width=64, image_basis_dim=12, edge_basis_dim=12, seeds=8, g_steps=700, out_dir="edge_aware_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, folds = rw.training_poses()
    test = rw.test_poses()
    train_images = np.stack([rw.render_pose(p, height, width) for p in train])
    test_images = np.stack([rw.render_pose(p, height, width) for p in test])

    # Normal image target/decoder: exactly the target used in the original room
    # experiment.  It is used to score reconstruction for every frozen plate.
    mean, image_basis, image_coeff, image_singulars = rw.fit_basis(train_images, image_basis_dim)

    # New material-learning target.  No test images enter it.
    edge_coeff, edge_singulars, edge_scales = fit_edge_targets(train_images, edge_basis_dim)

    ytr = vt.labels(train)
    yte = vt.labels(test)
    boundary = np.asarray([rw.occlusion_boundary_distance(p) for p in test], np.float64)
    order = np.argsort(boundary)
    k = max(1, len(test) // 3)
    near = order[:k]
    far = order[-k:]

    fixed_rows = []
    global_rows = []
    edge_rows = []
    fixed_vis_preds = []
    global_vis_preds = []
    edge_vis_preds = []
    fixed_image_errors = []
    global_image_errors = []
    edge_image_errors = []

    for seed in range(int(seeds)):
        fixed = OperatorPlate(dim=6, seed=seed)
        global_plate, global_info = train_material(train, image_coeff, folds, seed, g_steps)
        edge_plate, edge_info = train_material(train, edge_coeff, folds, seed, g_steps)

        row = {}
        for name, plate, info in (
            ("fixed", fixed, {"initial_cv_mse": None, "final_cv_mse": None, "accepted": 0, "material_l2_change": 0.0}),
            ("global", global_plate, global_info),
            ("edge", edge_plate, edge_info),
        ):
            # Ordinary image reconstruction from the frozen representation.
            pred_coeff, recon_info = fit_readout_for_frozen_plate(
                plate, train, image_coeff, folds, test, 5
            )
            pred_img = decode_coeffs(mean, image_basis, pred_coeff, (height, width))
            rel = rw.rel_rmse(pred_img, test_images)
            per_pose_img = rw.image_errors(pred_img, test_images)

            # Visibility is a downstream probe only; it never modifies g.
            vis_pred, vis_info = visibility_from_plate(plate, train, folds, test, ytr)
            vis = vt.classification_metrics(vis_pred, yte)
            near_vis = vt.eval_subset(vis_pred, yte, near)
            far_vis = vt.eval_subset(vis_pred, yte, far)

            rec = {
                "seed": int(seed),
                "image_rel_rmse": float(rel),
                "visibility_accuracy": float(vis["accuracy"]),
                "visibility_balanced_accuracy": float(vis["balanced_accuracy"]),
                "visibility_brier": float(vis["brier"]),
                "near_visibility_brier": float(near_vis["brier"]),
                "far_visibility_brier": float(far_vis["brier"]),
                "image_readout_cv_mse": float(recon_info["train_cv_mse"]),
                "image_readout_ridge": float(recon_info["ridge"]),
                "visibility_readout_cv_mse": float(vis_info["train_cv_mse"]),
                "visibility_readout_ridge": float(vis_info["ridge"]),
                "material_initial_cv_mse": info.get("initial_cv_mse"),
                "material_final_cv_mse": info.get("final_cv_mse"),
                "accepted": int(info.get("accepted", 0)),
                "material_l2_change": float(info.get("material_l2_change", 0.0)),
            }
            row[name] = (rec, vis_pred, per_pose_img)

        fixed_rows.append(row["fixed"][0])
        global_rows.append(row["global"][0])
        edge_rows.append(row["edge"][0])
        fixed_vis_preds.append(row["fixed"][1])
        global_vis_preds.append(row["global"][1])
        edge_vis_preds.append(row["edge"][1])
        fixed_image_errors.append(row["fixed"][2])
        global_image_errors.append(row["global"][2])
        edge_image_errors.append(row["edge"][2])

    fixed_vis_preds = np.asarray(fixed_vis_preds)
    global_vis_preds = np.asarray(global_vis_preds)
    edge_vis_preds = np.asarray(edge_vis_preds)
    fixed_image_errors = np.asarray(fixed_image_errors)
    global_image_errors = np.asarray(global_image_errors)
    edge_image_errors = np.asarray(edge_image_errors)

    def dist(rows, key):
        return summarize([r[key] for r in rows])

    fixed_b = np.asarray([r["visibility_brier"] for r in fixed_rows])
    global_b = np.asarray([r["visibility_brier"] for r in global_rows])
    edge_b = np.asarray([r["visibility_brier"] for r in edge_rows])
    fixed_r = np.asarray([r["image_rel_rmse"] for r in fixed_rows])
    global_r = np.asarray([r["image_rel_rmse"] for r in global_rows])
    edge_r = np.asarray([r["image_rel_rmse"] for r in edge_rows])
    fixed_nb = np.asarray([r["near_visibility_brier"] for r in fixed_rows])
    global_nb = np.asarray([r["near_visibility_brier"] for r in global_rows])
    edge_nb = np.asarray([r["near_visibility_brier"] for r in edge_rows])

    median_fixed_vis = np.median(fixed_vis_preds, axis=0)
    median_global_vis = np.median(global_vis_preds, axis=0)
    median_edge_vis = np.median(edge_vis_preds, axis=0)

    # Per-pose medians let us see whether any recovered visibility behavior is
    # actually concentrated at the true boundary.
    median_fixed_image_error = np.median(fixed_image_errors, axis=0)
    median_global_image_error = np.median(global_image_errors, axis=0)
    median_edge_image_error = np.median(edge_image_errors, axis=0)

    metrics = {
        "question": "Can an image-only edge-sensitive objective improve reconstruction without destroying the fixed operator's held-out visibility partition?",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seeds": int(seeds),
        "g_steps": int(g_steps),
        "image_basis_dim": int(image_basis.shape[0]),
        "edge_basis_dim": int(edge_coeff.shape[1]),
        "edge_target": {
            "intensity_weight": INTENSITY_WEIGHT,
            "dx_weight": DX_WEIGHT,
            "dy_weight": DY_WEIGHT,
            "train_channel_scales": edge_scales,
            "selection_guard": "weights and dimensions fixed before test evaluation; all channel scales are estimated from training images only",
        },
        "fixed": {
            "image_rel_rmse": dist(fixed_rows, "image_rel_rmse"),
            "visibility_brier": dist(fixed_rows, "visibility_brier"),
            "visibility_balanced_accuracy": dist(fixed_rows, "visibility_balanced_accuracy"),
            "near_visibility_brier": dist(fixed_rows, "near_visibility_brier"),
        },
        "global_image_trained": {
            "image_rel_rmse": dist(global_rows, "image_rel_rmse"),
            "visibility_brier": dist(global_rows, "visibility_brier"),
            "visibility_balanced_accuracy": dist(global_rows, "visibility_balanced_accuracy"),
            "near_visibility_brier": dist(global_rows, "near_visibility_brier"),
        },
        "edge_image_trained": {
            "image_rel_rmse": dist(edge_rows, "image_rel_rmse"),
            "visibility_brier": dist(edge_rows, "visibility_brier"),
            "visibility_balanced_accuracy": dist(edge_rows, "visibility_balanced_accuracy"),
            "near_visibility_brier": dist(edge_rows, "near_visibility_brier"),
        },
        "paired_edge_vs_global": {
            "visibility_brier_wins": int(np.sum(edge_b < global_b)),
            "near_visibility_brier_wins": int(np.sum(edge_nb < global_nb)),
            "image_rmse_wins": int(np.sum(edge_r < global_r)),
            "n": int(seeds),
            "median_brier_improvement": float(np.median(global_b - edge_b)),
            "median_near_brier_improvement": float(np.median(global_nb - edge_nb)),
            "median_image_rmse_improvement": float(np.median(global_r - edge_r)),
        },
        "paired_edge_vs_fixed": {
            "visibility_brier_wins": int(np.sum(edge_b < fixed_b)),
            "near_visibility_brier_wins": int(np.sum(edge_nb < fixed_nb)),
            "image_rmse_wins": int(np.sum(edge_r < fixed_r)),
            "n": int(seeds),
            "median_brier_improvement": float(np.median(fixed_b - edge_b)),
            "median_near_brier_improvement": float(np.median(fixed_nb - edge_nb)),
            "median_image_rmse_improvement": float(np.median(fixed_r - edge_r)),
        },
        "median_prediction_visibility": {
            "fixed": vt.classification_metrics(median_fixed_vis, yte),
            "global": vt.classification_metrics(median_global_vis, yte),
            "edge": vt.classification_metrics(median_edge_vis, yte),
            "near_fixed": vt.eval_subset(median_fixed_vis, yte, near),
            "near_global": vt.eval_subset(median_global_vis, yte, near),
            "near_edge": vt.eval_subset(median_edge_vis, yte, near),
            "far_fixed": vt.eval_subset(median_fixed_vis, yte, far),
            "far_global": vt.eval_subset(median_global_vis, yte, far),
            "far_edge": vt.eval_subset(median_edge_vis, yte, far),
        },
        "boundary_distance": {
            "near_mean": float(np.mean(boundary[near])),
            "far_mean": float(np.mean(boundary[far])),
        },
        "image_singular_values": [float(x) for x in image_singulars[:image_basis.shape[0]]],
        "edge_singular_values": [float(x) for x in edge_singulars[:edge_coeff.shape[1]]],
        "fixed_rows": fixed_rows,
        "global_rows": global_rows,
        "edge_rows": edge_rows,
        "guard": "visibility labels never alter g; all material learning is visual and training-only",
    }

    metrics["edge_beats_global_visibility"] = bool(
        metrics["paired_edge_vs_global"]["visibility_brier_wins"] >= int(seeds) - 1
        and np.median(edge_b) < np.median(global_b)
    )
    metrics["edge_beats_global_near_boundary"] = bool(
        metrics["paired_edge_vs_global"]["near_visibility_brier_wins"] >= int(seeds) - 1
        and np.median(edge_nb) < np.median(global_nb)
    )
    metrics["edge_surpasses_fixed_visibility"] = bool(
        np.median(edge_b) < np.median(fixed_b)
        and np.median(edge_nb) < np.median(fixed_nb)
    )
    metrics["edge_improves_image_over_fixed"] = bool(np.median(edge_r) < np.median(fixed_r))

    if (
        metrics["edge_beats_global_visibility"]
        and metrics["edge_beats_global_near_boundary"]
        and metrics["edge_surpasses_fixed_visibility"]
        and metrics["edge_improves_image_over_fixed"]
    ):
        verdict = "EDGE_AWARE_VISUAL_LEARNING_IMPROVES_RECONSTRUCTION_AND_VISIBILITY_TOPOLOGY"
    elif metrics["edge_beats_global_visibility"] and metrics["edge_beats_global_near_boundary"]:
        verdict = "EDGE_AWARE_OBJECTIVE_REPAIRS_PART_OF_VISIBILITY_DAMAGE"
    else:
        verdict = "EDGE_AWARE_OBJECTIVE_DOES_NOT_REPAIR_VISIBILITY_TOPOLOGY"
    metrics["verdict"] = verdict

    (out / "edge_aware_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "edge_aware_data.npz",
        test_poses=test,
        test_visibility=yte,
        boundary_distance=boundary,
        median_fixed_visibility=median_fixed_vis,
        median_global_visibility=median_global_vis,
        median_edge_visibility=median_edge_vis,
        median_fixed_image_error=median_fixed_image_error,
        median_global_image_error=median_global_image_error,
        median_edge_image_error=median_edge_image_error,
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def selftest():
    train, folds = rw.training_poses()
    imgs = np.stack([rw.render_pose(p, 16, 24) for p in train[:10]])
    coeff, _s, scales = fit_edge_targets(imgs, 5)
    assert coeff.shape == (10, 5)
    assert all(scales[k]["rms"] > 0 for k in ("intensity", "dx", "dy"))
    p = OperatorPlate(dim=6, seed=0)
    pred, info = fit_readout_for_frozen_plate(p, train[:10], coeff, np.arange(10) % 2, train[:3])
    assert pred.shape == (3, 5) and np.isfinite(pred).all() and info["ridge"] > 0
    print("edge_aware_learning selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--image-basis-dim", type=int, default=12)
    ap.add_argument("--edge-basis-dim", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--g-steps", type=int, default=700)
    ap.add_argument("--out-dir", default="edge_aware_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.height, args.width, args.image_basis_dim, args.edge_basis_dim, args.seeds, args.g_steps, args.out_dir)


if __name__ == "__main__":
    main()
