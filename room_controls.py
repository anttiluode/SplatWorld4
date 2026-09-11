#!/usr/bin/env python3
"""Stronger controls for the SplatWorld4 counterfactual-room experiment.

The first room run produced a small paired improvement from learned g.  This file
attacks that result rather than promoting it.  It adds:

* a capacity-matched 5-RBF pose feature map with centers chosen from training;
* a capacity-matched nonlinear pose/interactions map;
* full kernel-ridge pose regression as a deliberately stronger smooth baseline;
* multi-seed per-pose fixed-vs-learned operator errors;
* direct measurement of whether the learned-g gain is concentrated near the
  geometric occlusion boundary.

No test pose is used to choose centers, widths, ridge, g, or seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import room_counterfactual as rw
from operator_plate import OperatorPlate


def farthest_centers(train_poses: np.ndarray, k=5) -> np.ndarray:
    X = rw.pose_core(train_poses)
    mean = X.mean(axis=0)
    chosen = [int(np.argmin(np.sum((X - mean) ** 2, axis=1)))]
    while len(chosen) < min(k, len(X)):
        C = X[chosen]
        d2 = np.min(np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)
        d2[chosen] = -1.0
        chosen.append(int(np.argmax(d2)))
    return X[chosen]


def rbf_features(poses: np.ndarray, centers: np.ndarray, gamma: float) -> np.ndarray:
    X = rw.pose_core(poses)
    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.exp(-float(gamma) * d2)


def cv_rbf5(train_poses, Y, folds, test_poses):
    centers = farthest_centers(train_poses, 5)
    best = None
    for gamma in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        F = rbf_features(train_poses, centers, gamma)
        score, ridge, _W = rw.cv_ridge(F, Y, folds)
        if best is None or score < best[0]:
            best = (score, gamma, ridge)
    score, gamma, ridge = best
    Ftr = rbf_features(train_poses, centers, gamma)
    Fte = rbf_features(test_poses, centers, gamma)
    W = rw._ridge_fit(Ftr, Y, ridge)
    return rw._ridge_predict(Fte, W), {
        "train_cv_mse": float(score), "gamma": float(gamma), "ridge": float(ridge),
        "centers": centers.tolist(),
    }


def interaction5(poses: np.ndarray) -> np.ndarray:
    p = np.asarray(poses, np.float64)
    x = p[:, 0] / rw.ROOM
    z = p[:, 1] / rw.ROOM
    y = p[:, 2]
    return np.column_stack([x, z, np.sin(y), np.cos(y), x * z])


def sqdist(A, B):
    return np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)


def kernel_ridge_predict(train_poses, Y, folds, test_poses):
    X = rw.pose_core(train_poses)
    XT = rw.pose_core(test_poses)
    best = None
    for gamma in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        Kall = np.exp(-gamma * sqdist(X, X))
        for ridge in (0.01, 0.1, 1.0, 10.0):
            errs = []
            for f in sorted(set(map(int, folds))):
                va = folds == f
                tr = ~va
                K = Kall[np.ix_(tr, tr)]
                alpha = np.linalg.solve(K + ridge * np.eye(np.sum(tr)), Y[tr])
                pred = Kall[np.ix_(va, tr)] @ alpha
                errs.append(np.mean((pred - Y[va]) ** 2))
            s = float(np.mean(errs))
            if best is None or s < best[0]:
                best = (s, float(gamma), float(ridge))
    score, gamma, ridge = best
    K = np.exp(-gamma * sqdist(X, X))
    alpha = np.linalg.solve(K + ridge * np.eye(len(X)), Y)
    KT = np.exp(-gamma * sqdist(XT, X))
    return KT @ alpha, {"train_cv_mse": score, "gamma": gamma, "ridge": ridge}


def coeff_to_images(mean, basis, C, shape):
    return np.stack([rw.decode(mean, basis, c, shape) for c in C])


def run(height=32, width=64, basis_dim=12, seeds=8, g_steps=700, out_dir="room_controls_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, folds = rw.training_poses()
    test = rw.test_poses()
    train_images = np.stack([rw.render_pose(p, height, width) for p in train])
    test_images = np.stack([rw.render_pose(p, height, width) for p in test])
    mean, basis, Y, _S = rw.fit_basis(train_images, basis_dim)
    true_test_coeff = (test_images.reshape(len(test), -1) - mean) @ basis.T
    oracle = coeff_to_images(mean, basis, true_test_coeff, (height, width))

    # Capacity-matched nonlinear pose interactions (5 features + intercept).
    Fi_tr = interaction5(train)
    Fi_te = interaction5(test)
    icv, iridge, iW = rw.cv_ridge(Fi_tr, Y, folds)
    iC = rw._ridge_predict(Fi_te, iW)
    iImg = coeff_to_images(mean, basis, iC, (height, width))

    # Five RBFs + intercept, same feature count as the operator readout.
    rbfC, rbf_info = cv_rbf5(train, Y, folds, test)
    rbfImg = coeff_to_images(mean, basis, rbfC, (height, width))

    # Strong smooth pose-only baseline with one kernel center per training pose.
    kC, k_info = kernel_ridge_predict(train, Y, folds, test)
    kImg = coeff_to_images(mean, basis, kC, (height, width))

    fixed_global, learned_global = [], []
    fixed_per_pose, learned_per_pose = [], []
    paired_rows = []
    for seed in range(int(seeds)):
        base = OperatorPlate(dim=6, seed=seed)
        Ftr = rw.operator_features(base, train, 5)
        Fte = rw.operator_features(base, test, 5)
        cv0, ridge0, W0 = rw.cv_ridge(Ftr, Y, folds)
        C0 = rw._ridge_predict(Fte, W0)
        I0 = coeff_to_images(mean, basis, C0, (height, width))
        e0 = rw.image_errors(I0, test_images)

        plate, W1, info = rw.fit_plate_cv(train, Y, folds, seed, 5, g_steps)
        F1 = rw.operator_features(plate, test, 5)
        C1 = rw._ridge_predict(F1, W1)
        I1 = coeff_to_images(mean, basis, C1, (height, width))
        e1 = rw.image_errors(I1, test_images)

        g0, g1 = rw.rel_rmse(I0, test_images), rw.rel_rmse(I1, test_images)
        fixed_global.append(g0)
        learned_global.append(g1)
        fixed_per_pose.append(e0)
        learned_per_pose.append(e1)
        paired_rows.append({
            "seed": seed,
            "fixed_rel_rmse": g0,
            "learned_rel_rmse": g1,
            "delta_fixed_minus_learned": g0 - g1,
            "initial_cv_mse": cv0,
            "learned_final_cv_mse": info["final_cv_mse"],
            "accepted": info["accepted"],
            "material_l2_change": info["material_l2_change"],
        })

    fixed_per_pose = np.asarray(fixed_per_pose)
    learned_per_pose = np.asarray(learned_per_pose)
    med_fixed_pose = np.median(fixed_per_pose, axis=0)
    med_learned_pose = np.median(learned_per_pose, axis=0)
    gain_pose = med_fixed_pose - med_learned_pose
    boundary = np.asarray([rw.occlusion_boundary_distance(p) for p in test])

    pose_F = rw.direct_pose_features(train)
    p_cv, p_ridge, pW = rw.cv_ridge(pose_F, Y, folds)
    pC = rw._ridge_predict(rw.direct_pose_features(test), pW)
    pImg = coeff_to_images(mean, basis, pC, (height, width))
    knnImg = coeff_to_images(mean, basis, rw.local_knn(train, Y, test), (height, width))

    per_methods = {
        "direct_pose": rw.image_errors(pImg, test_images),
        "interaction5": rw.image_errors(iImg, test_images),
        "rbf5": rw.image_errors(rbfImg, test_images),
        "kernel_ridge": rw.image_errors(kImg, test_images),
        "operator_fixed_median_seed": med_fixed_pose,
        "operator_learned_median_seed": med_learned_pose,
        "local_knn": rw.image_errors(knnImg, test_images),
        "basis_oracle": rw.image_errors(oracle, test_images),
    }
    bins = rw.summarize_bins(boundary, per_methods)
    gain_bins = rw.summarize_bins(boundary, {"fixed_minus_learned_gain": gain_pose})
    corr = float(np.corrcoef(boundary, gain_pose)[0, 1]) if np.std(boundary) > 1e-12 else 0.0

    fixed_global = np.asarray(fixed_global)
    learned_global = np.asarray(learned_global)
    paired_delta = fixed_global - learned_global
    metrics = {
        "purpose": "attack the provisional learned-g room advantage with stronger pose-only controls",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seeds": int(seeds),
        "direct_pose": {"rel_rmse": rw.rel_rmse(pImg, test_images), "train_cv_mse": p_cv, "ridge": p_ridge},
        "interaction5": {"rel_rmse": rw.rel_rmse(iImg, test_images), "train_cv_mse": icv, "ridge": iridge},
        "rbf5": {"rel_rmse": rw.rel_rmse(rbfImg, test_images), **rbf_info},
        "kernel_ridge": {"rel_rmse": rw.rel_rmse(kImg, test_images), **k_info},
        "local_knn": {"rel_rmse": rw.rel_rmse(knnImg, test_images)},
        "basis_oracle": {"rel_rmse": rw.rel_rmse(oracle, test_images)},
        "operator_fixed": {
            "median": float(np.median(fixed_global)), "mean": float(np.mean(fixed_global)),
            "min_diagnostic_only": float(np.min(fixed_global)), "max": float(np.max(fixed_global)),
        },
        "operator_learned": {
            "median": float(np.median(learned_global)), "mean": float(np.mean(learned_global)),
            "min_diagnostic_only": float(np.min(learned_global)), "max": float(np.max(learned_global)),
        },
        "paired_g_effect": {
            "wins": int(np.sum(paired_delta > 0)),
            "n": int(len(paired_delta)),
            "median_absolute_improvement": float(np.median(paired_delta)),
            "median_relative_improvement": float(np.median(paired_delta / fixed_global)),
            "rows": paired_rows,
        },
        "occlusion_distance_bins": bins,
        "g_gain_by_occlusion_distance": gain_bins,
        "corr_boundary_distance_vs_g_gain": corr,
        "occlusion_specificity_guard": "If g encodes useful occlusion geometry, its advantage should be strongest near boundaries; a gain that grows far from boundaries is evidence against that interpretation.",
        "test_selection_guard": "No model, width, ridge, g, or seed is selected from counterfactual test error.",
    }

    learned_med = metrics["operator_learned"]["median"]
    strongest_pose = min(metrics["direct_pose"]["rel_rmse"], metrics["interaction5"]["rel_rmse"], metrics["rbf5"]["rel_rmse"], metrics["kernel_ridge"]["rel_rmse"], metrics["local_knn"]["rel_rmse"])
    metrics["strongest_pose_only_rel_rmse"] = float(strongest_pose)
    metrics["learned_operator_beats_strongest_pose_only"] = bool(learned_med < strongest_pose)
    near_gain = gain_bins["near"]["errors"]["fixed_minus_learned_gain"]
    far_gain = gain_bins["far"]["errors"]["fixed_minus_learned_gain"]
    metrics["g_gain_stronger_near_than_far"] = bool(near_gain > far_gain)
    metrics["verdict"] = (
        "ROBUST_ROOM_GEOMETRY_SIGNAL_SURVIVES_CONTROLS"
        if metrics["learned_operator_beats_strongest_pose_only"] and metrics["g_gain_stronger_near_than_far"]
        else "PROVISIONAL_ROOM_SIGNAL_DOES_NOT_SURVIVE_STRONG_INTERPRETATION"
    )

    (out / "controls_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "controls_data.npz",
        test_poses=test,
        boundary_distance=boundary,
        median_fixed_error=med_fixed_pose,
        median_learned_error=med_learned_pose,
        learned_gain=gain_pose,
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def selftest():
    train, folds = rw.training_poses()
    test = rw.test_poses()
    centers = farthest_centers(train, 5)
    assert centers.shape == (5, 3)
    F = rbf_features(test[:3], centers, 2.0)
    assert F.shape == (3, 5) and np.isfinite(F).all()
    I = interaction5(test[:3])
    assert I.shape == (3, 5)
    print("room_controls selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--basis-dim", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--g-steps", type=int, default=700)
    ap.add_argument("--out-dir", default="room_controls_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.height, args.width, args.basis_dim, args.seeds, args.g_steps, args.out_dir)


if __name__ == "__main__":
    main()
