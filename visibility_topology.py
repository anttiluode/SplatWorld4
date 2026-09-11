#!/usr/bin/env python3
"""Probe whether the image-trained substrate contains held-out visibility topology.

The operator material is trained ONLY on room-image prediction, exactly as in
room_counterfactual.py.  It is then frozen.  A tiny ridge readout is trained on
training-route labels for one discrete event: is the back-wall beacon occluded
by a pillar?  Counterfactual interior poses are never used to train g or select
classifier hyperparameters.

This asks a sharper question than aggregate image RMSE: did image training bend
the compact operator coordinates into a representation that better respects a
real visibility boundary?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

import room_counterfactual as rw
import room_controls as rc
from operator_plate import OperatorPlate


def labels(poses: np.ndarray) -> np.ndarray:
    return np.asarray([float(rw.beacon_visible_xy(p[0], p[1])) for p in poses], np.float64)


def classify_features(Ftr, ytr, folds, Fte):
    score, ridge, W = rw.cv_ridge(Ftr, ytr[:, None], folds)
    pred = rw._ridge_predict(Fte, W).ravel()
    return pred, {"train_cv_mse": float(score), "ridge": float(ridge)}


def classification_metrics(pred, y):
    pred = np.asarray(pred, np.float64)
    y = np.asarray(y, np.float64)
    hard = pred >= 0.5
    truth = y >= 0.5
    acc = float(np.mean(hard == truth))
    brier = float(np.mean((pred - y) ** 2))
    pos = truth
    neg = ~truth
    tpr = float(np.mean(hard[pos] == truth[pos])) if np.any(pos) else 1.0
    tnr = float(np.mean(hard[neg] == truth[neg])) if np.any(neg) else 1.0
    bal = 0.5 * (tpr + tnr)
    return {"accuracy": acc, "balanced_accuracy": float(bal), "brier": brier}


def eval_subset(pred, y, idx):
    return classification_metrics(np.asarray(pred)[idx], np.asarray(y)[idx])


def run(height=32, width=64, basis_dim=12, seeds=8, g_steps=700, out_dir="visibility_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, folds = rw.training_poses()
    test = rw.test_poses()
    ytr, yte = labels(train), labels(test)
    boundary = np.asarray([rw.occlusion_boundary_distance(p) for p in test])
    order = np.argsort(boundary)
    k = max(1, len(test) // 3)
    near = order[:k]
    far = order[-k:]

    # Recreate the image task using training images only.  These coefficients are
    # the ONLY target used to alter g.
    train_images = np.stack([rw.render_pose(p, height, width) for p in train])
    mean, basis, image_coeff, _S = rw.fit_basis(train_images, basis_dim)

    # Direct compact pose classifier.
    p_pred, p_info = classify_features(rw.direct_pose_features(train), ytr, folds, rw.direct_pose_features(test))

    # Capacity-matched interaction map.
    i_pred, i_info = classify_features(rc.interaction5(train), ytr, folds, rc.interaction5(test))

    # Capacity-matched five-RBF map; gamma/ridge selected on TRAIN visibility CV.
    rbf_pred2d, rbf_info = rc.cv_rbf5(train, ytr[:, None], folds, test)
    rbf_pred = rbf_pred2d.ravel()

    # Strong pose-only kernel classifier/regressor.
    k_pred2d, k_info = rc.kernel_ridge_predict(train, ytr[:, None], folds, test)
    kernel_pred = k_pred2d.ravel()

    fixed_rows, learned_rows, random_rows = [], [], []
    fixed_preds, learned_preds = [], []
    for seed in range(int(seeds)):
        # Fixed operator coordinates.
        base = OperatorPlate(dim=6, seed=seed)
        f0, info0 = classify_features(
            rw.operator_features(base, train, 5), ytr, folds,
            rw.operator_features(base, test, 5),
        )
        m0 = classification_metrics(f0, yte)
        fixed_rows.append({"seed": seed, **m0, **info0})
        fixed_preds.append(f0)

        # Train g ONLY on image coefficients, freeze, then fit visibility readout.
        plate, _imageW, ginfo = rw.fit_plate_cv(train, image_coeff, folds, seed, 5, g_steps)
        f1, info1 = classify_features(
            rw.operator_features(plate, train, 5), ytr, folds,
            rw.operator_features(plate, test, 5),
        )
        m1 = classification_metrics(f1, yte)
        learned_rows.append({"seed": seed, **m1, **info1, "image_g_final_cv_mse": ginfo["final_cv_mse"], "material_l2_change": ginfo["material_l2_change"]})
        learned_preds.append(f1)

        # Matched generic smooth random features.
        fr, rinfo = classify_features(
            rw.random_features(train, seed, 5), ytr, folds,
            rw.random_features(test, seed, 5),
        )
        random_rows.append({"seed": seed, **classification_metrics(fr, yte), **rinfo})

    fixed_preds = np.asarray(fixed_preds)
    learned_preds = np.asarray(learned_preds)
    med_fixed_pred = np.median(fixed_preds, axis=0)
    med_learned_pred = np.median(learned_preds, axis=0)

    def distribution(rows, key):
        a = np.asarray([r[key] for r in rows], np.float64)
        return {"median": float(np.median(a)), "mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a))}

    paired_brier_gain = np.asarray([a["brier"] - b["brier"] for a, b in zip(fixed_rows, learned_rows)])
    paired_acc_gain = np.asarray([b["balanced_accuracy"] - a["balanced_accuracy"] for a, b in zip(fixed_rows, learned_rows)])

    methods = {
        "direct_pose": p_pred,
        "interaction5": i_pred,
        "rbf5": rbf_pred,
        "kernel_ridge": kernel_pred,
        "operator_fixed_median_prediction": med_fixed_pred,
        "operator_learned_median_prediction": med_learned_pred,
    }
    method_metrics = {name: classification_metrics(pred, yte) for name, pred in methods.items()}
    near_metrics = {name: eval_subset(pred, yte, near) for name, pred in methods.items()}
    far_metrics = {name: eval_subset(pred, yte, far) for name, pred in methods.items()}

    metrics = {
        "question": "Does image-trained g encode a held-out pillar/beacon visibility boundary?",
        "n_train": int(len(train)), "n_test": int(len(test)), "seeds": int(seeds),
        "train_visible": int(np.sum(ytr)), "train_hidden": int(len(ytr) - np.sum(ytr)),
        "test_visible": int(np.sum(yte)), "test_hidden": int(len(yte) - np.sum(yte)),
        "near_boundary_n": int(len(near)), "far_boundary_n": int(len(far)),
        "near_boundary_mean_distance": float(np.mean(boundary[near])),
        "far_boundary_mean_distance": float(np.mean(boundary[far])),
        "methods": method_metrics,
        "near_boundary": near_metrics,
        "far_boundary": far_metrics,
        "direct_pose_train_info": p_info,
        "interaction5_train_info": i_info,
        "rbf5_train_info": rbf_info,
        "kernel_train_info": k_info,
        "fixed_operator_distribution": {
            "balanced_accuracy": distribution(fixed_rows, "balanced_accuracy"),
            "brier": distribution(fixed_rows, "brier"),
        },
        "image_trained_operator_distribution": {
            "balanced_accuracy": distribution(learned_rows, "balanced_accuracy"),
            "brier": distribution(learned_rows, "brier"),
        },
        "random5_distribution": {
            "balanced_accuracy": distribution(random_rows, "balanced_accuracy"),
            "brier": distribution(random_rows, "brier"),
        },
        "paired_image_training_effect": {
            "brier_wins": int(np.sum(paired_brier_gain > 0)),
            "balanced_accuracy_wins": int(np.sum(paired_acc_gain > 0)),
            "n": int(seeds),
            "median_brier_improvement": float(np.median(paired_brier_gain)),
            "median_balanced_accuracy_improvement": float(np.median(paired_acc_gain)),
        },
        "fixed_rows": fixed_rows,
        "learned_rows": learned_rows,
        "random_rows": random_rows,
        "guard": "g never sees visibility labels during material learning; only the final small readout sees route visibility labels. Test poses remain untouched.",
    }

    learned = method_metrics["operator_learned_median_prediction"]
    fixed = method_metrics["operator_fixed_median_prediction"]
    kernel = method_metrics["kernel_ridge"]
    near_l = near_metrics["operator_learned_median_prediction"]
    near_f = near_metrics["operator_fixed_median_prediction"]
    metrics["image_training_improves_visibility_brier"] = bool(learned["brier"] < fixed["brier"])
    metrics["image_training_improves_near_boundary_brier"] = bool(near_l["brier"] < near_f["brier"])
    metrics["learned_operator_beats_kernel_brier"] = bool(learned["brier"] < kernel["brier"])
    metrics["verdict"] = (
        "IMAGE_TRAINED_G_CONTAINS_VISIBILITY_TOPOLOGY_SIGNAL"
        if metrics["image_training_improves_visibility_brier"]
        and metrics["image_training_improves_near_boundary_brier"]
        and metrics["paired_image_training_effect"]["brier_wins"] >= int(seeds) - 1
        else "VISIBILITY_TOPOLOGY_SIGNAL_NOT_SHOWN"
    )

    (out / "visibility_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "visibility_data.npz",
        train_poses=train, test_poses=test, train_labels=ytr, test_labels=yte,
        boundary_distance=boundary, median_fixed_prediction=med_fixed_pred,
        median_learned_prediction=med_learned_pred, kernel_prediction=kernel_pred,
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def selftest():
    tr, _ = rw.training_poses(); te = rw.test_poses()
    a, b = labels(tr), labels(te)
    assert set(np.unique(a)).issubset({0.0, 1.0})
    assert np.any(a == 0) and np.any(a == 1)
    assert np.any(b == 0) and np.any(b == 1)
    print("visibility_topology selftest: PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--basis-dim", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--g-steps", type=int, default=700)
    ap.add_argument("--out-dir", default="visibility_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.height, args.width, args.basis_dim, args.seeds, args.g_steps, args.out_dir)


if __name__ == "__main__":
    main()
