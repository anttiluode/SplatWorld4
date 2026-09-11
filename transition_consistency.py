#!/usr/bin/env python3
"""SplatWorld4 Gate 4: relational transition consistency.

The previous route-derivative target failed because it asked one static pose to
encode a traversal-specific derivative.  Here motion is represented as a
relation between TWO substrate states.

Training uses only pairs of observed route views.  For each pair (p0, p1), a
small ridge readout receives

    [ h_g(p0), h_g(p1)-h_g(p0), delta_pose ]

and predicts the change in TRAIN-only image-PCA coefficients from view 0 to
view 1.  Candidate material vectors g are selected only by blocked CV over
training-route transition pairs.

Held-out evaluation uses short interior counterfactual motions.  Every motion
has the same floor-plane step.  Geometry is NEVER supplied to g; only after all
predictions are frozen do we label each test transition as crossing or not
crossing the real pillar/beacon visibility boundary.

The decisive question is therefore local and relational:

    Does transition-trained g help specifically on small motions that cross a
    visibility discontinuity, beyond fixed g, static-edge-trained g, and
    pose-only transition controls?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import room_counterfactual as rw
import room_controls as rc
import edge_aware_learning as ea
from operator_plate import OperatorPlate

ROUTES = ((0, 9), (9, 17), (17, 25))


def pose_delta(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return np.array([
        (b[0] - a[0]) / rw.ROOM,
        (b[1] - a[1]) / rw.ROOM,
        rw.wrap_angle(float(b[2] - a[2])) / math.pi,
    ], dtype=np.float64)


def training_pairs(poses: np.ndarray):
    """Within-block route pairs, both directions, with three blocked CV folds.

    Each route is split into three contiguous blocks.  Pairs never cross a block
    boundary, so a validation fold is a contiguous route segment on all three
    traversals rather than an interleaved random sample.
    """
    pairs = []
    folds = []
    for lo, hi in ROUTES:
        idx = np.arange(lo, hi)
        blocks = np.array_split(idx, 3)
        for fold, block in enumerate(blocks):
            block = list(map(int, block))
            for step in (1, 2):
                for q in range(len(block) - step):
                    i, j = block[q], block[q + step]
                    pairs.append((i, j)); folds.append(fold)
                    pairs.append((j, i)); folds.append(fold)
    return np.asarray(pairs, np.int32), np.asarray(folds, np.int32)


def pair_deltas(poses: np.ndarray, pairs: np.ndarray):
    return np.stack([pose_delta(poses[i], poses[j]) for i, j in pairs])


def pair_features_from_state(H: np.ndarray, poses: np.ndarray, pairs: np.ndarray):
    d = pair_deltas(poses, pairs)
    h0 = H[pairs[:, 0]]
    dh = H[pairs[:, 1]] - h0
    return np.concatenate([h0, dh, d], axis=1)


def operator_pair_features(plate: OperatorPlate, poses: np.ndarray, pairs: np.ndarray, feat_dim=5):
    H = rw.operator_features(plate, poses, feat_dim)
    return pair_features_from_state(H, poses, pairs)


def direct_pair_features(poses: np.ndarray, pairs: np.ndarray):
    H = rw.direct_pose_features(poses)
    return pair_features_from_state(H, poses, pairs)


def random_state_features(poses: np.ndarray, seed: int, feat_dim=5):
    return rw.random_features(poses, seed, feat_dim)


def random_pair_features(poses: np.ndarray, pairs: np.ndarray, seed: int, feat_dim=5):
    H = random_state_features(poses, seed, feat_dim)
    return pair_features_from_state(H, poses, pairs)


def pair_targets(coeff: np.ndarray, pairs: np.ndarray):
    return coeff[pairs[:, 1]] - coeff[pairs[:, 0]]


def fit_transition_plate(train_poses, pairs, targets, folds, seed, steps=700, feat_dim=5):
    rng = np.random.default_rng(seed + 41000)
    plate = OperatorPlate(dim=6, seed=seed)
    initial_g = plate.g.copy()
    total = plate.material_sum

    def score_current():
        F = operator_pair_features(plate, train_poses, pairs, feat_dim)
        score, ridge, _W = rw.cv_ridge(F, targets, folds)
        return score, ridge

    best, best_ridge = score_current()
    initial = float(best)
    accepted = 0
    delta = 0.045
    for step in range(int(steps)):
        old = plate.g.copy()
        if not plate.mutate(rng, delta):
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
    F = operator_pair_features(plate, train_poses, pairs, feat_dim)
    _s, _r, W = rw.cv_ridge(F, targets, folds, ridges=(best_ridge,))
    return plate, W, {
        "initial_pair_cv_mse": initial,
        "final_pair_cv_mse": float(best),
        "ridge": float(best_ridge),
        "accepted": int(accepted),
        "material_l2_change": float(np.linalg.norm(plate.g - initial_g)),
    }


def fit_frozen_transition_readout(plate, train_poses, pairs, targets, folds, feat_dim=5):
    F = operator_pair_features(plate, train_poses, pairs, feat_dim)
    score, ridge, W = rw.cv_ridge(F, targets, folds)
    return W, {"pair_cv_mse": float(score), "ridge": float(ridge)}


def kernel_transition_fit(F: np.ndarray, Y: np.ndarray, folds: np.ndarray):
    best = None
    for gamma in (0.25, 0.5, 1.0, 2.0, 4.0):
        Kall = np.exp(-gamma * rc.sqdist(F, F))
        for ridge in (0.01, 0.1, 1.0, 10.0):
            errs = []
            for fold in sorted(set(map(int, folds))):
                va = folds == fold
                tr = ~va
                K = Kall[np.ix_(tr, tr)]
                alpha = np.linalg.solve(K + ridge * np.eye(np.sum(tr)), Y[tr])
                pred = Kall[np.ix_(va, tr)] @ alpha
                errs.append(np.mean((pred - Y[va]) ** 2))
            score = float(np.mean(errs))
            if best is None or score < best[0]:
                best = (score, float(gamma), float(ridge))
    score, gamma, ridge = best
    K = np.exp(-gamma * rc.sqdist(F, F))
    alpha = np.linalg.solve(K + ridge * np.eye(len(F)), Y)
    return alpha, {"pair_cv_mse": score, "gamma": gamma, "ridge": ridge}


def kernel_transition_predict(F_train, F_test, alpha, gamma):
    return np.exp(-gamma * rc.sqdist(F_test, F_train)) @ alpha


def knn_transition_predict(F_train, Y_train, F_test, k=4):
    out = []
    # standardize from train only so pose, angular and feature coordinates have
    # comparable influence in the local transition lookup.
    mu = F_train.mean(axis=0)
    sd = F_train.std(axis=0) + 1e-6
    A = (F_train - mu) / sd
    B = (F_test - mu) / sd
    for b in B:
        d = np.sqrt(np.sum((A - b) ** 2, axis=1))
        ix = np.argsort(d)[:k]
        w = 1.0 / (d[ix] + 0.12) ** 2
        w /= w.sum()
        out.append(np.sum(Y_train[ix] * w[:, None], axis=0))
    return np.asarray(out)


def _pose_key(p):
    return tuple(round(float(x), 7) for x in p)


def counterfactual_transition_poses(step=0.28):
    """Short same-length interior motions; visibility is NOT used to create them."""
    xs = np.linspace(-1.72, 1.72, 8)
    zs = np.linspace(-1.48, 1.64, 8)
    directions = ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step))
    starts, ends = [], []
    seen = set()
    for z in zs:
        for x in xs:
            if not rw._safe_pose(float(x), float(z)):
                continue
            for dx, dz in directions:
                xx, zz = float(x + dx), float(z + dz)
                if not rw._safe_pose(xx, zz):
                    continue
                p0 = np.array([x, z, rw.look_at(float(x), float(z))], np.float64)
                p1 = np.array([xx, zz, rw.look_at(xx, zz)], np.float64)
                key = (_pose_key(p0), _pose_key(p1))
                if key in seen:
                    continue
                seen.add(key)
                starts.append(p0); ends.append(p1)
    return np.asarray(starts), np.asarray(ends)


def project_images(mean, basis, images):
    return (images.reshape(len(images), -1) - mean) @ basis.T


def pair_feature_from_two_state_arrays(H0, H1, P0, P1):
    d = np.stack([pose_delta(a, b) for a, b in zip(P0, P1)])
    return np.concatenate([H0, H1 - H0, d], axis=1)


def operator_test_pair_features(plate, P0, P1, feat_dim=5):
    H0 = rw.operator_features(plate, P0, feat_dim)
    H1 = rw.operator_features(plate, P1, feat_dim)
    return pair_feature_from_two_state_arrays(H0, H1, P0, P1)


def direct_test_pair_features(P0, P1):
    return pair_feature_from_two_state_arrays(rw.direct_pose_features(P0), rw.direct_pose_features(P1), P0, P1)


def random_test_pair_features(P0, P1, seed, feat_dim=5):
    H0 = random_state_features(P0, seed, feat_dim)
    H1 = random_state_features(P1, seed, feat_dim)
    return pair_feature_from_two_state_arrays(H0, H1, P0, P1)


def decode_next(mean, basis, C0, dC, shape):
    C = C0 + dC
    return np.stack([rw.decode(mean, basis, c, shape) for c in C])


def summarize(a):
    a = np.asarray(a, np.float64)
    return {
        "median": float(np.median(a)), "mean": float(np.mean(a)),
        "min_diagnostic_only": float(np.min(a)), "max": float(np.max(a)),
        "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75)),
    }


def subset_mean(errors, idx):
    return float(np.mean(np.asarray(errors)[idx])) if len(idx) else None


def run(height=32, width=64, basis_dim=12, seeds=8, g_steps=700, out_dir="transition_out"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_poses, _old_folds = rw.training_poses()
    train_images = np.stack([rw.render_pose(p, height, width) for p in train_poses])
    mean, basis, train_coeff, _S = rw.fit_basis(train_images, basis_dim)

    pairs, folds = training_pairs(train_poses)
    targets = pair_targets(train_coeff, pairs)
    F_direct_train = direct_pair_features(train_poses, pairs)

    # Pose-only transition controls, selected using training pairs only.
    dcv, dridge, dW = rw.cv_ridge(F_direct_train, targets, folds)
    kalpha, kinfo = kernel_transition_fit(F_direct_train, targets, folds)

    # Test motions are generated without consulting visibility.
    P0, P1 = counterfactual_transition_poses()
    I0 = np.stack([rw.render_pose(p, height, width) for p in P0])
    I1 = np.stack([rw.render_pose(p, height, width) for p in P1])
    C0 = project_images(mean, basis, I0)
    C1 = project_images(mean, basis, I1)
    true_dC = C1 - C0

    F_direct_test = direct_test_pair_features(P0, P1)
    direct_dC = rw._ridge_predict(F_direct_test, dW)
    kernel_dC = kernel_transition_predict(F_direct_train, F_direct_test, kalpha, kinfo["gamma"])
    knn_dC = knn_transition_predict(F_direct_train, targets, F_direct_test, 4)
    zero_dC = np.zeros_like(true_dC)

    direct_img = decode_next(mean, basis, C0, direct_dC, (height, width))
    kernel_img = decode_next(mean, basis, C0, kernel_dC, (height, width))
    knn_img = decode_next(mean, basis, C0, knn_dC, (height, width))
    zero_img = decode_next(mean, basis, C0, zero_dC, (height, width))
    oracle_img = np.stack([rw.decode(mean, basis, c, (height, width)) for c in C1])

    # Geometry enters only here, after all train-only control fits are frozen.
    v0 = np.asarray([rw.beacon_visible_xy(p[0], p[1]) for p in P0], bool)
    v1 = np.asarray([rw.beacon_visible_xy(p[0], p[1]) for p in P1], bool)
    crossing = v0 != v1
    cross_idx = np.flatnonzero(crossing)
    non_idx = np.flatnonzero(~crossing)
    boundary_score = np.minimum(
        np.asarray([rw.occlusion_boundary_distance(p) for p in P0]),
        np.asarray([rw.occlusion_boundary_distance(p) for p in P1]),
    )
    # Same-size motion is already matched by construction.  For an especially
    # hard non-crossing control, choose the non-crossing transitions closest to
    # the boundary, with the same count as crossing transitions.
    nmatch = min(len(cross_idx), len(non_idx))
    near_non_idx = non_idx[np.argsort(boundary_score[non_idx])[:nmatch]]

    base_errors = {
        "direct_pose": rw.image_errors(direct_img, I1),
        "pose_kernel": rw.image_errors(kernel_img, I1),
        "transition_knn": rw.image_errors(knn_img, I1),
        "no_change": rw.image_errors(zero_img, I1),
        "basis_oracle": rw.image_errors(oracle_img, I1),
    }

    edge_coeff, _edgeS, _edgeReceipt = ea.fit_edge_targets(train_images, basis_dim)

    fixed_rows, edge_rows, transition_rows, random_rows = [], [], [], []
    fixed_cross, edge_cross, transition_cross, random_cross = [], [], [], []
    fixed_all, edge_all, transition_all, random_all = [], [], [], []

    for seed in range(int(seeds)):
        # Fixed operator + transition readout.
        fixed = OperatorPlate(dim=6, seed=seed)
        Wf, infof = fit_frozen_transition_readout(fixed, train_poses, pairs, targets, folds)
        fdc = rw._ridge_predict(operator_test_pair_features(fixed, P0, P1), Wf)
        fimg = decode_next(mean, basis, C0, fdc, (height, width))
        ferr = rw.image_errors(fimg, I1)

        # Static edge-trained material, then the SAME transition readout task.
        edge_plate, edge_info = ea.train_material(train_poses, edge_coeff, np.arange(len(train_poses)) % 4, seed, g_steps)
        We, trans_edge_info = fit_frozen_transition_readout(edge_plate, train_poses, pairs, targets, folds)
        edc = rw._ridge_predict(operator_test_pair_features(edge_plate, P0, P1), We)
        eimg = decode_next(mean, basis, C0, edc, (height, width))
        eerr = rw.image_errors(eimg, I1)

        # Relationally trained material: g itself selected by pair prediction.
        tplate, Wt, tinfo = fit_transition_plate(train_poses, pairs, targets, folds, seed, g_steps)
        tdc = rw._ridge_predict(operator_test_pair_features(tplate, P0, P1), Wt)
        timg = decode_next(mean, basis, C0, tdc, (height, width))
        terr = rw.image_errors(timg, I1)

        # Matched generic smooth state features + identical relational readout.
        Fr = random_pair_features(train_poses, pairs, seed, 5)
        rcv, rridge, rW = rw.cv_ridge(Fr, targets, folds)
        rdc = rw._ridge_predict(random_test_pair_features(P0, P1, seed, 5), rW)
        rimg = decode_next(mean, basis, C0, rdc, (height, width))
        rerr = rw.image_errors(rimg, I1)

        def row(errors, extra):
            return {
                "seed": int(seed),
                "all_mean_pair_rel_rmse": float(np.mean(errors)),
                "crossing_mean_pair_rel_rmse": subset_mean(errors, cross_idx),
                "near_noncrossing_mean_pair_rel_rmse": subset_mean(errors, near_non_idx),
                **extra,
            }

        fixed_rows.append(row(ferr, infof))
        edge_rows.append(row(eerr, {**edge_info, "transition_pair_cv_mse": trans_edge_info["pair_cv_mse"], "transition_ridge": trans_edge_info["ridge"]}))
        transition_rows.append(row(terr, tinfo))
        random_rows.append(row(rerr, {"pair_cv_mse": float(rcv), "ridge": float(rridge)}))
        fixed_cross.append(subset_mean(ferr, cross_idx)); edge_cross.append(subset_mean(eerr, cross_idx)); transition_cross.append(subset_mean(terr, cross_idx)); random_cross.append(subset_mean(rerr, cross_idx))
        fixed_all.append(float(np.mean(ferr))); edge_all.append(float(np.mean(eerr))); transition_all.append(float(np.mean(terr))); random_all.append(float(np.mean(rerr)))

    fixed_cross = np.asarray(fixed_cross); edge_cross = np.asarray(edge_cross); transition_cross = np.asarray(transition_cross); random_cross = np.asarray(random_cross)
    fixed_all = np.asarray(fixed_all); edge_all = np.asarray(edge_all); transition_all = np.asarray(transition_all); random_all = np.asarray(random_all)

    baseline_summary = {}
    for name, err in base_errors.items():
        baseline_summary[name] = {
            "all_mean_pair_rel_rmse": float(np.mean(err)),
            "crossing_mean_pair_rel_rmse": subset_mean(err, cross_idx),
            "near_noncrossing_mean_pair_rel_rmse": subset_mean(err, near_non_idx),
        }

    result = {
        "question": "Does pair-trained g improve small counterfactual transitions specifically when they cross a held-out visibility boundary?",
        "n_train_poses": int(len(train_poses)),
        "n_train_pairs": int(len(pairs)),
        "train_pair_folds": {str(i): int(np.sum(folds == i)) for i in sorted(set(folds.tolist()))},
        "n_test_pairs": int(len(P0)),
        "n_crossing_pairs": int(len(cross_idx)),
        "n_noncrossing_pairs": int(len(non_idx)),
        "n_matched_near_noncrossing_pairs": int(len(near_non_idx)),
        "test_floor_step": 0.28,
        "basis_dim": int(basis.shape[0]),
        "seeds": int(seeds),
        "g_steps": int(g_steps),
        "direct_pose_train": {"pair_cv_mse": float(dcv), "ridge": float(dridge)},
        "pose_kernel_train": kinfo,
        "pose_only_baselines": baseline_summary,
        "fixed_operator": {
            "all": summarize(fixed_all), "crossing": summarize(fixed_cross), "rows": fixed_rows,
        },
        "edge_trained_operator": {
            "all": summarize(edge_all), "crossing": summarize(edge_cross), "rows": edge_rows,
        },
        "transition_trained_operator": {
            "all": summarize(transition_all), "crossing": summarize(transition_cross), "rows": transition_rows,
        },
        "random_smooth_state": {
            "all": summarize(random_all), "crossing": summarize(random_cross), "rows": random_rows,
        },
        "paired_transition_effect_crossing": {
            "beats_fixed_count": int(np.sum(transition_cross < fixed_cross)),
            "beats_edge_count": int(np.sum(transition_cross < edge_cross)),
            "beats_random_count": int(np.sum(transition_cross < random_cross)),
            "n": int(seeds),
            "median_improvement_vs_fixed": float(np.median(fixed_cross - transition_cross)),
            "median_improvement_vs_edge": float(np.median(edge_cross - transition_cross)),
            "median_improvement_vs_random": float(np.median(random_cross - transition_cross)),
        },
        "paired_transition_effect_all": {
            "beats_fixed_count": int(np.sum(transition_all < fixed_all)),
            "beats_edge_count": int(np.sum(transition_all < edge_all)),
            "n": int(seeds),
            "median_improvement_vs_fixed": float(np.median(fixed_all - transition_all)),
            "median_improvement_vs_edge": float(np.median(edge_all - transition_all)),
        },
        "crossing_boundary_distance_mean": float(np.mean(boundary_score[cross_idx])) if len(cross_idx) else None,
        "matched_noncross_boundary_distance_mean": float(np.mean(boundary_score[near_non_idx])) if len(near_non_idx) else None,
        "guard": "visibility is used only after all predictions are frozen to label test pairs; it never selects g, a readout, a seed, or a hyperparameter",
    }

    strongest_pose_cross = min(
        baseline_summary["direct_pose"]["crossing_mean_pair_rel_rmse"],
        baseline_summary["pose_kernel"]["crossing_mean_pair_rel_rmse"],
        baseline_summary["transition_knn"]["crossing_mean_pair_rel_rmse"],
        baseline_summary["no_change"]["crossing_mean_pair_rel_rmse"],
    ) if len(cross_idx) else float("inf")
    result["strongest_pose_or_local_crossing_rmse"] = float(strongest_pose_cross)
    result["transition_beats_fixed_crossing"] = bool(np.sum(transition_cross < fixed_cross) >= int(seeds) - 1)
    result["transition_beats_edge_crossing"] = bool(np.sum(transition_cross < edge_cross) >= int(seeds) - 1)
    result["transition_beats_strongest_pose_crossing"] = bool(np.median(transition_cross) < strongest_pose_cross)
    result["transition_beats_fixed_all"] = bool(np.sum(transition_all < fixed_all) >= int(seeds) - 1)

    if (
        result["transition_beats_fixed_crossing"]
        and result["transition_beats_edge_crossing"]
        and result["transition_beats_strongest_pose_crossing"]
        and result["transition_beats_fixed_all"]
    ):
        verdict = "RELATIONAL_G_SHOWS_COUNTERFACTUAL_BOUNDARY_TRANSITION_SIGNAL"
    elif result["transition_beats_fixed_crossing"] and result["transition_beats_edge_crossing"]:
        verdict = "RELATIONAL_G_IMPROVES_OPERATOR_BOUNDARY_TRANSITIONS_BUT_NOT_STRONGEST_CONTROL"
    else:
        verdict = "RELATIONAL_G_BOUNDARY_TRANSITION_SIGNAL_NOT_SHOWN"
    result["verdict"] = verdict

    (out / "transition_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out / "transition_data.npz",
        train_pairs=pairs, train_pair_folds=folds,
        test_start_poses=P0, test_end_poses=P1,
        crossing=crossing, boundary_score=boundary_score,
        direct_errors=base_errors["direct_pose"], kernel_errors=base_errors["pose_kernel"],
        knn_errors=base_errors["transition_knn"], no_change_errors=base_errors["no_change"],
        oracle_errors=base_errors["basis_oracle"],
    )
    print(json.dumps(result, indent=2))
    return result


def selftest():
    poses, _ = rw.training_poses()
    pairs, folds = training_pairs(poses)
    assert pairs.shape[1] == 2 and len(pairs) >= 30
    assert set(folds.tolist()) == {0, 1, 2}
    Fd = direct_pair_features(poses, pairs)
    assert Fd.shape == (len(pairs), 13)
    p = OperatorPlate(dim=6, seed=0)
    Fo = operator_pair_features(p, poses, pairs)
    assert Fo.shape == (len(pairs), 13) and np.isfinite(Fo).all()
    P0, P1 = counterfactual_transition_poses()
    assert len(P0) == len(P1) and len(P0) >= 40
    cross = np.asarray([rw.beacon_visible_xy(a[0], a[1]) != rw.beacon_visible_xy(b[0], b[1]) for a, b in zip(P0, P1)])
    assert np.sum(cross) >= 2 and np.sum(~cross) >= 20
    steps = np.linalg.norm(P1[:, :2] - P0[:, :2], axis=1)
    assert np.max(np.abs(steps - 0.28)) < 1e-8
    print(f"transition_consistency selftest: PASS ({len(pairs)} train pairs, {len(P0)} test pairs, {int(np.sum(cross))} crossing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--basis-dim", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--g-steps", type=int, default=700)
    ap.add_argument("--out-dir", default="transition_out")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if not args.run:
        ap.error("use --selftest or --run")
    run(args.height, args.width, args.basis_dim, args.seeds, args.g_steps, args.out_dir)


if __name__ == "__main__":
    main()
