#!/usr/bin/env python3
"""Gate 7A: active inverse updates in the world-space operator splat field.

A hidden material edit changes one shared operator field. The solver starts from
the old material, receives only bounded features from cameras it has selected,
measures local response Jacobians by finite differences, and uses damped least
squares to update the material estimate. Active acquisition chooses the next
camera that best exposes the current null/weak material subspace.

The hidden edit is generated inside the same material family. This is a
positive-control inverse experiment, not yet arbitrary moved-mesh recovery.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import obj_xyz_world as ow
import operator_splat_field as osf
from active_inverse_core import (
    damped_least_squares,
    material_tangent_basis,
    select_active_view,
    svd_diagnostics,
)
from xyz_operator_train import image_rel_rmse, seed_all


@dataclass
class InverseProblem:
    model: torch.nn.Module
    base_theta: np.ndarray
    basis: np.ndarray
    kernels: torch.Tensor
    order: torch.Tensor
    projection: np.ndarray
    truth_z: np.ndarray
    truth_features: np.ndarray
    truth_images: np.ndarray
    baseline_images: np.ndarray
    camera_views: int
    truth_change_rel_rmse: float
    device: str


def make_sensor_projection(pixel_dim: int, feature_dim: int, seed: int) -> np.ndarray:
    """Fixed orthonormal linear sensor used to bound information per camera."""
    pixel_dim = int(pixel_dim)
    feature_dim = int(feature_dim)
    if pixel_dim < 1 or feature_dim < 1 or feature_dim > pixel_dim:
        raise ValueError("need 1 <= feature_dim <= pixel_dim")
    rng = np.random.default_rng(int(seed))
    raw = rng.normal(size=(pixel_dim, feature_dim))
    q, _ = np.linalg.qr(raw, mode="reduced")
    return q.T.astype(np.float64)


def _set_model_latent(model, base_theta, basis, z) -> None:
    theta = np.asarray(base_theta, np.float64) + np.asarray(basis, np.float64) @ np.asarray(z, np.float64)
    with torch.no_grad():
        target = torch.tensor(theta, dtype=model.plate.theta.dtype, device=model.plate.theta.device)
        model.plate.theta.copy_(target)


def _render_model_views(model, base_theta, basis, z, kernels, order, camera_views: int) -> np.ndarray:
    _set_model_latent(model, base_theta, basis, z)
    model.eval()
    idx = torch.arange(int(camera_views), dtype=torch.long, device=kernels.device)
    with torch.no_grad():
        rgb, alpha = model()
        images = osf.render_properties(rgb, alpha, kernels, order, idx)
    return images.detach().cpu().numpy().astype(np.float64)


def _sensor_features(images: np.ndarray, projection: np.ndarray) -> np.ndarray:
    flat = np.asarray(images, np.float64).reshape(len(images), -1)
    return flat @ np.asarray(projection, np.float64).T


def _randomize_head(model, seed: int) -> None:
    """Make a deterministic visible probe world without training another model."""
    rng = np.random.default_rng(10000 + int(seed))
    weight = rng.normal(0.0, 1.15, size=tuple(model.head.weight.shape)).astype(np.float32)
    # The opacity channel is given a positive bias so the world is actually visible.
    bias = np.asarray([0.20, -0.15, 0.10, 2.05], np.float32)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(weight, device=model.head.weight.device))
        model.head.bias.copy_(torch.tensor(bias, device=model.head.bias.device))


def build_problem(
    seed: int = 0,
    operator_dim: int = 5,
    camera_views: int = 9,
    image_size: int = 10,
    features_per_view: int = 3,
    hidden_magnitude: float = 0.25,
    min_truth_change: float = 1e-3,
    device: str = "cpu",
) -> InverseProblem:
    """Construct one hidden in-family material change and its bounded camera sensors."""
    seed_all(int(seed))
    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"

    anchors, q_norm, _ijk, _axes = osf.make_anchor_grid(
        4, 3, 3,
        x_span=1.05,
        y_span=0.76,
        z_min=-1.10,
        z_max=0.72,
    )
    poses, _cam_ijk, _cam_axes = ow.camera_grid(
        int(camera_views), 1, 1,
        x_span=1.10,
        y_span=0.0,
        z_near=2.85,
        z_far=2.85,
    )
    kernels, order, _depth = osf.precompute_renderer(
        anchors,
        poses,
        int(image_size),
        int(image_size),
        radius_world=0.22,
        device=dev,
    )

    model = osf.OperatorField(q_norm, dim=int(operator_dim), seed=int(seed), learn_material=True).to(dev)
    _randomize_head(model, int(seed))
    base_theta = model.plate.theta.detach().cpu().numpy().astype(np.float64).copy()
    basis = material_tangent_basis(len(base_theta))
    d = basis.shape[1]
    projection = make_sensor_projection(
        int(image_size) * int(image_size) * 3,
        int(features_per_view),
        seed=20000 + int(seed),
    )
    z0 = np.zeros(d, np.float64)
    baseline_images = _render_model_views(
        model, base_theta, basis, z0, kernels, order, int(camera_views)
    )

    rng = np.random.default_rng(30000 + int(seed))
    best = None
    for _attempt in range(32):
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction) + 1e-12
        truth_z = direction * float(hidden_magnitude)
        truth_images = _render_model_views(
            model, base_theta, basis, truth_z, kernels, order, int(camera_views)
        )
        change = float(np.mean(image_rel_rmse(baseline_images, truth_images)))
        if best is None or change > best[0]:
            best = (change, truth_z.copy(), truth_images.copy())
        if change >= float(min_truth_change):
            break
    assert best is not None
    change, truth_z, truth_images = best
    truth_features = _sensor_features(truth_images, projection)
    _set_model_latent(model, base_theta, basis, z0)

    return InverseProblem(
        model=model,
        base_theta=base_theta,
        basis=basis,
        kernels=kernels,
        order=order,
        projection=projection,
        truth_z=truth_z,
        truth_features=truth_features,
        truth_images=truth_images,
        baseline_images=baseline_images,
        camera_views=int(camera_views),
        truth_change_rel_rmse=float(change),
        device=dev,
    )


def render_all_views(problem: InverseProblem, z: np.ndarray) -> np.ndarray:
    return _render_model_views(
        problem.model,
        problem.base_theta,
        problem.basis,
        z,
        problem.kernels,
        problem.order,
        problem.camera_views,
    )


def observe_all_views(problem: InverseProblem, z: np.ndarray) -> np.ndarray:
    return _sensor_features(render_all_views(problem, z), problem.projection)


def measure_view_jacobians(
    problem: InverseProblem,
    z: np.ndarray,
    epsilon: float = 0.015,
):
    """Measure d(sensor features)/d(z) by central finite differences for each camera."""
    z = np.asarray(z, np.float64).copy()
    eps = float(epsilon)
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    pred = observe_all_views(problem, z)
    nviews, nfeatures = pred.shape
    d = z.size
    jac = np.empty((nviews, nfeatures, d), np.float64)
    for k in range(d):
        zp = z.copy(); zp[k] += eps
        zm = z.copy(); zm[k] -= eps
        plus = observe_all_views(problem, zp)
        minus = observe_all_views(problem, zm)
        jac[:, :, k] = (plus - minus) / (2.0 * eps)
    _set_model_latent(problem.model, problem.base_theta, problem.basis, z)
    return pred, jac


def selected_sensor_residual(problem: InverseProblem, z: np.ndarray, observed_views) -> float:
    observed = np.asarray(sorted({int(v) for v in observed_views}), dtype=np.int64)
    pred = observe_all_views(problem, z)
    residual = problem.truth_features[observed] - pred[observed]
    return float(np.linalg.norm(residual.reshape(-1)))


def inverse_update(
    problem: InverseProblem,
    z: np.ndarray,
    observed_views,
    epsilon: float = 0.015,
    damping: float = 1e-6,
    trust_radius: float = 0.18,
    relative_tol: float = 1e-6,
):
    """One measured local inverse step using target data from selected cameras only."""
    observed = np.asarray(sorted({int(v) for v in observed_views}), dtype=np.int64)
    pred, jac = measure_view_jacobians(problem, z, epsilon=epsilon)
    d = problem.basis.shape[1]
    J_obs = jac[observed].reshape(-1, d)
    # This is the only target-data access in the update. Unselected views remain hidden.
    residual = (problem.truth_features[observed] - pred[observed]).reshape(-1)
    step = damped_least_squares(
        J_obs,
        residual,
        damping=float(damping),
        trust_radius=float(trust_radius),
    )
    z_new = np.asarray(z, np.float64) + step
    diag = svd_diagnostics(J_obs, parameter_dim=d, relative_tol=relative_tol, weak_count=1)
    info = {
        "residual_before": float(np.linalg.norm(residual)),
        "step_norm": float(np.linalg.norm(step)),
        "rank": int(diag["rank"]),
        "singular_values": [float(x) for x in diag["singular_values"]],
    }
    return z_new, info


def _orbit_metrics(problem: InverseProblem, z: np.ndarray, observed_views) -> dict:
    images = render_all_views(problem, z)
    per_view = image_rel_rmse(images, problem.truth_images)
    observed = {int(v) for v in observed_views}
    unseen = [v for v in range(problem.camera_views) if v not in observed]
    return {
        "orbit_rel_rmse": float(np.mean(per_view)),
        "unseen_rel_rmse": float(np.mean(per_view[unseen])) if unseen else 0.0,
        "material_error": float(np.linalg.norm(np.asarray(z) - problem.truth_z)),
    }


def run_policy(
    problem: InverseProblem,
    policy: str,
    max_views: int = 3,
    inner_steps: int = 2,
    epsilon: float = 0.015,
    damping: float = 1e-6,
    trust_radius: float = 0.18,
    relative_tol: float = 1e-6,
    weak_count: int = 2,
    random_seed: int = 0,
) -> dict:
    if policy not in ("active", "random"):
        raise ValueError("policy must be active or random")
    max_views = min(max(1, int(max_views)), problem.camera_views)
    rng = np.random.default_rng(int(random_seed))
    d = problem.basis.shape[1]
    z = np.zeros(d, np.float64)
    observed = [problem.camera_views // 2]
    history = []

    for acquisition in range(max_views):
        update_info = None
        for _ in range(max(1, int(inner_steps))):
            z, update_info = inverse_update(
                problem,
                z,
                observed,
                epsilon=epsilon,
                damping=damping,
                trust_radius=trust_radius,
                relative_tol=relative_tol,
            )

        pred, jac = measure_view_jacobians(problem, z, epsilon=epsilon)
        J_obs = jac[np.asarray(observed, np.int64)].reshape(-1, d)
        diag = svd_diagnostics(
            J_obs,
            parameter_dim=d,
            relative_tol=relative_tol,
            weak_count=weak_count,
        )
        residual_after = float(
            np.linalg.norm((problem.truth_features[observed] - pred[observed]).reshape(-1))
        )
        rec = {
            "views_used": int(len(observed)),
            "observed_views": [int(v) for v in observed],
            "sensor_residual": residual_after,
            "jacobian_rank": int(diag["rank"]),
            "weak_mode": str(diag["weak_mode"]),
            "singular_values": [float(x) for x in diag["singular_values"]],
            "last_step_norm": float(update_info["step_norm"] if update_info else 0.0),
        }
        rec.update(_orbit_metrics(problem, z, observed))

        if acquisition + 1 < max_views:
            if policy == "active":
                chosen, scores, _choice_diag = select_active_view(
                    jac,
                    observed,
                    relative_tol=relative_tol,
                    weak_count=weak_count,
                )
                rec["next_view"] = int(chosen)
                rec["next_view_score"] = float(scores[chosen])
                rec["candidate_scores"] = {str(k): float(v) for k, v in scores.items()}
            else:
                unseen = [v for v in range(problem.camera_views) if v not in set(observed)]
                chosen = int(rng.choice(unseen))
                rec["next_view"] = chosen
            observed.append(chosen)
        history.append(rec)

    return {
        "policy": policy,
        "history": history,
        "orbit_auc": float(np.mean([r["orbit_rel_rmse"] for r in history])),
        "unseen_auc": float(np.mean([r["unseen_rel_rmse"] for r in history])),
        "final_orbit_rel_rmse": float(history[-1]["orbit_rel_rmse"]),
        "final_unseen_rel_rmse": float(history[-1]["unseen_rel_rmse"]),
        "final_material_error": float(history[-1]["material_error"]),
        "final_rank": int(history[-1]["jacobian_rank"]),
    }


def run_benchmark(
    seeds: int = 5,
    random_repeats: int = 8,
    operator_dim: int = 5,
    camera_views: int = 9,
    image_size: int = 10,
    features_per_view: int = 3,
    hidden_magnitude: float = 0.25,
    max_views: int = 3,
    inner_steps: int = 2,
    epsilon: float = 0.015,
    damping: float = 1e-6,
    trust_radius: float = 0.18,
    device: str = "cpu",
) -> dict:
    receipts = []
    active_auc = []
    random_auc = []
    active_final = []
    random_final = []
    active_material = []
    random_material = []
    active_wins = 0

    for seed in range(int(seeds)):
        problem = build_problem(
            seed=seed,
            operator_dim=operator_dim,
            camera_views=camera_views,
            image_size=image_size,
            features_per_view=features_per_view,
            hidden_magnitude=hidden_magnitude,
            device=device,
        )
        active = run_policy(
            problem,
            "active",
            max_views=max_views,
            inner_steps=inner_steps,
            epsilon=epsilon,
            damping=damping,
            trust_radius=trust_radius,
            random_seed=90000 + seed,
        )
        random_runs = [
            run_policy(
                problem,
                "random",
                max_views=max_views,
                inner_steps=inner_steps,
                epsilon=epsilon,
                damping=damping,
                trust_radius=trust_radius,
                random_seed=100000 + seed * 1000 + rep,
            )
            for rep in range(int(random_repeats))
        ]
        r_auc = float(np.median([r["orbit_auc"] for r in random_runs]))
        r_final = float(np.median([r["final_orbit_rel_rmse"] for r in random_runs]))
        r_mat = float(np.median([r["final_material_error"] for r in random_runs]))
        if active["orbit_auc"] < r_auc:
            active_wins += 1
        active_auc.append(active["orbit_auc"])
        random_auc.append(r_auc)
        active_final.append(active["final_orbit_rel_rmse"])
        random_final.append(r_final)
        active_material.append(active["final_material_error"])
        random_material.append(r_mat)
        receipts.append({
            "seed": seed,
            "truth_change_rel_rmse": problem.truth_change_rel_rmse,
            "material_coordinates": int(problem.basis.shape[1]),
            "active": active,
            "random_median": {
                "orbit_auc": r_auc,
                "final_orbit_rel_rmse": r_final,
                "final_material_error": r_mat,
            },
            "random_runs": random_runs,
        })
        print(
            f"seed {seed}: change={problem.truth_change_rel_rmse:.6f} "
            f"active_auc={active['orbit_auc']:.6f} random_med={r_auc:.6f} "
            f"active_final={active['final_orbit_rel_rmse']:.6f} random_final={r_final:.6f}"
        )

    summary = {
        "active_orbit_auc_mean": float(np.mean(active_auc)),
        "random_median_orbit_auc_mean": float(np.mean(random_auc)),
        "active_final_orbit_mean": float(np.mean(active_final)),
        "random_median_final_orbit_mean": float(np.mean(random_final)),
        "active_final_material_error_mean": float(np.mean(active_material)),
        "random_median_final_material_error_mean": float(np.mean(random_material)),
        "active_auc_wins": int(active_wins),
        "seeds": int(seeds),
    }
    summary["scientific_gate_pass"] = bool(
        summary["active_orbit_auc_mean"] < summary["random_median_orbit_auc_mean"]
        and summary["active_final_orbit_mean"] < summary["random_median_final_orbit_mean"]
        and summary["active_final_material_error_mean"] < summary["random_median_final_material_error_mean"]
        and active_wins >= max(1, int(np.ceil(0.6 * int(seeds))))
    )
    return {
        "experiment": "Gate 7A active inverse world",
        "claim_scope": "hidden change is inside the shared operator-material family",
        "config": {
            "operator_dim": int(operator_dim),
            "camera_views": int(camera_views),
            "image_size": int(image_size),
            "features_per_view": int(features_per_view),
            "hidden_magnitude": float(hidden_magnitude),
            "max_views": int(max_views),
            "inner_steps": int(inner_steps),
            "finite_difference_epsilon": float(epsilon),
            "damping": float(damping),
            "trust_radius": float(trust_radius),
            "random_repeats": int(random_repeats),
        },
        "summary": summary,
        "receipts": receipts,
    }


def selftest() -> None:
    problem = build_problem(
        seed=7,
        operator_dim=4,
        camera_views=5,
        image_size=8,
        features_per_view=2,
        hidden_magnitude=0.12,
        min_truth_change=1e-4,
        device="cpu",
    )
    z0 = np.zeros(problem.basis.shape[1], np.float64)
    first = problem.camera_views // 2
    pred, jac = measure_view_jacobians(problem, z0, epsilon=0.01)
    J0 = jac[[first]].reshape(-1, problem.basis.shape[1])
    diag = svd_diagnostics(J0, parameter_dim=problem.basis.shape[1])
    assert diag["rank"] < problem.basis.shape[1]
    before = float(np.linalg.norm((problem.truth_features[[first]] - pred[[first]]).reshape(-1)))
    z1, _ = inverse_update(
        problem, z0, [first], epsilon=0.01, damping=1e-6, trust_radius=0.20
    )
    after = selected_sensor_residual(problem, z1, [first])
    assert np.isfinite(after) and after < before, (before, after)
    _pred1, jac1 = measure_view_jacobians(problem, z1, epsilon=0.01)
    chosen, _scores, _ = select_active_view(jac1, [first])
    assert chosen != first
    print(
        "active_inverse_world selftest: PASS "
        f"(material_dim={problem.basis.shape[1]} rank1={diag['rank']} "
        f"residual={before:.6g}->{after:.6g} next={chosen})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out-dir", default="active_inverse_out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--random-repeats", type=int, default=8)
    ap.add_argument("--operator-dim", type=int, default=5)
    ap.add_argument("--camera-views", type=int, default=9)
    ap.add_argument("--image-size", type=int, default=10)
    ap.add_argument("--features-per-view", type=int, default=3)
    ap.add_argument("--hidden-magnitude", type=float, default=0.25)
    ap.add_argument("--max-views", type=int, default=3)
    ap.add_argument("--inner-steps", type=int, default=2)
    ap.add_argument("--epsilon", type=float, default=0.015)
    ap.add_argument("--damping", type=float, default=1e-6)
    ap.add_argument("--trust-radius", type=float, default=0.18)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.run:
        metrics = run_benchmark(
            seeds=args.seeds,
            random_repeats=args.random_repeats,
            operator_dim=args.operator_dim,
            camera_views=args.camera_views,
            image_size=args.image_size,
            features_per_view=args.features_per_view,
            hidden_magnitude=args.hidden_magnitude,
            max_views=args.max_views,
            inner_steps=args.inner_steps,
            epsilon=args.epsilon,
            damping=args.damping,
            trust_radius=args.trust_radius,
            device=args.device,
        )
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "active_inverse_metrics.json"
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("\n=== GATE 7A SUMMARY ===")
        print(json.dumps(metrics["summary"], indent=2))
        print(f"wrote {path}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
