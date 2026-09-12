"""Pure numerical core for Gate 7 active inverse sensing.

This module deliberately knows nothing about Torch, images, or the hidden truth.
It only turns measured local response matrices into a bounded correction and
chooses a new measurement whose Jacobian exposes the current weak/null space.
"""
from __future__ import annotations

import numpy as np


def material_tangent_basis(n: int) -> np.ndarray:
    """Return an orthonormal basis orthogonal to the all-ones softmax gauge."""
    n = int(n)
    if n < 2:
        raise ValueError("material tangent needs at least two coordinates")
    raw = np.zeros((n, n - 1), dtype=np.float64)
    raw[: n - 1, :] = np.eye(n - 1, dtype=np.float64)
    raw[n - 1, :] = -1.0
    q, _r = np.linalg.qr(raw, mode="reduced")
    return q.astype(np.float64)


def damped_least_squares(
    J: np.ndarray,
    residual: np.ndarray,
    damping: float = 1e-3,
    trust_radius: float | None = None,
) -> np.ndarray:
    """Solve a damped local inverse step and optionally clip its Euclidean norm."""
    J = np.asarray(J, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if J.ndim != 2:
        raise ValueError("J must be a matrix")
    if J.shape[0] != residual.size:
        raise ValueError("residual length must match J rows")
    if damping < 0:
        raise ValueError("damping must be nonnegative")
    if J.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    u, s, vh = np.linalg.svd(J, full_matrices=False)
    if s.size == 0:
        step = np.zeros(J.shape[1], dtype=np.float64)
    else:
        filt = s / (s * s + float(damping))
        step = vh.T @ (filt * (u.T @ residual))

    if trust_radius is not None:
        radius = float(trust_radius)
        if radius <= 0:
            raise ValueError("trust_radius must be positive")
        norm = float(np.linalg.norm(step))
        if norm > radius:
            step = step * (radius / norm)
    return np.asarray(step, dtype=np.float64)


def svd_diagnostics(
    J: np.ndarray,
    parameter_dim: int | None = None,
    relative_tol: float = 1e-6,
    weak_count: int = 1,
) -> dict:
    """Measure observable rank and return exact-null or weakest parameter directions."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        raise ValueError("J must be a matrix")
    d = int(J.shape[1] if parameter_dim is None else parameter_dim)
    if d != J.shape[1]:
        raise ValueError("parameter_dim must equal J column count")
    if d < 1:
        raise ValueError("parameter_dim must be positive")
    if relative_tol < 0:
        raise ValueError("relative_tol must be nonnegative")

    _u, s, vh = np.linalg.svd(J, full_matrices=True)
    scale = float(s[0]) if s.size else 0.0
    threshold = float(relative_tol) * scale
    rank = int(np.sum(s > threshold)) if scale > 0.0 else 0

    if rank < d:
        weak_basis = vh[rank:d].T
        mode = "null"
    else:
        k = min(max(1, int(weak_count)), d)
        weak_basis = vh[d - k : d].T
        mode = "weak"

    return {
        "rank": rank,
        "parameter_dim": d,
        "singular_values": np.asarray(s, dtype=np.float64),
        "threshold": threshold,
        "weak_basis": np.asarray(weak_basis, dtype=np.float64),
        "weak_mode": mode,
    }


def select_active_view(
    view_jacobians: np.ndarray,
    observed_views,
    relative_tol: float = 1e-6,
    weak_count: int = 1,
):
    """Choose the unseen camera that most exposes the current weak/null subspace."""
    view_jacobians = np.asarray(view_jacobians, dtype=np.float64)
    if view_jacobians.ndim != 3:
        raise ValueError("view_jacobians must have shape (views, features, parameters)")
    nviews, _features, d = view_jacobians.shape
    observed = sorted({int(v) for v in observed_views})
    if not observed:
        raise ValueError("at least one observed view is required")
    if min(observed) < 0 or max(observed) >= nviews:
        raise IndexError("observed view out of range")

    J_obs = view_jacobians[observed].reshape(-1, d)
    diag = svd_diagnostics(
        J_obs,
        parameter_dim=d,
        relative_tol=relative_tol,
        weak_count=weak_count,
    )
    weak = diag["weak_basis"]

    scores: dict[int, float] = {}
    observed_set = set(observed)
    for v in range(nviews):
        if v in observed_set:
            continue
        exposed = view_jacobians[v] @ weak
        scores[v] = float(np.sum(exposed * exposed))
    if not scores:
        raise ValueError("all views have already been observed")

    best_score = max(scores.values())
    chosen = min(v for v, score in scores.items() if np.isclose(score, best_score, rtol=1e-12, atol=1e-15))
    return chosen, scores, diag
