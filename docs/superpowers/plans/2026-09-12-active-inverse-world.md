# Gate 7 Active Inverse World Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Gate 7A experiment that recovers a hidden shared-material edit from bounded visual observations and compares null-space-driven camera selection against matched random views.

**Architecture:** Keep the inverse algebra in a pure-NumPy module so it can be tested independently. A separate Torch experiment module reuses `operator_splat_field.py` for the world-space operator field and renderer, measures finite-difference sensor Jacobians, applies the inverse core, and records active-vs-random acquisition curves. CI runs unit tests plus a small CPU smoke benchmark and uploads its receipt.

**Tech Stack:** Python 3.11/3.13, NumPy, PyTorch CPU in CI, existing SplatWorld4 renderer.

**Spec:** `docs/superpowers/specs/2026-09-12-active-inverse-world-design.md`

## Global Constraints

- The solver must never receive the hidden material delta or unselected target views.
- Jacobians used by the inverse solver must be measured by finite differences, not autograd.
- Remove the softmax common-offset gauge with an explicit material tangent basis.
- Active and random policies must share the initial view, inverse solver, budgets, and hidden change.
- Scientific failure is recorded, not converted into CI failure.

---

### Task 1: Pure inverse core

**Files:**
- Create: `active_inverse_core.py`
- Create: `tests/test_active_inverse_core.py`

**Interfaces:**
- Produces: `material_tangent_basis(n)`, `damped_least_squares(J, residual, damping, trust_radius)`, `svd_diagnostics(J, parameter_dim, relative_tol, weak_count)`, and `select_active_view(view_jacobians, observed_views, relative_tol, weak_count)`.

- [ ] **Step 1: Write failing tests** for orthonormal zero-sum tangent coordinates, damped residual reduction, rank/null-space diagnostics, and selection of the camera with maximal weak-subspace sensitivity.
- [ ] **Step 2: Run** `python -m unittest discover -s tests -p 'test_active_inverse_core.py' -v` and verify failure because `active_inverse_core` does not exist.
- [ ] **Step 3: Implement** the four functions with NumPy SVD/QR and deterministic tie breaking.
- [ ] **Step 4: Re-run** the focused test and require all tests to pass.
- [ ] **Step 5: Commit** `test: define Gate 7 inverse algebra` then `feat: add Gate 7 inverse algebra`.

### Task 2: Torch finite-difference world inverse

**Files:**
- Create: `active_inverse_world.py`
- Extend: `tests/test_active_inverse_core.py`

**Interfaces:**
- Consumes: Gate 6 `OperatorField`, `make_anchor_grid`, `precompute_renderer`, `render_properties` and Task-1 inverse functions.
- Produces: `make_sensor_projection`, `render_all_views`, `measure_view_jacobians`, `run_policy`, `run_benchmark`, `--selftest`, and `--run` CLI.

- [ ] **Step 1: Add a failing integration test** that constructs a tiny operator field, verifies one bounded view is rank deficient, applies one measured inverse step, and requires selected-view sensor residual to decrease.
- [ ] **Step 2: Run** the focused unittest and confirm it fails because the world module is absent.
- [ ] **Step 3: Implement** deterministic sensor projection, gauge-free latent material coordinates, central finite-difference Jacobian measurement for all camera views, damped updates, active/random acquisition, and full-orbit evaluation.
- [ ] **Step 4: Add CLI receipts** containing per-view material error, orbit relative RMSE, Jacobian rank/singular values, selected views, active scores, random medians, and a non-enforcing `scientific_gate_pass` boolean.
- [ ] **Step 5: Run** `python active_inverse_world.py --selftest` and the focused unittest.

### Task 3: Reproducible smoke benchmark and documentation

**Files:**
- Create: `.github/workflows/active-inverse-world.yml`
- Create: `GATE7_ACTIVE_INVERSE_RESULTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `active_inverse_world.py --run`.
- Produces: CI artifact `active-inverse-smoke/active_inverse_metrics.json` and documented Gate 7 result/limits.

- [ ] **Step 1: Add CI** for Python 3.11/3.13 core+selftest and one Python 3.11 CPU smoke benchmark using a tiny field.
- [ ] **Step 2: Open a PR** so GitHub Actions executes the branch in isolation.
- [ ] **Step 3: Inspect smoke logs/receipt** and record the actual result rather than the hoped-for result.
- [ ] **Step 4: Update `GATE7_ACTIVE_INVERSE_RESULTS.md` and README** with exact active-vs-random numbers and the honesty boundary.
- [ ] **Step 5: Run all Gate 7 CI checks** and verify existing operator-splat checks remain green before merge.