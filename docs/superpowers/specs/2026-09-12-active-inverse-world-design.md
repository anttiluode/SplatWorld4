# Gate 7 — Active Inverse World Design

## Goal

Test whether a bounded visual observer can infer a hidden change to SplatWorld4's shared operator material and choose the next camera view from the currently unobservable material directions.

## Scientific question

Gate 6 established the forward map

```text
shared material g -> world-space operator field -> splat properties -> camera image
```

Gate 7A adds the inverse loop without giving the solver the hidden material change:

```text
observed image features
        -> residual
        -> measured finite-difference response J = d(observation)/d(material coordinates)
        -> damped local inverse update
        -> weak/null material subspace
        -> next camera chosen to expose that subspace
```

The hidden change is deliberately generated inside the existing material family. This is a positive-control gate for the inverse mechanism, not yet a claim that arbitrary moved mesh geometry can be absorbed by the material model.

## Bounded observation

The solver does not receive the full orbit or hidden material vector. Each selected camera image is compressed by a fixed deterministic linear sensor projection to a small number of features. With fewer measurements than material tangent coordinates, early observations are rank deficient by construction.

Full rendered images from all cameras are retained only for evaluation of unseen-orbit error.

## Parameterization

`TorchOperatorPlate.theta` has a softmax gauge: adding the same scalar to all entries changes no material. Gate 7 removes that trivial null direction with an orthonormal tangent basis `B` satisfying

```math
B^T B = I,   1^T B = 0.
```

The inverse state is `z` with `theta = theta0 + B z`.

## Inverse update

At the current estimate, finite differences measure the sensor Jacobian for every camera. Selected cameras are stacked into `J_obs`. The update is damped least squares with a trust radius:

```math
Delta z = argmin ||J_obs Delta z - r||^2 + lambda ||Delta z||^2.
```

No autograd Jacobian and no hidden truth parameters are used by the solver.

## Active camera rule

Compute the SVD of `J_obs`. Exact null directions are used while rank deficient; after full rank, the weakest singular directions are used. For each unobserved camera `v`, score

```math
score(v) = ||J_v N_weak||_F^2.
```

Choose the highest-scoring view. This uses only the current forward model and previously selected observations.

## Controls

Compare:

- `active`: same initial camera and same inverse update, next view chosen by weak/null-space exposure;
- `random`: same initial camera and same inverse update, next views sampled uniformly without replacement.

Random is repeated several times per hidden change and summarized by its median.

## Primary metrics

For each observation count report:

- material-coordinate error `||z_hat-z_true||`;
- full-orbit image relative RMSE, including unobserved cameras;
- numerical rank and weakest singular value of the selected observation Jacobian;
- chosen camera indices and active information scores.

The scientific gate is positive only if active acquisition improves the orbit-error curve and material recovery relative to the matched random-view median across multiple hidden changes. CI must not be made to fail merely because the scientific hypothesis fails.

## Honesty boundary

Gate 7A does not establish arbitrary 3-D reconstruction, real-camera calibration, recovery of off-manifold object motion, or superiority to modern NeRF/3DGS systems. It tests one narrower missing operation: observation -> measured local inverse -> active view selection -> coherent shared-material update.