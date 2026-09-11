# Gate 6 — world-space operator splat field

## Why this gate exists

The XYZ depth gate made camera `z` functionally meaningful, but the representation is still camera-addressed:

```text
camera (x,y,z) -> operator response -> image coefficients
```

That leaves a strong shortcut. A direct pose network can learn the same smooth view field without storing a 3-D world.

Gate 6 removes that shortcut.

## Representation

The operator is now queried at **world-space locations**:

```text
world q=(X,Y,Z)
    -> M_g(q)
    -> compact response h_g(q)
    -> RGB + opacity for a splat anchored at q
```

A perspective camera then renders those splats. Camera pose never enters the operator or splat-property network.

So the causal direction becomes:

```text
shared material g
    -> 3-D field of splat properties
    -> camera projection / alpha compositing
    -> image
```

This is much closer to a genuine 3-D representation than pose-to-image regression.

## What is fixed in the first gate

To keep the question clean:

- splat anchor positions are a fixed 3-D lattice;
- splat scale is fixed;
- only RGB and opacity are generated;
- the target is the existing external perspective/z-buffer OBJ scene;
- training uses only a small number of camera views;
- entire intermediate camera-depth planes remain test-only.

If this works, later gates can learn position offsets, anisotropic covariance, and stateful writes.

## Controls

All methods use the **same anchor grid and the same differentiable splat renderer**.

| method | field parameterization |
|---|---|
| `operator` | learned material `g`; `h_g(q)` -> linear RGB/opacity head |
| `fixed_operator` | same operator, but `g` frozen at initialization |
| `mlp` | ordinary coordinate MLP `q -> RGB,opacity` |
| `free` | independent RGB/opacity tuple at every anchor |

The MLP and free-splat controls deliberately have more direct capacity. Parameter count is reported beside held-out error.

## Split

The camera still uses five physical depth planes.

```text
z0  candidate training/validation views
z1  TEST ONLY
z2  candidate training/validation views
z3  TEST ONLY
z4  candidate training/validation views
```

Only `--sparse-views` camera views are selected for training from the available planes. Validation is disjoint and also restricted to the available planes.

The primary metric is relative image RMSE on all views in the two untouched depth planes.

## Claims

The first useful claim is deliberately modest:

```text
A single shared operator material can generate a view-independent 3-D splat
field that reconstructs unseen camera-depth planes from sparse views.
```

Stronger evidence would be:

1. learned `operator` beats `fixed_operator`;
2. it approaches or beats the coordinate MLP with substantially fewer parameters;
3. the advantage grows as the number of training views decreases;
4. later, a small local material write changes coherent unseen views without retraining every splat.

Failure is equally informative. If the MLP/free controls dominate and learning `g` does not improve on fixed material, the current resolvent is not yet a useful 3-D field generator.

## Run

First use the procedural scene:

```bash
python operator_splat_field.py --run --procedural --device cuda --steps 1800 --seeds 3 --out-dir operator_splat_out
```

Then use the downloaded OBJ scene:

```bash
python operator_splat_field.py --run --models spot rounded_cube avocado --device cuda --steps 2200 --seeds 3 --out-dir operator_splat_obj_out
```

Important outputs:

```text
operator_splat_metrics.json
operator_splat_comparison.png
operator_splat_field_slices.png
```

The comparison image shows held-out camera-depth views. The field-slice image shows the generated RGB×opacity splat field itself, so the 3-D object can be inspected independently of any one camera.
