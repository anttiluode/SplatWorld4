# SplatWorld4

Fresh continuation of the SplatWorld operator experiments.

## Repository note

This repository is intentionally being used as a **random/fresh working folder**. Older local SplatWorld4 material existed elsewhere but was never uploaded here, so none of that untracked material should be treated as part of this repository's provenance or as an implicit baseline. Work in this repo starts from what is actually committed here.

The room/counterfactual line established a useful boundary: static edge-aware visual learning can preserve some held-out visibility structure, while route derivatives and pair-transition training still do not beat strong smooth/pose controls at the actual occlusion boundary. See `ROOM_RESULTS.md`, `EDGE_AWARE_RESULTS.md`, `VIEW_CHANGE_RESULTS.md`, and `TRANSITION_RESULTS.md`.

The physical-XYZ gate then established a narrower positive result: correct camera depth helps on wholly unseen depth planes, but a direct XYZ predictor and even the fixed operator remain stronger than learned material on that benchmark. That means camera-addressed `pose -> image` prediction still leaves too easy a shortcut.

## Current experiment: make the operator a 3-D splat field

Gate 6 moves the address from **camera space to world space**.

Old formulation:

```text
camera (x,y,z) -> M_g(camera) -> image coefficients
```

New formulation:

```text
world q=(X,Y,Z) -> M_g(q) -> RGB + opacity at a 3-D splat anchor
                              -> perspective alpha renderer -> image
```

The camera no longer enters the operator. It only renders a world-fixed field.

This is the first SplatWorld4 gate that behaves like a genuine 3-D splat representation rather than a camera-conditioned image regressor. One shared material vector `g` generates the properties of many 3-D anchors; sparse camera views train the field; entire intermediate camera-depth planes remain test-only.

Controls use the same anchors and renderer:

- `operator`: learned material `g` + a linear splat-property head
- `fixed_operator`: identical operator coordinates but frozen material
- `mlp`: ordinary world-coordinate MLP `q -> RGB, opacity`
- `free`: independent RGB/opacity parameters for every anchor

Read [`OPERATOR_SPLAT_PLAN.md`](OPERATOR_SPLAT_PLAN.md) for the claim boundary.

### Procedural smoke / first local run

```bash
python3.13 operator_splat_field.py --selftest
python3.13 operator_splat_field.py --run --procedural --device cuda --steps 1800 --seeds 3 --out-dir operator_splat_out
```

### OBJ run

```bash
python3.13 -m pip install -r requirements-xyz.txt
python3.13 obj_xyz_world.py --download spot rounded_cube avocado
python3.13 operator_splat_field.py --run --models spot rounded_cube avocado --device cuda --steps 2200 --seeds 3 --out-dir operator_splat_obj_out
```

The key outputs are:

```text
operator_splat_metrics.json
operator_splat_comparison.png
operator_splat_field_slices.png
```

The decisive first question is no longer merely whether `z` matters. It is whether a **single compact operator substrate can instantiate a view-independent 3-D field** that survives sparse-view training and renders coherent views at untouched camera depths.

If that works, the next step is the one the earlier sparse-world result points toward: observe a local change, modify only the shared material, and ask whether unseen views of the changed 3-D world update coherently without retraining every splat.
