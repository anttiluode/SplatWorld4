# SplatWorld4

Fresh continuation of the SplatWorld operator experiments.

## Repository note

This repository is intentionally being used as a **random/fresh working folder**. Older local SplatWorld4 material existed elsewhere but was never uploaded here, so none of that untracked material should be treated as part of this repository's provenance or as an implicit baseline. Work in this repo starts from what is actually committed here.

The room/counterfactual line established a useful boundary: static edge-aware visual learning can preserve some held-out visibility structure, while route derivatives and pair-transition training still do not beat strong smooth/pose controls at the actual occlusion boundary. See `ROOM_RESULTS.md`, `EDGE_AWARE_RESULTS.md`, `VIEW_CHANGE_RESULTS.md`, and `TRANSITION_RESULTS.md`.

The physical-XYZ gate then established a narrower positive result: correct camera depth helps on wholly unseen depth planes, but a direct XYZ predictor and even the fixed operator remain stronger than learned material on that benchmark. That means camera-addressed `pose -> image` prediction still leaves too easy a shortcut.

## Gate 6: world-space operator splat field

Gate 6 moves the address from **camera space to world space**.

```text
world q=(X,Y,Z) -> M_g(q) -> RGB + opacity at a 3-D splat anchor
                              -> perspective alpha renderer -> image
```

The camera no longer enters the operator. It only renders a world-fixed field. One shared material vector `g` generates the properties of many 3-D anchors; sparse camera views train the field; entire intermediate camera-depth planes remain test-only.

Controls use the same anchors and renderer:

- `operator`: learned material `g` + linear splat-property head
- `fixed_operator`: same operator coordinates but frozen material
- `mlp`: ordinary world-coordinate MLP
- `free`: independent RGB/opacity parameters for every anchor

Read [`OPERATOR_SPLAT_PLAN.md`](OPERATOR_SPLAT_PLAN.md) for the design and [`OPERATOR_SPLAT_RESULTS.md`](OPERATOR_SPLAT_RESULTS.md) for the first full GPU result.

### First full result

Held-out relative RMSE:

| scene | learned operator | fixed operator | coordinate MLP | free splats |
|---|---:|---:|---:|---:|
| procedural cubes | **0.420867** | 0.484071 | 0.341714 | 0.329520 |
| OBJ world | **0.450841** | 0.469289 | 0.375716 | 0.362376 |

The learned material beats its fixed counterpart in all 3/3 paired seeds on both scenes. The fixed-to-learned error reduction is **13.06%** on the procedural world and **3.93%** on the OBJ world.

The operator is still less accurate than the large MLP and free splats, but it uses only **64 trainable parameters**, versus 796 and 576 respectively.

## Gate 6b: capacity attack

The fixed operator has 44 trainable parameters; the learned operator has 64. The extra 20 are exactly the learned material degrees of freedom. So Gate 6 by itself does not prove that material is a particularly useful place to spend those parameters.

`operator_splat_capacity.py` adds the missing controls:

```text
operator64    learned material + linear head                  64 params
fixed_deep64  frozen material + nonlinear 10 -> 4 -> 4 head  64 params
tiny_mlp56    coordinate MLP 3 -> 4 -> 4 -> 4                56 params
tiny_mlp74    coordinate MLP 3 -> 5 -> 5 -> 4                74 params
```

The decisive comparison is:

```text
operator64 < fixed_deep64
```

on held-out views. If it survives on both procedural and OBJ scenes, the evidence becomes much stronger that **changing the shared substrate is a useful inductive bias per parameter**, rather than merely extra capacity.

### Run Gate 6b

Procedural:

```bash
python3.13 operator_splat_capacity.py --run --procedural \
  --device cuda --steps 1800 --seeds 3 \
  --out-dir operator_splat_capacity_out
```

OBJ:

```bash
python3.13 operator_splat_capacity.py --run \
  --models spot rounded_cube avocado \
  --device cuda --steps 2200 --seeds 3 \
  --out-dir operator_splat_capacity_obj_out
```

The output is `operator_splat_capacity_metrics.json`.

## Gate 7A: active inverse world

Gate 7A adds the inverse operation that the world-space field was missing:

```text
bounded camera observation
        ↓
finite-difference local response J = d(observation)/d(material)
        ↓
damped inverse material update
        ↓
null / weak material directions
        ↓
choose the next camera that exposes them
```

The solver never receives the hidden material change. It receives target features only for cameras it has already acquired. Candidate next cameras are chosen from the current **predicted** Jacobian, not by peeking at unseen target images.

The frozen confirmation uses nine gauge-free material coordinates, only three sensor features per camera, seven candidate cameras and at most three acquired views. Across **12 hidden material changes**, with **16 matched random-view controls per change**:

| metric | active | random-view median |
|---|---:|---:|
| mean orbit-error AUC | **6.8936e-4** | 8.5519e-4 |
| final full-orbit rel-RMSE | **2.7085e-4** | 4.4184e-4 |
| final material-coordinate error | **0.09963** | 0.12252 |

That is a **19.39%** lower orbit-error curve, **38.70%** lower final full-orbit error and **18.69%** lower final material error for active sensing.

Paired outcomes:

```text
orbit-error AUC      active wins 11 / 12
final orbit error    active wins 12 / 12
final material error active wins 12 / 12
```

The observability pattern is exact in every seed:

```text
1 view -> rank 3 / 9
2 views -> rank 6 / 9
3 views -> rank 9 / 9
```

The active camera normally goes from the center to one edge and then the opposite edge. So the mechanism is doing something more specific than generic multiview averaging: it buys the view predicted to expose the currently missing material directions.

Read [`GATE7_ACTIVE_INVERSE_RESULTS.md`](GATE7_ACTIVE_INVERSE_RESULTS.md) for the full interpretation, caveats and reproduction command. A compact frozen receipt is in [`results/gate7_active_inverse_summary.json`](results/gate7_active_inverse_summary.json).

### Claim boundary

This is a positive control. The hidden change was generated **inside the same shared-material family used by the inverse solver**. Gate 7A therefore establishes the active inverse mechanism, not arbitrary 3-D scene understanding.

The next meaningful attacker is Gate 7B: change the rendered world outside the material family—move local geometry, add/remove an object or alter a localized component—and test both whether active sensing still helps and whether the residual/singular diagnostics can correctly report **model mismatch** rather than inventing a confident material explanation.

### Run Gate 7A

```bash
python -m unittest discover -s tests -p 'test_active_inverse_*.py' -v
python active_inverse_world.py --selftest
python active_inverse_world.py --run --device cpu \
  --seeds 12 --random-repeats 16 \
  --operator-dim 5 --camera-views 7 --image-size 8 \
  --features-per-view 3 --hidden-magnitude 0.22 \
  --max-views 3 --inner-steps 2 \
  --out-dir active_inverse_confirmation
```
