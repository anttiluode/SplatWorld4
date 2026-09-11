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

## Current experiment: Gate 6b capacity attack

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

If the material survives this control, the next gate is the more ambitious one suggested by the earlier sparse-world result: **observe a local world change, modify only the shared material, and ask whether unseen viewpoints update coherently without retraining every splat**.
