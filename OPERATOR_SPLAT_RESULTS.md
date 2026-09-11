# Gate 6 result — world-space operator splat field

These are the first full local GPU runs of the world-space splat architecture introduced in Gate 6.

The key architectural change from the earlier XYZ gate is that the operator is queried at **world-space splat anchors**, not at camera poses:

```text
q = (X,Y,Z) world anchor
    -> shared operator M_g(q)
    -> RGB + opacity
    -> fixed perspective splat renderer
    -> image
```

The camera therefore does not directly condition the learned field. Entire intermediate camera-depth planes remain test-only.

## Run A — procedural three-cube scene

Command:

```bash
python3.13 operator_splat_field.py --run --procedural \
  --device cuda --steps 1800 --seeds 3 \
  --out-dir operator_splat_out
```

Mean held-out relative RMSE over three seeds:

| representation | test rel-RMSE | std | trainable parameters |
|---|---:|---:|---:|
| free per-anchor splats | **0.329520** | 0.000000 | 576 |
| coordinate MLP | 0.341714 | 0.005420 | 796 |
| **learned operator** | **0.420867** | 0.003180 | **64** |
| fixed operator | 0.484071 | 0.001129 | 44 |

Learning the shared material changes

```text
0.484071 -> 0.420867
```

which is an absolute improvement of `0.063204`, or **13.06% relative to the fixed-operator error**.

Per seed, learned operator test error was:

```text
0.41637
0.42308
0.42315
```

while fixed operator was:

```text
0.48543
0.48266
0.48412
```

So material learning wins the direct paired comparison in all three seeds.

The learned operator used roughly 98–104 splats above opacity `0.10`; fixed operator used 92.

## Run B — OBJ scene (`spot`, `rounded_cube`, `avocado`)

Command:

```bash
python3.13 operator_splat_field.py --run \
  --models spot rounded_cube avocado \
  --device cuda --steps 2200 --seeds 3 \
  --out-dir operator_splat_obj_out
```

Mean held-out relative RMSE:

| representation | test rel-RMSE | std | trainable parameters |
|---|---:|---:|---:|
| free per-anchor splats | **0.362376** | 0.000000 | 576 |
| coordinate MLP | 0.375716 | 0.002227 | 796 |
| **learned operator** | **0.450841** | 0.000493 | **64** |
| fixed operator | 0.469289 | 0.000421 | 44 |

Learning the shared material changes

```text
0.469289 -> 0.450841
```

an absolute improvement of `0.018448`, or **3.93% relative to fixed-operator error**.

Per seed, learned operator test error was:

```text
0.45138
0.45095
0.45019
```

versus fixed operator:

```text
0.46986
0.46885
0.46916
```

Again, learned material wins all three paired seeds.

## What is supported

The result supports a narrower but useful statement:

> In a view-independent 3-D splat field, changing one shared operator material vector improves held-out novel-depth rendering relative to the same operator with frozen material.

The effect is large on the procedural scene and smaller but very stable on the OBJ scene.

The learned operator is also compact:

```text
operator / full MLP parameter ratio = 64 / 796 = 0.0804
operator / free-splat parameter ratio = 64 / 576 = 0.1111
```

So it uses about **8% of the MLP parameters** and **11% of the free-splat parameters** in this configuration.

## What is not yet supported

The operator is not currently the most accurate representation. Both the large coordinate MLP and free per-anchor parameters win on held-out image error.

More importantly, the learned operator has 20 more trainable parameters than the fixed operator:

```text
fixed operator = 44 trainable parameters
learned operator = 64 trainable parameters
Delta = 20
```

Those 20 parameters are the learnable material degrees of freedom. Therefore the present fixed-vs-learned comparison does **not yet isolate whether material is a particularly useful place to spend 20 parameters**, rather than merely showing that 64 trainable parameters can outperform 44.

That is the purpose of Gate 6b.

## Gate 6b preregistered question

Compare the 64-parameter learned operator against controls that have approximately the same parameter budget but cannot alter the substrate:

1. frozen operator + nonlinear head with exactly 64 trainable parameters;
2. tiny coordinate MLP below the operator budget;
3. tiny coordinate MLP just above the operator budget;
4. existing 44-parameter fixed operator, 796-parameter MLP, and 576-parameter free field as context.

A meaningful operator-specific result is:

```text
learned operator < capacity-matched frozen operator
```

on held-out views, preferably in paired seeds and on both procedural and OBJ scenes.

If that fails, Gate 6 should be interpreted mainly as a compact-function-capacity result. If it passes, the evidence becomes much stronger that **changing the shared physical/operator substrate is a useful inductive bias per parameter**.
