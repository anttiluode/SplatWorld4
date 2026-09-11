# Gate 4 — relational transition consistency

The previous Gate 3 negative showed that putting route-view `dt` into a **static pose target** was the wrong formulation. Gate 4 instead makes motion relational.

For an observed training pair `(p0,p1)`, the transition readout receives

```text
[h_g(p0), h_g(p1) - h_g(p0), delta_pose]
```

and predicts the change in train-only image-PCA coefficients. Candidate material `g` is selected only by blocked CV over observed training-route pairs.

The held-out test contains **160 equal-length (0.28 floor-unit) counterfactual motions** generated without consulting visibility. Only after predictions are frozen are the test pairs labeled geometrically:

```text
20   cross the beacon/pillar visibility boundary
140  do not cross it
20   nearest-boundary non-crossing pairs used as a hard matched diagnostic
```

Visibility, hit identity, depth, test images, and counterfactual geometry never select `g`, the readout, seed, or hyperparameter.

Eight paired material seeds are used throughout.

---

## Result

### Boundary-crossing transitions

Median held-out relative image RMSE:

| representation | crossing RMSE |
|---|---:|
| direct pose transition | **0.422968** |
| matched random smooth state | **0.422456** |
| fixed operator | 0.423179 |
| static-edge-trained operator | 0.423182 |
| transition-trained operator | **0.423177** |
| image-basis oracle | 0.341096 |

Against the two operator controls, relational material learning is perfectly consistent but extremely small:

```text
transition g beats fixed g on crossing pairs:   8 / 8
transition g beats edge-trained g:              8 / 8
median improvement vs fixed:                    0.00000212 RMSE
median improvement vs edge:                     0.00000491 RMSE
```

That is a reproducible direction, but it is **not an operator-specific win**.

The transition-trained operator loses to the matched random smooth-state control in 6/8 seeds, with median disadvantage `0.000721` RMSE. It also loses to the simple direct-pose transition control (`0.422968` vs `0.423177`).

### All counterfactual transitions

Median held-out relative RMSE:

```text
fixed operator                 0.419432
static-edge-trained operator   0.419428
transition-trained operator    0.419435
random smooth state            0.418760
direct pose                    0.419119
```

Transition training is worse than fixed on the full test in **8/8 seeds** and worse than static-edge training in **8/8**:

```text
median change vs fixed   -0.00000348
median change vs edge    -0.00000691
```

So the tiny crossing-specific improvement does not generalize to ordinary small motions.

---

## What survived the control attack

A very narrow fact survived:

> selecting `g` from pair-transition prediction nudged the operator in the same direction on held-out visibility-crossing motions in all eight seeds.

But the effect is only a few parts in `10^-6` in relative image RMSE, and generic smooth coordinates do substantially better. The strongest pose-only/local control also wins.

Therefore this does **not** show that the substrate learned occlusion geometry, a world transition law, or a useful operator advantage.

The preregistered strong condition fails because relational `g` does not beat the strongest pose/local control and does not improve the full transition set.

```text
RELATIONAL_G_IMPROVES_OPERATOR_BOUNDARY_TRANSITIONS_BUT_NOT_STRONGEST_CONTROL
```

The CI run was green on Python 3.11 and 3.12 and the eight-seed result job completed successfully.

---

## Interpretation

This gate was still useful because the static-vs-relational distinction was real:

```text
route derivative as a static target       -> large degradation
pair-transition training                  -> removes that degradation
                                            and gives a tiny boundary-local nudge
```

But the control hierarchy now says the bottleneck is no longer the loss formulation alone. The present operator coordinates are simply not a better transition representation than cheap smooth functions of pose.

That suggests a change of question rather than another weight sweep. If SplatWorld4 continues, the next experiment should demand something pose interpolation cannot get for free — for example **latent state that persists when the same camera pose can correspond to different images because the world itself changed**. A static room with deterministic pose->image mapping still lets pose remain an excellent shortcut, even around occlusion.
