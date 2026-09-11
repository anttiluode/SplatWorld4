# Route-view change objective — clean negative

The static edge-aware visual objective was the first SplatWorld4 material-learning rule to improve ordinary image reconstruction while also restoring the global beacon/pillar visibility signal. Its remaining failure was local: it still did not beat the fixed substrate at the true occlusion boundary.

Gate 3 tested a natural next hypothesis:

> Occlusion is not merely a static edge. It is visual structure that appears and disappears as viewpoint changes.

No geometry or visibility labels were added. The new material target was constructed only from the ordered training-route images.

## Training target

For each of the 25 training-route views, the target contains train-normalized:

```text
intensity        weight 0.20
spatial dx       weight 0.50
spatial dy       weight 0.50
signed route dt  weight 1.00
absolute |dt|    weight 1.00
```

`dt` is a finite image derivative along each observed camera route, divided by floor-plane distance traveled. A 12-dimensional train-only PCA of the combined field is used as the material-learning target.

As before, `g` is selected only by blocked prediction inside the training routes. The counterfactual test views and visibility labels never alter the material.

Eight paired seeds compare fixed, global-image-trained, static-edge-trained, and route-change-trained substrates.

---

## Result

### Ordinary image reconstruction

Median held-out relative RMSE:

| material | image RMSE |
|---|---:|
| global image | **0.595364** |
| static edge-aware | 0.595719 |
| route-view change | 0.598592 |
| fixed | 0.600267 |

The route-change material still improves reconstruction over fixed coordinates in **8/8 seeds**, but the improvement is much smaller (`0.001582` median absolute RMSE) and it loses to the static edge-aware material in **8/8** paired seeds.

### Visibility topology

Median Brier error:

| material | global Brier | near-boundary Brier |
|---|---:|---:|
| **static edge-aware** | **0.235400** | 0.307899 |
| fixed | 0.240698 | **0.303306** |
| global image | 0.244520 | 0.334957 |
| route-view change | **0.248842** | **0.341157** |

The temporal/change objective is worse than the static edge objective in every seed:

```text
change vs edge global visibility wins   0 / 8
change vs edge near-boundary wins       0 / 8
change vs edge image-RMSE wins          0 / 8
```

It is also worse than the fixed substrate on visibility in every seed:

```text
change vs fixed global visibility wins  0 / 8
change vs fixed near-boundary wins      0 / 8
```

Median paired deterioration relative to static edge training:

```text
visibility Brier       +0.013700
near-boundary Brier    +0.033439
image RMSE             +0.002818
```

Balanced accuracy returns to the same poor `0.5417` regime produced by global image training.

The preregistered verdict is therefore:

```text
VIEW_CHANGE_OBJECTIVE_DOES_NOT_IMPROVE_BOUNDARY
```

---

## Interpretation

This is useful because it separates two superficially similar ideas.

```text
static visual derivative structure   -> helps
route image derivative as a target   -> hurts
```

A static edge tells the substrate something about **state**: where image structure is discontinuous now.

The route derivative `dt` is different. It is a directional quantity tied to a particular traversal. Asking a pose-indexed static substrate to regress that derivative appears to bend the coordinates toward route-specific change patterns rather than toward the underlying visibility state. The material can still become marginally better at ordinary image prediction, but its topology probe gets worse.

So the negative does **not** imply that motion information is useless. It says this formulation is wrong:

```text
pose -> substrate -> regress observed temporal derivative
```

The cleaner next formulation would make motion a **relation between two substrate states**:

```text
state at pose p
    + known camera displacement delta
    -> predict state/image at p + delta
```

or constrain the local substrate Jacobian under camera motion, rather than forcing a single pose to encode a route-dependent derivative target.

That would test transition consistency — a world model property — instead of treating visual motion as another static label.
