# Edge-aware image-only learning — partial repair of visibility damage

The first SplatWorld4 room stage established a specific failure:

- fixed operator coordinates already carried useful beacon/pillar visibility structure;
- training `g` only on global image-PCA prediction improved held-out image reconstruction a little;
- but that same training damaged the visibility partition, especially near a true occlusion boundary.

This gate changes **only the visual objective**. `g` still receives no depth, hit identity, beacon visibility, geometry labels, or counterfactual test images.

## Edge-aware target

From the 25 training-route images only, three channels are built:

```text
intensity
horizontal finite difference dx
vertical finite difference dy
```

Each channel is centered and normalized by its RMS measured on training images only. Fixed preregistered weights are then applied:

```text
intensity  0.35
dx         1.00
dy         1.00
```

A 12-dimensional PCA target is fitted to that augmented training field. `g` is optimized by the same blocked training-only CV and conservative material transfers as before.

Afterwards the material is frozen. It is evaluated on:

1. ordinary held-out image reconstruction using the *original* normal 12-D image target;
2. the unchanged held-out beacon-visibility probe.

Eight paired seeds compare fixed material, the old global-image objective, and the new edge-aware image objective.

---

## Main result

### Ordinary image reconstruction

Median held-out relative image RMSE:

| material | RMSE |
|---|---:|
| fixed operator | 0.600267 |
| global-image-trained `g` | **0.595364** |
| edge-aware-image-trained `g` | 0.595719 |

The edge-aware objective still improves image reconstruction over fixed coordinates in **8/8 seeds**, by median `0.004543` absolute RMSE.

It is slightly worse than the global image objective for pure reconstruction (`+0.000479` median RMSE), also in 8/8 paired comparisons. That is the expected price of asking the material to preserve more high-frequency visual structure.

### Held-out visibility

Median Brier error across seeds:

| material | visibility Brier | balanced accuracy |
|---|---:|---:|
| fixed operator | 0.240698 | 0.7083 |
| global-image-trained `g` | 0.244520 | 0.5417 |
| **edge-aware-image-trained `g`** | **0.235400** | **0.7083** |

The edge-aware objective beats the old global-image objective on visibility Brier in **8/8 seeds**.

It also beats the fixed operator on global visibility Brier in **8/8 seeds**.

For the median prediction across seeds:

```text
fixed       Brier 0.240212   balanced accuracy 0.7083
global      Brier 0.244517   balanced accuracy 0.5417
edge-aware  Brier 0.235212   balanced accuracy 0.7083
```

So an image-only objective can change `g` in a direction that simultaneously:

```text
improves ordinary image reconstruction over fixed g
AND
restores the visibility accuracy destroyed by global image training
AND
improves global visibility calibration beyond fixed g
```

That is the first positive cross-task effect in this room line.

---

## But the strongest geometry claim still fails

The critical location remains the actual occlusion boundary.

Median near-boundary Brier across seeds:

```text
fixed       0.303306
global      0.334957
edge-aware  0.307899
```

The edge-aware objective repairs most of the damage done by the global objective:

```text
global -> edge-aware median improvement: 0.027380
paired wins: 8 / 8
```

But it does **not** beat the fixed substrate near the boundary:

```text
fixed -> edge-aware median change: -0.004654  (edge-aware is worse)
paired edge-aware wins vs fixed: 2 / 8
```

Thus the edge-aware material has not earned the strongest claim that visual learning sharpened the true topological boundary beyond what happened to exist in the fixed resolvent coordinates.

Far from the boundary, however, the edge-aware representation is clearly better calibrated:

```text
median-prediction Brier
fixed       0.214395
global      0.211396
edge-aware  0.198615
```

---

## Verdict

The preregistered strong success condition required the edge-aware material to beat global image training, improve near the boundary, surpass the fixed visibility representation there, **and** improve image reconstruction over fixed coordinates.

The first, second and fourth conditions pass. The third does not.

```text
EDGE_AWARE_OBJECTIVE_REPAIRS_PART_OF_VISIBILITY_DAMAGE
```

Current status:

```text
edge-aware g beats global g on visibility       YES, 8/8
edge-aware g beats global g near boundary       YES, 8/8
edge-aware g beats fixed g globally (Brier)     YES, 8/8
edge-aware g beats fixed g near boundary        NO, 2/8
edge-aware g improves image RMSE over fixed     YES, 8/8
edge-aware g beats global image RMSE             NO
geometry labels given to g                       NO
counterfactual test views used in training       NO
```

## Why this matters

The previous negative could have meant that the material mechanism simply had no route from visual learning to visibility structure.

This experiment rules out that stronger pessimistic interpretation.

Changing **which visual errors matter** changes what topology survives in the substrate. A derivative-sensitive image objective recovers the fixed representation's global visibility accuracy and improves global calibration, while retaining the reconstruction benefit of learned material.

The remaining failure is highly localized: the true discontinuity itself is still not sharpened beyond the fixed coordinates.

That suggests the next gate should target **visual change across nearby observations**, rather than static edge magnitude alone. Occlusion is not merely an edge in one frame; it is an edge that appears, moves, or disappears under viewpoint change. A route-only temporal/disocclusion objective can encode that distinction without ever supplying a visibility label.
