# Visibility topology probe — image-trained `g` does not preserve the boundary

This is the sharper follow-up to the first SplatWorld4 room reconstruction test.

The repository remains a **fresh/random working folder**. Older local SplatWorld4 material was never uploaded here, so this result is based only on the self-contained room and operator code committed in this repository.

## Question

The reconstruction experiment found a small but extremely repeatable improvement when the shared material vector `g` was trained from route images:

```text
fixed operator, 8-seed median    0.600267 relative image RMSE
image-trained g, 8-seed median  0.595364
paired wins                      8 / 8
```

But that gain was strongest **far from**, not near, a true pillar/beacon occlusion boundary. So aggregate image error did not support a geometry-memory interpretation.

This probe asks something more direct:

> After `g` has been trained only to predict route images, does the frozen substrate make a real held-out visibility boundary easier to read out?

The visibility label is binary: whether the bright beacon on the back wall is visible from a camera position or occluded by a pillar.

Crucially, `g` never receives that label during material learning. The sequence is:

```text
route images
    -> train g on image coefficients only
    -> freeze g
    -> fit tiny visibility readout on training-route labels
    -> evaluate counterfactual interior poses
```

The 18 counterfactual test poses remain untouched during material training and model selection.

---

## Data

```text
training poses:       25
counterfactual test:  18
training visible:     12
training hidden:      13
test visible:         12
test hidden:           6
operator seeds:        8
```

The six test poses nearest a beacon-visibility flip have mean geometric boundary distance `0.215`. The six farthest have mean distance `0.825`.

---

## Global visibility result

| representation | accuracy | balanced accuracy | Brier error |
|---|---:|---:|---:|
| **fixed operator coordinates** | **0.6667** | **0.7083** | **0.2402** |
| image-trained `g` | 0.5556 | 0.5417 | 0.2445 |
| pose kernel ridge | 0.5000 | 0.6250 | 0.4464 |
| compact 5-RBF pose map | 0.3333 | 0.5000 | 0.5635 |
| direct pose | 0.3333 | 0.5000 | 1.0025 |
| compact interaction pose map | 0.3333 | 0.5000 | 1.0693 |

The surprising result is not that the image-trained substrate wins. It does not.

The surprising result is that the **untrained/fixed operator coordinates already provide a useful visibility partition**. Their Brier error is much lower than the tested smooth pose-only baselines.

Image training then makes that partition worse.

---

## Multi-seed paired effect

Across all eight operator initializations:

```text
fixed operator balanced accuracy
median  0.708333
mean    0.682292
range   0.625000 .. 0.708333

image-trained operator balanced accuracy
median  0.541667
mean    0.541667
range   0.541667 .. 0.541667
```

Brier error:

```text
fixed operator
median  0.240698
mean    0.240935

image-trained operator
median  0.244520
mean    0.244632
```

Paired image-training effect:

```text
Brier improvements             1 / 8
balanced-accuracy improvements 0 / 8
median Brier improvement      -0.004198   (worse)
median balanced-acc change    -0.166667   (worse)
```

So this is not a seed accident. The image objective consistently bends the material away from the representation that was useful for the visibility event.

---

## Near the actual occlusion boundary

The original reason for building this probe was to ask whether any learned advantage appears exactly where geometry should matter most.

For the nearest six counterfactual poses:

| representation | accuracy | balanced accuracy | Brier error |
|---|---:|---:|---:|
| fixed operator | 0.1667 | 0.1000 | **0.3034** |
| image-trained `g` | 0.1667 | 0.1000 | 0.3349 |
| pose kernel ridge | 0.3333 | **0.6000** | 0.6500 |
| 5-RBF pose | 0.1667 | 0.5000 | 0.9925 |
| direct pose | 0.1667 | 0.5000 | 1.1476 |

The learned substrate is **worse than the fixed substrate near the boundary**.

This directly fails the geometry-specific prediction.

Far from the boundary, the image-trained operator gets a slightly lower Brier error than fixed (`0.2114` vs `0.2144`), which is consistent with the reconstruction result: image training helps the smooth part of the view field, not the discontinuity.

---

## Verdict

```text
VISIBILITY_TOPOLOGY_SIGNAL_NOT_SHOWN
```

More specifically:

```text
fixed operator coordinates carry visibility signal     YES
image-only g training improves image reconstruction     YES, slightly and 8/8 seeds
image-only g training improves visibility topology      NO
image-only g training helps near occlusion boundary     NO
learned geometry / world topology in g                  NOT SHOWN
```

This sharpens the interpretation of the reconstruction result.

The current material-learning objective is not discovering hidden room topology. It is optimizing a compact smooth coordinate warp for global image-basis prediction, and that optimization actually **damages** a visibility boundary that the fixed resolvent coordinates represented better.

---

## What follows

The next experiment should change the **image objective**, not make the room more complicated.

A natural next gate is still label-free with respect to geometry:

```text
same route images
    -> emphasize local high-gradient / disocclusion-sensitive image residuals
    -> train g without visibility labels
    -> freeze g
    -> repeat the same held-out visibility probe
```

If edge-aware image training improves reconstruction *and* stops destroying the visibility partition, that would be the first evidence that the substrate can be pushed toward scene topology using only visual prediction error.

If it still smooths across the boundary, the material mechanism is behaving as a generic low-dimensional function approximator rather than a useful world model.
