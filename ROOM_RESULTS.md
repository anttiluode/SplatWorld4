# SplatWorld4 room result — counterfactual views behind occlusion

## Repository provenance

SplatWorld4 is being used here as a **fresh/random working folder**. Older local material existed but was never uploaded to this repository, so this experiment is built only from files actually committed here. The analytic room is a new self-contained reconstruction of the intended translated/occluding test, not a claim that the historical local `worldplus` files are present.

## Question

Can one shared material vector `g`, exposed through an address-conditioned resolvent and queried by camera `(x,z,yaw)`, use sparse traversed routes to predict **counterfactual views at unvisited interior positions** better than direct pose models and generic smooth interpolation?

Unlike the SplatWorld3D orbit, this room has camera translation, three pillars, hard occlusion/disocclusion, sparse route observations, and held-out interior viewpoints.

The important result is not the first headline-looking win. The important result is what survives stronger controls.

---

## Design

Training contains **25** camera poses on three sparse routes around the outside of the room. Test contains **18** interior counterfactual poses that are never visited by those routes.

A 12-dimensional image basis is fitted from training images only. Test images are not used for:

- basis fitting;
- ridge selection;
- `g` selection;
- random/operator seed selection;
- RBF width selection;
- kernel width selection.

`g` is changed only by conservative edge-to-edge material transfers. Its objective is blocked four-fold prediction inside the training routes.

Eight operator initializations are run as a paired fixed-vs-learned comparison.

The room also contains a bright beacon on the back wall. Pillars can hide it. For every held-out pose, a geometric search measures the floor-plane distance until beacon visibility flips. This gives an explicit **distance to an occlusion boundary** that the models never receive.

---

## First pass: a small repeatable `g` effect

The first control set looked encouraging:

| method | held-out relative RMSE |
|---|---:|
| learned operator `g`, 8-seed median | **0.595364** |
| random smooth 5-D features, 8-seed median | 0.598748 |
| fixed operator, 8-seed median | 0.600267 |
| direct 5-D pose regression | 0.601722 |
| local 4-NN interpolation | 0.622708 |
| training-basis oracle | 0.336840 |

Learning `g` beat its own fixed initialization in **8/8 paired seeds**.

Median fixed -> learned improvement:

```text
0.600267 -> 0.595364
absolute improvement  0.004976
relative improvement  0.829%
```

The material did not merely stay near initialization: individual runs accepted roughly 170–205 conservative edits and moved `||Delta g||` by about 1.5–1.7.

So there is a real, reproducible effect of changing the substrate under the training-only objective.

That is **not yet the same thing as learning room geometry**.

---

## Stronger attack: pose-only nonlinear controls

The small advantage was then attacked with stronger controls fixed without looking at test error.

### Capacity-matched controls

All of these use five nonlinear pose features plus an intercept, matching the five operator response coordinates plus intercept:

| compact representation | held-out relative RMSE |
|---|---:|
| learned operator, 8-seed median | **0.595364** |
| direct pose + one interaction | 0.601169 |
| five training-selected RBF centers | 0.605311 |

At the same compact feature budget, the operator substrate remains competitive and wins this particular test.

### Strong smooth control

A kernel-ridge pose model with one kernel center per training pose gives:

```text
kernel ridge: 0.589401
learned operator median: 0.595364
```

The stronger pose-only model therefore beats the learned operator.

This is not capacity matched, but it matters for the interpretation: the held-out images do not require a learned physical substrate to obtain the best tested prediction.

---

## The occlusion-boundary test is the decisive failure

If `g` had learned something specifically useful about visibility geometry, its advantage over fixed operator coordinates should become largest near a real occlusion boundary.

It does the opposite.

Median fixed-operator minus learned-operator per-pose improvement:

| distance bin | mean boundary distance | `g` gain |
|---|---:|---:|
| near | 0.215 | **0.002231** |
| middle | 0.360 | **0.003807** |
| far | 0.825 | **0.005597** |

Correlation between boundary distance and `g` gain:

```text
r = +0.223
```

The gain grows, if anything, **away from the occlusion boundary**.

Near the boundary the strongest smooth pose-only model also slightly beats the learned operator:

```text
kernel ridge          0.590140
learned operator      0.590878
fixed operator        0.593109
direct compact pose   0.595350
```

So the learned substrate is not showing the signature we asked for.

---

## What this means

The room experiment is more informative than the orbit, but the strongest interpretation still fails.

What survived:

```text
external rendered room                          YES
camera translation                              YES
hard occlusion/disocclusion                     YES
unvisited counterfactual test viewpoints        YES
training-only model selection                   YES
multi-seed paired g improvement                 YES (8/8)
compact learned operator beats compact controls YES on this run
learned operator beats strong pose kernel       NO
g gain is strongest near occlusion boundaries   NO
geometry-specific world memory in g             NOT SHOWN
```

The most defensible current interpretation is:

> changing `g` learns a small, highly repeatable nonlinear coordinate warp that improves compact view prediction, but the improvement is not localized where occlusion geometry matters and is beaten by a higher-capacity smooth pose-only kernel.

That is closer to **compact nonlinear representation learning** than to a demonstrated internal 3-D world model.

The stronger-control verdict is therefore:

```text
PROVISIONAL_ROOM_SIGNAL_DOES_NOT_SURVIVE_STRONG_INTERPRETATION
```

---

## What follows

The next useful experiment should not merely make the renderer harder. It should ask a target that smooth view regression cannot blur away.

A clean next gate is **visibility topology**:

1. train from the same sparse routes;
2. freeze the image-trained substrate;
3. ask whether its compact response predicts a held-out discrete visibility event — e.g. whether the beacon is visible, which side of a pillar owns the first-hit ray, or which object becomes newly disoccluded;
4. compare against direct pose, capacity-matched RBFs, and full kernel ridge;
5. score especially the held-out poses nearest the geometric boundary.

If `g` contains useful scene topology, that is where it should finally separate from a generic smooth coordinate map. If it does not, the repeated small reconstruction gain should be treated as generic function approximation, not world memory.
