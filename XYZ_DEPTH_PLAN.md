# Gate 5 — teach the operator a physical z coordinate

The question is deliberately narrower than "did 3-D emerge?":

> If the operator is queried by `(x,y,z)`, can its third coordinate be trained to mean **physical camera depth**, and can one learned material vector `g` predict views on entire camera-z planes that were never shown during training?

This is Phase A. The camera coordinates are known because the training world is rendered from OBJ geometry. A later phase can remove explicit z and ask whether parallax / occlusion can infer it.

## Scene

`obj_xyz_world.py` downloads permissively licensed OBJ meshes and puts several objects at different world depths. The default scene uses:

- `spot`
- `rounded_cube`
- `avocado`

from Adobe's `lagrange-test-data`; those entries are listed there as CC0-1.0.

The renderer is a small NumPy perspective triangle rasterizer with a z-buffer. It ignores texture files on purpose: the first question is geometry, parallax and occlusion, not photorealism.

The default camera lattice is:

```text
x: 5 locations
y: 3 locations
z: 5 depth planes
```

The meshes themselves occupy different world-z positions, so lateral camera motion produces depth-dependent parallax and occlusion.

## The z test

Entire intermediate depth planes are forbidden during learning:

```text
z plane 0    TRAIN / validation
z plane 1    TEST ONLY
z plane 2    TRAIN / validation
z plane 3    TEST ONLY
z plane 4    TRAIN / validation
```

The image PCA basis is also fit on training images only. Thus the final test asks for genuine interpolation through an unobserved physical camera-depth slice.

The operator uses a differentiable, material-conserving parameterization:

```text
g = g_min + free_budget * softmax(theta)
```

so gradient training can use the RTX 3060 while total material remains fixed.

## Controls

The run trains the following with the same train/validation split:

| model | question |
|---|---|
| `full_xyz` | learned `g`, correct x/y/z |
| `fixed_xyz` | does learning `g` itself help? |
| `xy_only` | can x/y solve the held-out z planes anyway? |
| `wrong_z` | does a wrong depth correspondence destroy the benefit? |
| `direct_xyz` | can a small ordinary pose MLP do better? |
| `direct_xy` | how much does z help a generic model? |
| `basis_oracle` | floor imposed by the train-only image basis |

The first z claim requires at minimum:

```text
full_xyz < xy_only
full_xyz < wrong_z
```

on the two unseen depth planes. Beating `direct_xyz` would be stronger and is reported separately; it is not silently assumed.

## Run on the Ryzen 5500 / RTX 3060 12 GB machine

Install:

```bash
python3.13 -m pip install numpy pillow torch
```

Download / preview the meshes:

```bash
python3.13 obj_xyz_world.py --download spot rounded_cube avocado
python3.13 obj_xyz_world.py --preview --models spot rounded_cube avocado
```

Run the full first experiment:

```bash
python3.13 xyz_operator_train.py --run \
  --models spot rounded_cube avocado \
  --device cuda \
  --height 64 --width 64 \
  --steps 2500 --seeds 3 \
  --out-dir xyz_depth_out
```

On Windows `cmd.exe`, put that on one line or replace `\` with `^` line continuations.

The renderer is CPU-side and cached after the first pass. Training is tiny compared with the GPU memory available: the 3060 is mostly useful because PyTorch's batched complex solves and the six comparison models run quickly there.

For a longer attack:

```bash
python3.13 xyz_operator_train.py --run --device cuda --steps 6000 --seeds 8 --out-dir xyz_depth_8seed
```

Use arbitrary local OBJ files instead of presets:

```bash
python3.13 xyz_operator_train.py --run --device cuda --obj model1.obj model2.obj model3.obj
```

## Outputs

The run writes:

```text
xyz_depth_out/
  ground_truth_xyz_atlas.png
  xyz_comparison.png
  xyz_metrics.json
  render_cache_*.npz
```

`ground_truth_xyz_atlas.png` shows the actual 3-D camera lattice. `xyz_comparison.png` puts ground truth, full operator, no-z, wrong-z and pose controls side-by-side specifically on the two unseen z planes.

## Smoke result, not the real OBJ result

A tiny local development run using only three procedural cubes (`20x20`, 180 steps, one seed) produced:

```text
full_xyz     0.3463
fixed_xyz    0.3704
xy_only      0.3726
wrong_z      0.4021
direct_xyz   0.3241
direct_xy    0.3818
basis_oracle 0.2947
```

So the implementation is capable of using z and learning `g` in the intended direction, while a generic xyz MLP still wins on that toy. This is only a smoke test; no claim about downloadable OBJ geometry is made until the real multi-model run is executed.

## What comes after Phase A

If the real OBJ run says z matters, the next experiment removes z labels from the learner.

Then camera motion supplies the training relation:

```text
same persistent object
+ known camera displacement
-> observed parallax / occlusion change
```

and a hidden z coordinate must explain why near objects move more than far objects and why one surface occludes another. That is the point at which "z" becomes inferred depth rather than a supervised camera coordinate.
