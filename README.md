# SplatWorld4

Fresh continuation of the SplatWorld operator experiments.

## Repository note

This repository is intentionally being used as a **random/fresh working folder**. Older local SplatWorld4 material existed elsewhere but was never uploaded here, so none of that untracked material should be treated as part of this repository's provenance or as an implicit baseline. Work in this repo starts from what is actually committed here.

The room/counterfactual line established a useful boundary: static edge-aware visual learning can preserve some held-out visibility structure, while route derivatives and pair-transition training still do not beat strong smooth/pose controls at the actual occlusion boundary. See `ROOM_RESULTS.md`, `EDGE_AWARE_RESULTS.md`, `VIEW_CHANGE_RESULTS.md`, and `TRANSITION_RESULTS.md`.

## Current experiment: make z physical

The next experiment changes the question instead of adding another room loss.

The operator is already a three-coordinate family,

```text
(x, y, z) -> M_g(x,y,z)
```

but three coordinates alone do not imply 3-D. `obj_xyz_world.py` and `xyz_operator_train.py` give the third coordinate an explicit physical meaning: **camera depth in a perspective 3-D scene**.

Several OBJ meshes are placed at different world depths, creating real parallax and occlusion. The camera is sampled on a 5 x 3 x 5 `(x,y,z)` lattice. Entire intermediate z planes are withheld:

```text
z0  train
z1  TEST ONLY
z2  train
z3  TEST ONLY
z4  train
```

The learned object remains one small material vector `g`; gradient training uses a material-conserving softmax parameterization. Controls include fixed `g`, no-z, deliberately wrong-z, direct xyz MLP, direct xy MLP, and the train-only image-basis ceiling.

Read [`XYZ_DEPTH_PLAN.md`](XYZ_DEPTH_PLAN.md) for the exact claim boundary and run commands.

### RTX 3060 run

```bash
python3.13 -m pip install -r requirements-xyz.txt
python3.13 obj_xyz_world.py --download spot rounded_cube avocado
python3.13 xyz_operator_train.py --run --device cuda --steps 2500 --seeds 3 --out-dir xyz_depth_out
```

You can also use arbitrary local Wavefront OBJ files:

```bash
python3.13 xyz_operator_train.py --run --device cuda --obj model1.obj model2.obj model3.obj
```

The decisive first question is not whether the operator beats every neural baseline. It is whether **correct z semantics improve prediction on wholly unseen depth planes compared with removing z or deliberately assigning the wrong z**. If that survives, the following gate will remove explicit z labels and ask whether parallax and occlusion can infer the hidden depth coordinate.
