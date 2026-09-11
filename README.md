# SplatWorld4

Fresh continuation of the SplatWorld operator experiments.

## Repository note

This repository is intentionally being used as a **random/fresh working folder**. Older local SplatWorld4 material existed elsewhere but was never uploaded here, so none of that untracked material should be treated as part of this repository's provenance or as an implicit baseline. Work in this repo starts from what is actually committed here.

The first target is the harder follow-up to the SplatWorld3D orbit control:

> Can one shared address-conditioned material substrate, queried by camera `(x, z, yaw)`, use sparse observed routes through an occluding room to predict **counterfactual views at unvisited positions** better than direct pose features, local interpolation, and matched smooth/random controls?

The experiment is designed to make a one-dimensional smooth camera-orbit explanation impossible. It will include camera translation, pillars/occlusion, disocclusion, held-out counterfactual poses, multi-seed controls, and evaluation binned by distance to an occlusion boundary.
