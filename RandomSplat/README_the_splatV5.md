# the_splatV5 — the octave cascade

**PerceptionLab / Antti Luode (Helsinki), with Claude (Sonnet 5). July 2026.**

> Do not hype. Do not lie. Just show.

V4's diagnostic showed the 2-epoch celeba field split *itself*, unsupervised,
into a shading/luminance channel (low-freq packets) and a clean oriented
contour/edge channel (high-freq packets) — the V1 simple-cell decomposition,
predicted by **Barlow's 1961 efficient-coding hypothesis** and turning up in the
code 65 years later. V4 split at the median into two bands. Cortex is a
*cascade* (V1 → V2 → V4, progressively larger receptive fields). V5 generalizes
to **N log-spaced frequency octaves**, each with its own LK window, precision
cap, and correction weight, plus a **graphical EQ** — a live fader per octave.

## The hierarchy (octave 0 = lowest freq)

```
octave    probes   LK window        weight   prec cap    role
  0 (low)   14      13px on 2x pyr    1.00      1.00      shading / pose / envelope
  1         11      11px on 2x pyr    0.75      0.80      major placement
  2          8       9px full-res     0.50      0.60      orientation refine
  3 (high)   5       5px full-res     0.25      0.40      fine contours (near-immovable)
```

Low octaves are **world-driven**: big windows, high trust — they track the
reliable, high-variance low-frequency motion (your lean, shoulders, position).
High octaves are **prior-dominated**: tiny windows, capped trust — the strong,
low-variance face-contour prior, steered only by strong evidence. The lowest
octave orients everything; each higher octave refines within the basin below it.

## The graphical EQ (the "sigh" idea, folded in)

Your old FFT tool taught that low frequencies carry the gist by filtering pixels
in Fourier space. The splat field *already is* that decomposition — each Gabor
packet is a localized frequency atom — so an EQ over packet-frequency is a live
mixing board on the manifold's octaves. Each octave gets a vertical slider (gain
0→1) that gates **both** its render contribution and its correction gradient.
Pull the top sliders down: the gist (pose, shading) survives on the low bands.
Pull the bottom sliders down: identity/detail drops out, structure holds.

## Falsifiable claims (`--selftest`)

- **[A]** all-octaves tracks pose better than open-loop.
- **[E] cascade inheritance — the headline.** The highest octave alone should
  become *more steerable* with the low octaves ON than with them OFF, because it
  inherits orientation from below. Holds iff `all ≤ high-octave-alone`. If
  high-alone is already as good, the cascade bought nothing over independent
  bands — and the ledger says so.
- **[B]** the low octave holds pose under luminance drift (carried from V4).
- **[D]** dynamic precision coasts through slop.

## Honest ledger — where each piece was checked

- **[V, verified in V2/V3/V4, carried verbatim]** strict `.pt` compatibility
  (bit-identical renders), `render_probes`/`render_full_subset` match the full
  render with the anchor sliced to the subset (the V4 512-vs-256 crash is fixed
  here), single-band LK recovers a known pixel shift exactly, the
  precision/prior/saccade loop, and the K-slider draw guard.
- **[V, verified here in NumPy at real 512-packet dims]** the new octave logic:
  `octave_bands` partitions all 512 packets into 4 log-spaced bands (128 each,
  no overlap, none dropped); the per-octave schedules are monotone (low = more
  probes / higher weight / higher trust); LK windows shrink 13px→5px with a
  pyramid on the low half; the EQ correctly gates a faded octave out of *both*
  correction and render; the belief EQ-mix is additive (no division, all-zero EQ
  is safe flat gray); the [E] routing differs only by which octaves' gradients
  are summed — exactly the inheritance test.
- **[UNVERIFIED — run it]** the full torch scorecard ([A], **[E]**, [B], [D])
  did not run here: the sandbox package proxy blocked installing torch this
  session. On your GPU it is seconds: `python the_splatV5.py --selftest`.
- **[K, open]** the octave split is only as meaningful as the trained field's
  frequency organization — empirical on your `.pt`. Run `--diagnostic` first to
  *see* the cascade (FULL | O0 | O1 | O2 | O3): if O0 is shading and O3 is fine
  contours with the middle bands interpolating, the cascade is real structure.
  Acquisition is still the encoder's job (GIST); the cascade lets higher octaves
  inherit the low octave's orientation, which is what [E] measures.

## Run

```bash
pip install torch numpy pillow          # opencv only for --webcam; tkinter ships with Python
python the_splatV5.py --model "face model trained 2 epochs/model.pt" --diagnostic   # SEE the cascade first
python the_splatV5.py --selftest        # the scorecard (run [E] on your GPU)
python the_splatV5.py --model "face model trained 2 epochs/model.pt" --webcam
python the_splatV5.py --octaves 3 ...   # choose band count (default 4)
```

GUI: **START**; the **OCTAVE EQ** slider bank (cyan=low → warm=high) gates each
octave's render + correction live; **VIEW** cycles the belief pane MIX → O0 → O1
… so you can watch one octave at a time; **INJECT SLOP**, **prec DYN/FIX**,
**GIST**, **K** slider. Probes and flow lines are colored by octave. Try: drop
the top sliders and move around — the low-octave belief holds your pose; raise
them and the face detail returns.

## The .pt question

`load_v1()` infers `image_size`, `num_packets`, `latent`, `hidden` from the
checkpoint and loads `strict=True`. Your `the_splat` layout loads directly —
verified bit-identical against the repo's `splat_generator.SplatVAE`.
