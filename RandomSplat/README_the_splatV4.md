# the_splatV4 — two-band flow-probes

**PerceptionLab / Antti Luode (Helsinki), with Claude (Fable 5). July 2026.**

> Do not hype. Do not lie. Just show.

V3 proved flow-probes hold a manifold and survive lighting drift. But it threw
one band of identical probes at the whole frame. Live, the *shoulders* — a big,
smooth, slow structure — were the channel you could actually steer by, while the
face fought back (a table still renders a face). The reason is frequency: the
face is high spatial frequency (detail), the body and surround are low spatial
frequency (envelope). V4 splits the afferent into two bands matched to the
manifold's own frequency organization, and routes each band to its own packets.

## What closes, per band

```
d_coarse = LK(pyramid, 11px window)   # the envelope's motion (body / pose)
d_fine   = LK(full-res, 5px window)   # the detail's motion  (face)

coarse packets: freq <= median   fine packets: freq > median

loss = w_coarse * prec_c * || R_coarse(p+d_c; z) - R_coarse(p; z_prev) ||^2   # HIGH weight
     + w_fine   * prec_f * || R_fine  (p+d_f; z) - R_fine  (p; z_prev) ||^2   # LOW weight, prec_f capped
```

The coarse band is **world-driven**: reliable low-frequency motion the manifold
should honor, so it corrects hard — the belief tracks your lean, turn, shoulder
rise, your position. The fine band is **prior-dominated**: it corrects weakly
(`w_fine=0.25`, `prec_f` capped at 0.5), so the face stays a soft frontal
amalgam, honest about being a projection. That division — world-driven vs
prior-driven — is now in the loss weighting, not an accident of the loop.

## The bet (selftest [E])

Coarse-band pose tracking should **hold** under a high-frequency surround
distractor (or lighting drift) that corrupts the *fine* band, because the coarse
band never listens to the channel those conditions wreck. If `both <= coarse`
under the distractor, or coarse jumps from its clean value, the two-band claim
is dead. Run `python the_splatV4.py --selftest` to get the number.

## Honest ledger — where each claim was checked

- **[V, verified in V3, carried over verbatim]** strict `.pt` compatibility
  (renders bit-identical to the repo class), `render_probes` matches the full
  render to float precision, single-band LK recovers a known pixel shift
  exactly, flow>color, flow holds under luminance drift.
- **[V, verified here in NumPy]** the one *new* mechanism — band split by the
  packets' own `freq`, two-band LK, per-band flow routing — separates two
  independent motions from one frame pair: a low-freq +3px slide reads as
  coarse `[+2.98, 0]`, a high-freq −2px nod reads as fine `[+0.29, −1.74]`.
  And when only low-freq packets move (+2.8px pose), the coarse band reads
  `+2.72px` and the fine band ~0 — routing attributes motion to the right band.
- **[static]** the correction wiring was checked by inspection: coarse loss →
  `coarse_idx` at `w_coarse`; fine loss → `fine_idx` at `w_fine`; `prec_f`
  structurally capped.
- **[UNVERIFIED — run it]** the full torch scorecard ([A] pose tracking, **[E]**
  the distractor bet, [B] lum-drift, [C] K-scaling, [D] slop coasting). The
  sandbox package proxy blocked installing torch this session, so these did not
  run here. On your GPU they take seconds: `python the_splatV4.py --selftest`.
- **[K, open]** the band split is only as meaningful as the trained field's
  frequency organization. On the random stand-in field it is by construction; on
  your real celeba `.pt` it is empirical — the GUI draws coarse probes in cyan
  and fine in magenta so you can *see* whether the low-freq packets really are
  the body and whether your shoulders drive the cyan band. Acquisition is still
  the encoder's job (GIST button); probes hold, they do not find.

## What it could be

A **pose-and-presence cortex at 31MB**: the coarse band tracks lean, turn,
shoulder-rise, approach/retreat — a low-dimensional pose state held by a trickle
of big-feature flow, the regime where the loop is strongest and the monolith is
most wasteful. The frontal-face limitation stops being a bug and becomes the
appearance manifold of a body-level tracker. The two-band observable is
manifold-agnostic: the same split wraps any field you train.

## See what each band draws (do this first)

```bash
python the_splatV4.py --model "face model trained 2 epochs/model.pt" --diagnostic
```

Writes `band_diagnostic.png`: several random faces from your field, each shown as
**FULL | COARSE (low-freq) | FINE (high-freq)**. This is the empirical test of
your frequency hypothesis *before any loop runs* — if the split is meaningful,
the COARSE column is the body/surround envelope (the shoulder channel) and the
FINE column is the face detail. If both columns look like scrambled faces, the
2-epoch field didn't organize frequency → content cleanly, and *that* is the
finding: the band split needs a better-trained manifold, not more loop code.

## Run the loop

```bash
pip install torch numpy pillow          # tkinter ships with most Python; opencv only for --webcam
python the_splatV4.py --selftest        # headless scorecard
python the_splatV4.py                    # GUI, synthetic world
python the_splatV4.py --model "face model trained 2 epochs/model.pt" --webcam
```

GUI: **START**; **BANDS** cycles BOTH → COARSE → FINE (which bands *correct* z);
**VIEW** cycles the belief pane FULL → COARSE → FINE (what you *see* the field
draw, live); **INJECT SLOP**, **precision DYN/FIXED**, **GIST** (encoder
re-anchor), **K** slider. Cyan probes/lines = coarse band, magenta = fine band.
Try: move your shoulders and watch whether the cyan flow lengthens; set VIEW
COARSE to watch the belief hold pose while the face detail drops out.

## Fixed since first cut

The subset renderer originally called the verbatim `ren.activate()`, which adds
the full-size `anchor_logit` buffer and so crashed on a packet subset
(`tensor a (512) must match tensor b (256)` on real 512-packet checkpoints). Now
`render_probes_subset` / `render_full_subset` inline the identical activation
with the anchor **sliced to the band** — verified (in NumPy, at 512 packets) to
return the same activations as the full render restricted to those packets, so
the numbers are unchanged; only the shape bug is gone. The verbatim v1 class is
untouched, so `.pt` compatibility is unaffected.

## The .pt question

`load_v1()` infers `image_size`, `num_packets`, `latent`, `hidden` from the
checkpoint itself and loads `strict=True`. Your original `the_splat` layout
(`enc.*`, `dec.net.*`, `ren.GX/GY/anchor_logit`) loads directly — verified
bit-identical against the repo's `splat_generator.SplatVAE`.
