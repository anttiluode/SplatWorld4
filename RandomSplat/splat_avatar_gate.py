#!/usr/bin/env python3
# =============================================================================
# splat_avatar_gate.py — the avatar gate, wired to the REAL model2.pt
#
# WHAT CHANGED vs avatar_gate_test.py (the Opus harness):
#   1. No adapter stubs. Loads runs/splat2/model2.pt, uses the trainer's own
#      Decoder + GaborRenderer. Bit-identical math to training.
#   2. P4 was vacuous there (d_B(t=1)=0 by construction for ANY param morph).
#      The real avatar gate is ROAD AGREEMENT: the avatar is driven by z, so
#      transport is useful iff transport(A->B, t) stays close to the
#      decoder's own road decode(lerp(zA,zB,t)). If it does, you can drive z
#      at low rate and fill frames with cheap transport, no melt.
#   3. Fire is measured on the PRE-SIGMOID field (its std), because sigmoid +
#      any normalization masks amplitude collapse (this is exactly why the
#      demo run showed no fire).
#
# FOUR CONDITIONS rendered per frame (and written to one GIF, side by side):
#   lerp     geometry-lerp + linear crossfade of the (a,b) phasors  [baseline]
#   phase    geometry-lerp + magnitude-lerp + shortest-arc phase    [transport]
#   scramble phase-transport toward a phase-scrambled copy of B     [control]
#   latent   decode((1-t)zA + t zB)                                  [the road]
#
# REGISTERED PREDICTIONS (comparative, scored automatically):
#   R1 ATOM FIRE     mean per-atom |phasor| at t=0.5: lerp dips below phase
#                    by the cos(dphi/2) factor; phase stays flat (>0.98 of lerp
#                    endpoints' mean). Printed, pass iff lerp_mid < 0.9*phase_mid.
#   R2 FIELD FIRE    pre-sigmoid field std: lerp midpoint < 0.9 * phase midpoint.
#   R3 STAYS SHARP   phase image sharpness stays > 0.5 * endpoint mean, all t.
#   R4 ROAD AGREE    max_t mse(phase(t), latent(t)) < 0.35 * mse(A,B)   << GATE
#                    (transport road hugs the decoder road tighter than the
#                     faces differ from each other; report the exact ratio)
#   R5 CONTROL       scramble's road disagreement > 2x phase's road
#                    disagreement (coherent phase is WHY the road is hugged).
#
#   AVATAR VIABLE iff R2 & R3 & R4 & R5.  R4-fail with R2/R3-pass means
#   coherent-but-off-road: transport moves cleanly but the decoder's manifold
#   bends away from straight parameter lines -> fix is trajectory curvature
#   (piecewise keyframes closer together), not phase.
#
# USAGE (3060, ~seconds):
#   python splat_avatar_gate.py                          # seeds 0,1
#   python splat_avatar_gate.py --seedA 3 --seedB 11     # a different pair
#   python splat_avatar_gate.py --pairs 5                # 5 random pairs, table
#
# HONESTY: smoke-tested end-to-end on CPU against a random-weight checkpoint
# in the trainer's exact save format (pipeline + scoring verified; verdicts on
# a random model are meaningless and were not read as results). Numbers on
# your trained model are yours to measure. Do not hype. Do not lie. Just show.
# V2 NOTE (post seed(0,1) run): R3 v1 failed for a harness reason — with a
# 9.9x endpoint sharpness asymmetry the threshold sat above face A itself.
# R3 v2 (pointwise vs the latent road, factor 0.7) was registered BEFORE any
# v2 run. Also learned: lerp hugs the decoder road TIGHTER than transport
# (the MLP road is approximately a coefficient crossfade — the decoder never
# learned phase rotation), so transport's road deviation is departure in the
# SHARP direction, correcting the road's mid-path VAE blur. R4 still gates on
# transport staying within 0.35*gap of the road (continuity), and the R4 line
# now prints lerp's road-dev beside it so the trade is visible per pair.
# =============================================================================
import argparse, csv, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Splat_trainer2 as ST


# ---------------------------------------------------------------- rendering
def render_field(ren, px, py, sigma, theta, freq, coeff):
    """Chunk loop from the trainer, but returns (pre_sigmoid_field, image)."""
    out = None
    for i in range(0, ren.N, ren.chunk):
        sl = slice(i, i + ren.chunk)
        c = ren._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                       theta[:, sl], freq[:, sl], coeff[:, sl])
        out = c if out is None else out + c
    return out, torch.sigmoid(out)


def activate(ren, raw):
    return ren.activate(raw.float())


# ---------------------------------------------------------------- morph ops
def _arc(a, b, t):
    """shortest-arc interpolation for angles."""
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + t * d


def geom_lerp(PA, PB, t):
    """Interpolate geometry: px,py,sigma,freq linear; theta shortest-arc.
    Same treatment in every condition so only the phasor handling differs."""
    pxA, pyA, sA, thA, fA, _ = PA
    pxB, pyB, sB, thB, fB, _ = PB
    L = lambda a, b: (1 - t) * a + t * b
    return L(pxA, pxB), L(pyA, pyB), L(sA, sB), _arc(thA, thB, t), L(fA, fB)


def coeff_lerp(cA, cB, t):
    """baseline: straight line in (a,b) — partial per-atom cancellation."""
    return (1 - t) * cA + t * cB


def coeff_phase(cA, cB, t):
    """transport: magnitude lerp + shortest-arc phase, per atom per channel."""
    aA, bA = cA[..., 0], cA[..., 1]
    aB, bB = cB[..., 0], cB[..., 1]
    mA = torch.sqrt(aA * aA + bA * bA + 1e-12)
    mB = torch.sqrt(aB * aB + bB * bB + 1e-12)
    phA = torch.atan2(bA, aA)
    phB = torch.atan2(bB, aB)
    m = (1 - t) * mA + t * mB
    ph = _arc(phA, phB, t)
    return torch.stack([m * torch.cos(ph), m * torch.sin(ph)], dim=-1)


def scramble_coeff(cB, seed):
    """control target: same magnitudes as B, random phases."""
    aB, bB = cB[..., 0], cB[..., 1]
    mB = torch.sqrt(aB * aB + bB * bB + 1e-12)
    g = torch.Generator().manual_seed(seed)
    ph = (torch.rand(mB.shape, generator=g) * 2 - 1) * math.pi
    ph = ph.to(cB.device)
    return torch.stack([mB * torch.cos(ph), mB * torch.sin(ph)], dim=-1)


def atom_amp(c):
    """mean per-atom phasor magnitude (over atoms and channels)."""
    return c.pow(2).sum(-1).sqrt().mean().item()


# ---------------------------------------------------------------- metrics
def lap_var(img):
    a = img.mean(1)[0]                                        # (H,W)
    k = (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]
         - 4 * a[1:-1, 1:-1])
    return k.var().item()


def mse(x, y):
    return float(((x - y) ** 2).mean())


# ---------------------------------------------------------------- one pair
def run_pair(model, seedA, seedB, frames, scr_seed, dev, make_frames=False):
    ren = model.ren
    gA = torch.Generator().manual_seed(seedA)
    gB = torch.Generator().manual_seed(seedB)
    zA = torch.randn(1, ST.LATENT, generator=gA).to(dev)
    zB = torch.randn(1, ST.LATENT, generator=gB).to(dev)

    with torch.no_grad():
        PA = activate(ren, model.dec(zA))
        PB = activate(ren, model.dec(zB))
        cA, cB = PA[5], PB[5]
        cScr = scramble_coeff(cB, scr_seed)

        _, imgA = render_field(ren, *PA)
        _, imgB = render_field(ren, *PB)
        gap = mse(imgA, imgB)
        end_sharp = 0.5 * (lap_var(imgA) + lap_var(imgB))

        rows, gif = [], []
        stat = {c: {"field": [], "sharp": [], "d_road": [], "amp": []}
                for c in ("lerp", "phase", "scramble", "latent")}
        for f in range(frames):
            t = f / (frames - 1)
            geo = geom_lerp(PA, PB, t)
            z_t = (1 - t) * zA + t * zB
            Pl = activate(ren, model.dec(z_t))
            _, img_road = render_field(ren, *Pl)

            conds = {
                "lerp":     (geo, coeff_lerp(cA, cB, t)),
                "phase":    (geo, coeff_phase(cA, cB, t)),
                "scramble": (geo, coeff_phase(cA, cScr, t)),
            }
            imgs = {}
            for name, (g_, c_) in conds.items():
                fld, img = render_field(ren, *g_, c_)
                imgs[name] = img
                stat[name]["field"].append(fld.std().item())
                stat[name]["sharp"].append(lap_var(img))
                stat[name]["d_road"].append(mse(img, img_road))
                stat[name]["amp"].append(atom_amp(c_))
                rows.append([f, round(t, 4), name, fld.std().item(),
                             lap_var(img), mse(img, imgA), mse(img, imgB),
                             mse(img, img_road)])
            stat["latent"]["field"].append(0.0)
            stat["latent"]["sharp"].append(lap_var(img_road))
            stat["latent"]["d_road"].append(0.0)
            stat["latent"]["amp"].append(atom_amp(Pl[5]))
            rows.append([f, round(t, 4), "latent", 0.0, lap_var(img_road),
                         mse(img_road, imgA), mse(img_road, imgB), 0.0])

            if make_frames:
                panel = torch.cat([imgs["lerp"], imgs["phase"],
                                   imgs["scramble"], img_road], dim=-1)
                gif.append((panel[0].clamp(0, 1) * 255).byte()
                           .permute(1, 2, 0).cpu().numpy())

    mid = frames // 2
    R = {}
    # R3 v2 (registered 2026-07-21 after the seed(0,1) run exposed v1 as
    # ill-posed: min sharpness sat at t=0 — face A itself — so the endpoint-
    # mean threshold was unpassable on asymmetric pairs). v2 is pointwise
    # against the decoder's own road: transport may never blur below 0.7x
    # what the decoder itself renders at the same t.
    sharp_ratio = [p / (l + 1e-12) for p, l in
                   zip(stat["phase"]["sharp"], stat["latent"]["sharp"])]
    R["R1"] = (stat["lerp"]["amp"][mid] < 0.9 * stat["phase"]["amp"][mid],
               f"atom amp mid: lerp {stat['lerp']['amp'][mid]:.4f} vs "
               f"phase {stat['phase']['amp'][mid]:.4f}")
    R["R2"] = (min(stat["lerp"]["field"]) < 0.9 * min(stat["phase"]["field"]),
               f"field std min: lerp {min(stat['lerp']['field']):.4f} vs "
               f"phase {min(stat['phase']['field']):.4f}")
    R["R3"] = (min(sharp_ratio) > 0.7,
               f"phase/road sharpness pointwise: min {min(sharp_ratio):.2f} "
               f"mean {sum(sharp_ratio)/len(sharp_ratio):.2f} (> 0.70) "
               f"[v1 endpoint-ref was ill-posed, see header]")
    ph_road = max(stat["phase"]["d_road"])
    lp_road = max(stat["lerp"]["d_road"])
    R["R4"] = (ph_road < 0.35 * gap,
               f"max road-dev {ph_road:.5f} vs 0.35*gap {0.35 * gap:.5f} "
               f"(ratio {ph_road / (gap + 1e-12):.2f}; lerp road-dev "
               f"{lp_road:.5f} — if lower, transport deviates sharp-ward, "
               f"see header)")
    sc_road = max(stat["scramble"]["d_road"])
    R["R5"] = (sc_road > 2.0 * ph_road,
               f"scramble road-dev {sc_road:.5f} vs 2x phase {2 * ph_road:.5f}")
    return rows, R, gap, gif


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./runs/splat2/model2.pt")
    ap.add_argument("--seedA", type=int, default=0)
    ap.add_argument("--seedB", type=int, default=1)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--pairs", type=int, default=1,
                    help=">1: run N random pairs, print a pass-rate table")
    ap.add_argument("--out", default="splat_avatar_ledger.csv")
    ap.add_argument("--gif", default="avatar_gate.gif")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
    model.load_state_dict(ck["sd"]); model.eval().to(dev)
    print(f"model {ck['image_size']}px / {ck['num_packets']} packets on {dev}")

    all_rows = [["frame", "t", "cond", "field_std", "sharp",
                 "d_A", "d_B", "d_road"]]
    if args.pairs == 1:
        rows, R, gap, gif = run_pair(model, args.seedA, args.seedB,
                                     args.frames, 123, dev, make_frames=True)
        all_rows += rows
        print(f"\npair seeds ({args.seedA},{args.seedB})  A-B gap {gap:.5f}")
        print("-" * 70)
        names = {"R1": "atom fire (lerp dips) ", "R2": "field fire            ",
                 "R3": "phase stays sharp     ", "R4": "ROAD AGREEMENT << GATE",
                 "R5": "scramble breaks road  "}
        for k in ("R1", "R2", "R3", "R4", "R5"):
            ok, note = R[k]
            print(f"{k} {names[k]} [{'V' if ok else 'K'}]  {note}")
        print("-" * 70)
        viable = all(R[k][0] for k in ("R2", "R3", "R4", "R5"))
        if viable:
            print("VERDICT [V]: transport is coherent AND hugs the decoder's "
                  "road, and the control confirms coherent phase is why. "
                  "Keyframe-driven tiny avatar is on.")
        elif R["R2"][0] and R["R3"][0] and not R["R4"][0]:
            print("VERDICT [~]: coherent but OFF-ROAD — clean motion, but the "
                  "decoder's manifold curves away from straight parameter "
                  "lines. Fix: closer keyframes (smaller z steps), not phase. "
                  "Rerun with more --frames over a shorter z distance to "
                  "measure the usable keyframe spacing.")
        elif not R["R5"][0]:
            print("VERDICT [~]: MECHANISM UNCONFIRMED — scramble control did "
                  "not disagree with the road more than transport does. "
                  "Investigate before trusting.")
        else:
            print("VERDICT [K]: transport did not clear the gate. Read the "
                  "failing lines.")
        # gif
        from PIL import Image
        ims = [Image.fromarray(f) for f in gif]
        ims[0].save(args.gif, save_all=True, append_images=ims[1:],
                    duration=60, loop=0)
        print(f"gif -> {args.gif}  panels: lerp | phase | scramble | latent-road")
    else:
        rng = np.random.default_rng(0)
        passes = {k: 0 for k in ("R1", "R2", "R3", "R4", "R5")}
        for p in range(args.pairs):
            sA, sB = int(rng.integers(0, 10_000)), int(rng.integers(0, 10_000))
            rows, R, gap, _ = run_pair(model, sA, sB, args.frames,
                                       1000 + p, dev)
            all_rows += rows
            line = " ".join(f"{k}:{'V' if R[k][0] else 'K'}" for k in passes)
            ratio = R["R4"][1].split("ratio ")[-1].rstrip(")")
            print(f"pair {p} ({sA},{sB}) gap {gap:.5f}  {line}  road-ratio {ratio}")
            for k in passes:
                passes[k] += int(R[k][0])
        print("\npass rates: " + "  ".join(
            f"{k} {passes[k]}/{args.pairs}" for k in passes))
        print("R4 pass rate is the avatar gate. If it passes on short-gap "
              "pairs and fails on long ones, that's the keyframe-spacing law.")

    with open(args.out, "w", newline="") as fh:
        csv.writer(fh).writerows(all_rows)
    print(f"ledger -> {args.out}")


if __name__ == "__main__":
    main()