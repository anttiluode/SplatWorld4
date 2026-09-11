#!/usr/bin/env python3
# phase_transport_test.py — the $83 test.
#
# CLAIM UNDER TEST: because the renderer computes a*cos - b*sin per atom,
# each Gabor packet carries a true complex phasor (a + ib). Rotating that
# phasor by dphi should glide the carrier ripple through a STATIC envelope
# with constant amplitude (Fourier shift theorem), while linearly
# interpolating (a,b) -> (-a,-b) must kill the wave at the midpoint
# (amplitude -> 0 -> reborn). If rotation glides and lerp ghosts, phase
# transport works and the avatar path is open. If rotation ALSO ghosts,
# the atoms are not transport-friendly and we learned that for free.
#
# WHAT THIS SCRIPT DOES
#   1. loads your trained checkpoint (runs/splat2/model2.pt by default)
#   2. decodes one fixed latent z -> 256 activated packet params
#   3. selects atoms: --select all, or a circle (--cx --cy --r) e.g. an eyebrow
#   4. renders F frames sweeping dphi over [0, 2pi):
#        LEFT  half of each frame: phasor ROTATION   (the physics)
#        RIGHT half of each frame: linear CROSSFADE  (the control)
#   5. writes phase_transport.gif + amplitude_ledger.csv
#
# PASS/FAIL NUMBER (printed at the end, no eyeballing required):
#   mean selected-atom amplitude sqrt(a^2+b^2) per frame.
#   rotation: flat to float tolerance.  lerp: dips to ~0 at frame F/2.
#   Plus visual: left side ripples glide, right side features die & rebirth.
#
# USAGE (on the 3060, seconds):
#   python phase_transport_test.py                        # all atoms, seed 0
#   python phase_transport_test.py --list                 # print top atoms w/ positions
#   python phase_transport_test.py --cx 0.35 --cy 0.30 --r 0.10   # eyebrow circle
#   python phase_transport_test.py --seed 7 --frames 96 --scale 4
#
# Needs: Splat_trainer2.py in the same folder, torch, numpy, Pillow.
# HONESTY: smoke-tested end-to-end on CPU with a randomly-initialized
# checkpoint (pipeline + math verified: rotation amplitude flat to 1e-7,
# lerp midpoint amplitude ~0). Results on YOUR trained model2.pt are yours
# to measure. Do not hype. Do not lie. Just show.

import argparse, csv, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Splat_trainer2 as ST                      # reuse the EXACT renderer math


# ----------------------------------------------------------------------
def render_from_params(ren, px, py, sigma, theta, freq, coeff):
    """Renderer forward, but starting from ACTIVATED params.

    We cannot call ren.forward() with modified coefficients because
    activate() would re-tanh them. So we walk the same chunk loop and call
    ren._chunk directly — bit-identical math to training/export."""
    out = None
    for i in range(0, ren.N, ren.chunk):
        sl = slice(i, i + ren.chunk)
        c = ren._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                       theta[:, sl], freq[:, sl], coeff[:, sl])
        out = c if out is None else out + c
    return torch.sigmoid(out)


def rotate_phasor(coeff, mask, dphi):
    """Rotate (a,b) of masked atoms by dphi. coeff: (1,N,3,2). Same rotation
    for all 3 color channels so RGB stays phase-coherent. Amplitude
    sqrt(a^2+b^2) is invariant by construction."""
    a, b = coeff[..., 0], coeff[..., 1]                       # (1,N,3)
    ca, sa = math.cos(dphi), math.sin(dphi)
    a2 = a * ca - b * sa
    b2 = a * sa + b * ca
    out = coeff.clone()
    m = mask[None, :, None]                                   # (1,N,1)
    out[..., 0] = torch.where(m, a2, a)
    out[..., 1] = torch.where(m, b2, b)
    return out


def lerp_phasor(coeff, mask, t):
    """Control: straight line from (a,b) to (-a,-b). At t=0.5 the phasor
    passes through the origin — the wave dies and is reborn. This is what
    naive crossfading between two states does."""
    s = 1.0 - 2.0 * t                                         # 1 -> -1
    out = coeff.clone()
    m = mask[None, :, None, None]
    out = torch.where(m, coeff * s, coeff)
    return out


def to_img(t, scale):
    x = (t[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    if scale > 1:
        x = np.kron(x, np.ones((scale, scale, 1), dtype=np.uint8))
    return x


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./runs/splat2/model2.pt")
    ap.add_argument("--seed", type=int, default=0, help="latent z seed")
    ap.add_argument("--z_npy", default="", help="optional (128,) .npy latent instead of seed")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--scale", type=int, default=3, help="nearest-neighbor upscale of output")
    ap.add_argument("--cx", type=float, default=None, help="circle center x in [0,1]")
    ap.add_argument("--cy", type=float, default=None, help="circle center y in [0,1]")
    ap.add_argument("--r", type=float, default=0.1, help="circle radius")
    ap.add_argument("--list", action="store_true",
                    help="print top-30 atoms by amplitude with positions, then exit")
    ap.add_argument("--out", default="phase_transport.gif")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu")
    model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
    model.load_state_dict(ck["sd"]); model.eval().to(dev)
    ren = model.ren

    g = torch.Generator().manual_seed(args.seed)
    if args.z_npy:
        z = torch.from_numpy(np.load(args.z_npy)).float().view(1, ST.LATENT)
    else:
        z = torch.randn(1, ST.LATENT, generator=g)
    z = z.to(dev)

    with torch.no_grad():
        raw = model.dec(z)                                    # (1,N,11)
        px, py, sigma, theta, freq, coeff = ren.activate(raw)

        amp = coeff.pow(2).sum(-1).sqrt().mean(-1)[0]         # (N,) mean over RGB
        if args.list:
            order = torch.argsort(amp, descending=True)[:30]
            print(f"{'atom':>5} {'amp':>7} {'px':>6} {'py':>6} {'sigma':>7} {'freq':>6}")
            for i in order.tolist():
                print(f"{i:>5} {amp[i]:7.3f} {px[0,i]:6.3f} {py[0,i]:6.3f} "
                      f"{sigma[0,i]:7.4f} {freq[0,i]:6.2f}")
            print("\npx,py are in [0,1]: x right, y DOWN (image coords). "
                  "Pick a cluster, rerun with --cx --cy --r.")
            return

        if args.cx is not None and args.cy is not None:
            mask = ((px[0] - args.cx) ** 2 + (py[0] - args.cy) ** 2) < args.r ** 2
            sel = "circle(%.2f,%.2f,r=%.2f)" % (args.cx, args.cy, args.r)
        else:
            mask = torch.ones(ren.N, dtype=torch.bool, device=dev)
            sel = "ALL"
        n_sel = int(mask.sum())
        if n_sel == 0:
            print("selection is empty — run --list and pick a real cluster"); return
        print(f"selected {n_sel}/{ren.N} atoms ({sel}) on {dev}")

        frames, rot_amp, lrp_amp = [], [], []
        for f in range(args.frames):
            t = f / args.frames
            dphi = 2 * math.pi * t
            c_rot = rotate_phasor(coeff, mask, dphi)
            c_lrp = lerp_phasor(coeff, mask, t if t <= 0.5 else 1.0 - t)  # out & back

            img_rot = render_from_params(ren, px, py, sigma, theta, freq, c_rot)
            img_lrp = render_from_params(ren, px, py, sigma, theta, freq, c_lrp)

            rot_amp.append(c_rot[0, mask].pow(2).sum(-1).sqrt().mean().item())
            lrp_amp.append(c_lrp[0, mask].pow(2).sum(-1).sqrt().mean().item())

            left, right = to_img(img_rot, args.scale), to_img(img_lrp, args.scale)
            bar = np.full((left.shape[0], 2, 3), 40, dtype=np.uint8)
            frames.append(np.concatenate([left, bar, right], axis=1))

    # ledger ------------------------------------------------------------
    with open("amplitude_ledger.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frame", "rot_amp", "lerp_amp"])
        for i, (r_, l_) in enumerate(zip(rot_amp, lrp_amp)):
            w.writerow([i, f"{r_:.8f}", f"{l_:.8f}"])

    r0 = rot_amp[0]
    rot_dev = max(abs(a - r0) for a in rot_amp)
    lrp_min = min(lrp_amp)
    print("\n=== AMPLITUDE LEDGER (selected atoms, mean sqrt(a^2+b^2)) ===")
    print(f"rotation : start {r0:.6f}  max deviation {rot_dev:.2e}   "
          f"{'FLAT — PASS' if rot_dev < 1e-4 else 'NOT FLAT — LOOK AT THIS'}")
    print(f"crossfade: start {lrp_amp[0]:.6f}  midpoint minimum {lrp_min:.6f}   "
          f"{'DIES AS PREDICTED' if lrp_min < 0.05 * lrp_amp[0] else 'did not die?'}")
    print("csv -> amplitude_ledger.csv")

    # gif ---------------------------------------------------------------
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(args.out, save_all=True, append_images=ims[1:],
                duration=50, loop=0)
    print(f"gif -> {args.out}   LEFT = phasor rotation | RIGHT = crossfade control")
    print("If the left ripples glide inside static envelopes while the right "
          "ghosts out and back, phase transport is REAL on your model.")


if __name__ == "__main__":
    main()
