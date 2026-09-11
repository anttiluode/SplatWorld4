#!/usr/bin/env python3
# =============================================================================
# avatar_gate_test.py  —  SplatWorld phase-transport A->B gate
#
# PURPOSE
#   The excursion GIF proved phase transport preserves ENERGY + STRUCTURE while
#   moving. It did NOT prove the morph can be AIMED: that steering toward a
#   distinct target face actually ARRIVES at that face. An avatar needs both.
#   This harness runs an A->B morph between two distinct faces under three
#   conditions and scores five pre-registered predictions automatically.
#
# REGISTERED PREDICTIONS  (write down BEFORE running; the script scores them)
#   P1  BASELINE FIRE   lerp contrast collapses near t=0.5   (min < 0.15*endpoint)
#   P2  NO DEAD ZONE    phase contrast never collapses        (min > 0.60*endpoint)
#   P3  STAYS SHARP     phase sharpness never blurs out        (min > 0.50*endpoint)
#   P4  ARRIVES AT B    phase morph reaches B at t=1  (d_B_end < 0.20 * A-B gap)   <-- THE AVATAR GATE
#   P5  CONTROL BREAKS  scrambled-phase reintroduces the fire (min < 0.30*endpoint)
#
#   AVATAR VIABLE  iff  P2 & P3 & P4 pass  AND  P5 breaks (control confirms the
#   mechanism is the coherent phase, not just "holding amplitude").
#   If P4 fails but P2/P3 pass: transport is energy-correct but NOT aimable ->
#   the avatar would drift coherently but couldn't be pointed. That is a
#   diagnosis, not a dead end: it says the fix is in target-matching, not phase.
#
# LEDGER SCHEMA (pre-specified columns)
#   frame, t, cond, rms_contrast, sharp, d_A, d_B, spec_dev
#
# PLUG IN YOUR MODEL
#   Replace the three adapter functions in the ADAPTER block:
#     load_decoder(), decode(z)->packets, render(packets)->HxW float image
#   and point morph_phase() at YOUR real phase-transport operator.
#   Confirm PACKET COLS match your decoder's output layout.
#   Run `--demo` first (synthetic model) to confirm the harness + verdict work.
# =============================================================================
import argparse, csv, numpy as np

# ---------------- PACKET LAYOUT --------------------------------------------
# Each of the N=256 packets is a row: [cx, cy, sigma, theta, freq, mag, phase]
# geometry cols are interpolated linearly in every condition; the (mag,phase)
# complex weight is what the three conditions treat differently.
CX, CY, SIG, THE, FRQ, MAG, PHA = range(7)
NCOL = 7

# ---------------- ADAPTER (replace for real model) -------------------------
def load_decoder(path=None):
    """Return your ONNX/torch decoder. Demo returns a fixed random z->packets map."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((128, 256 * NCOL)) * 0.3
    return {"W": W}

def decode(dec, z):
    """z (128,) -> packets (256, NCOL). Demo: linear map squashed into sane ranges."""
    raw = (z @ dec["W"]).reshape(256, NCOL)
    p = np.empty_like(raw)
    p[:, CX] = np.tanh(raw[:, CX]) * 40 + 48     # center in 96px canvas
    p[:, CY] = np.tanh(raw[:, CY]) * 40 + 48
    p[:, SIG] = 6 + 10 * (0.5 + 0.5 * np.tanh(raw[:, SIG]))
    p[:, THE] = raw[:, THE]                       # orientation (rad)
    p[:, FRQ] = 0.05 + 0.20 * (0.5 + 0.5 * np.tanh(raw[:, FRQ]))
    p[:, MAG] = 0.5 + 0.5 * np.tanh(raw[:, MAG])  # magnitude >= 0
    p[:, PHA] = raw[:, PHA]                        # carrier phase (rad)
    return p

def render(packets, H=96, W=96):
    """Sum of Gabor packets -> real image, normalized to [0,1]. Replace with yours."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    img = np.zeros((H, W))
    for p in packets:
        dx = xs - p[CX]; dy = ys - p[CY]
        ct, st = np.cos(p[THE]), np.sin(p[THE])
        xr = dx * ct + dy * st                    # coord along carrier
        env = np.exp(-(dx * dx + dy * dy) / (2 * p[SIG] ** 2))
        img += p[MAG] * env * np.cos(2 * np.pi * p[FRQ] * xr + p[PHA])
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-9)

# ---------------- MORPH OPERATORS ------------------------------------------
def _lin(a, b, t):
    return (1 - t) * a + t * b

def _geom(pa, pb, t):
    """interpolate geometry cols (everything except the complex weight)."""
    out = np.empty_like(pa)
    for c in (CX, CY, SIG, THE, FRQ):
        out[:, c] = _lin(pa[:, c], pb[:, c], t)
    return out

def morph_lerp(pa, pb, t):
    """CROSSFADE: linear interp of the COMPLEX weight -> magnitude can cancel."""
    out = _geom(pa, pb, t)
    za = pa[:, MAG] * np.exp(1j * pa[:, PHA])
    zb = pb[:, MAG] * np.exp(1j * pb[:, PHA])
    z = _lin(za, zb, t)
    out[:, MAG] = np.abs(z)
    out[:, PHA] = np.angle(z)
    return out

def morph_phase(pa, pb, t):
    """PHASE TRANSPORT (reference): magnitude interpolates, phase rotates along
    the shortest arc -> magnitude never cancels. REPLACE with your operator."""
    out = _geom(pa, pb, t)
    out[:, MAG] = _lin(pa[:, MAG], pb[:, MAG], t)
    d = (pb[:, PHA] - pa[:, PHA] + np.pi) % (2 * np.pi) - np.pi   # shortest arc
    out[:, PHA] = pa[:, PHA] + t * d
    return out

def scramble_target(pb, seed=1):
    """Negative control: destroy the coherent phase relationships of the target."""
    rng = np.random.default_rng(seed)
    pb2 = pb.copy()
    pb2[:, PHA] = rng.uniform(-np.pi, np.pi, size=pb.shape[0])
    return pb2

# ---------------- METRICS --------------------------------------------------
def _lap_var(a):
    k = (a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2] - 4 * a[1:-1, 1:-1])
    return k.var()

def _radial_power(a):
    F = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
    H, W = a.shape
    y, x = np.mgrid[0:H, 0:W]
    r = np.hypot(x - W / 2, y - H / 2).astype(int)
    tbin = np.bincount(r.ravel(), F.ravel())
    nbin = np.bincount(r.ravel())
    return tbin / (nbin + 1e-9)

def metrics(img, imgA, imgB, specA, specB):
    rms = img.std()                                   # contrast; ->0 at the fire
    sharp = _lap_var(img)                             # blur detector
    dA = float(((img - imgA) ** 2).mean())
    dB = float(((img - imgB) ** 2).mean())
    sp = _radial_power(img)
    ref = 0.5 * (specA + specB)
    n = min(len(sp), len(ref))
    spec_dev = float(np.abs(sp[:n] - ref[:n]).sum() / (np.abs(ref[:n]).sum() + 1e-9))
    return rms, sharp, dA, dB, spec_dev

# ---------------- RUN ------------------------------------------------------
def run(args):
    dec = load_decoder(args.model)
    rng = np.random.default_rng(args.seed)
    # two DISTINCT latents -> two distinct faces
    zA = rng.standard_normal(128)
    zB = rng.standard_normal(128)
    pA, pB = decode(dec, zA), decode(dec, zB)
    imgA, imgB = render(pA), render(pB)
    specA, specB = _radial_power(imgA), _radial_power(imgB)
    endpoint_contrast = 0.5 * (imgA.std() + imgB.std())
    endpoint_sharp = 0.5 * (_lap_var(imgA) + _lap_var(imgB))
    ab_gap = float(((imgA - imgB) ** 2).mean())       # A-B distance in image space

    conds = {"lerp": lambda t: morph_lerp(pA, pB, t),
             "phase": lambda t: morph_phase(pA, pB, t),
             "scramble": lambda t: morph_phase(pA, scramble_target(pB, args.seed), t)}

    T = args.frames
    rows, series = [], {c: {"rms": [], "sharp": [], "dB": []} for c in conds}
    for c, fn in conds.items():
        for f in range(T):
            t = f / (T - 1)
            img = render(fn(t))
            rms, sharp, dA, dB, sp = metrics(img, imgA, imgB, specA, specB)
            rows.append([f, round(t, 4), c, rms, sharp, dA, dB, sp])
            series[c]["rms"].append(rms)
            series[c]["sharp"].append(sharp)
            series[c]["dB"].append(dB)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t", "cond", "rms_contrast", "sharp", "d_A", "d_B", "spec_dev"])
        w.writerows(rows)

    # ---------- SCORE THE REGISTERED PREDICTIONS ----------
    def tag(ok): return "[V]" if ok else "[K]"
    lerp_rms_min = min(series["lerp"]["rms"])
    ph_rms_min = min(series["phase"]["rms"])
    ph_sharp_min = min(series["phase"]["sharp"])
    ph_dB_end = series["phase"]["dB"][-1]
    scr_rms_min = min(series["scramble"]["rms"])

    P1 = lerp_rms_min < 0.15 * endpoint_contrast
    P2 = ph_rms_min > 0.60 * endpoint_contrast
    P3 = ph_sharp_min > 0.50 * endpoint_sharp
    P4 = ph_dB_end < 0.20 * ab_gap
    P5 = scr_rms_min < 0.30 * endpoint_contrast

    print(f"\nendpoint_contrast={endpoint_contrast:.4f}  endpoint_sharp={endpoint_sharp:.4f}  A-B gap={ab_gap:.4f}")
    print("-" * 66)
    print(f"P1 baseline fire   {tag(P1)}  lerp min contrast {lerp_rms_min:.4f}  (< {0.15*endpoint_contrast:.4f})")
    print(f"P2 no dead zone    {tag(P2)}  phase min contrast {ph_rms_min:.4f}  (> {0.60*endpoint_contrast:.4f})")
    print(f"P3 stays sharp     {tag(P3)}  phase min sharp {ph_sharp_min:.4f}  (> {0.50*endpoint_sharp:.4f})")
    print(f"P4 ARRIVES AT B    {tag(P4)}  phase d_B(t=1) {ph_dB_end:.4f}  (< {0.20*ab_gap:.4f})   << AVATAR GATE")
    print(f"P5 control breaks  {tag(P5)}  scramble min contrast {scr_rms_min:.4f}  (< {0.30*endpoint_contrast:.4f})")
    print("-" * 66)
    viable = P2 and P3 and P4 and P5
    if viable:
        print("VERDICT [V]: TINY AVATAR VIABLE — transport is coherent, aimable, and the control confirms the mechanism.")
    elif P2 and P3 and not P4:
        print("VERDICT [~]: coherent but NOT AIMABLE — energy/structure preserved, fails to arrive at B. Fix target-matching, not phase.")
    elif (P2 or P3) and not P5:
        print("VERDICT [~]: MECHANISM UNCONFIRMED — scramble control did not break, so 'holding amplitude' may be trivial. Investigate before trusting.")
    else:
        print("VERDICT [K]: phase transport did not clear the gate on this pair. Read the failing lines above.")
    print(f"\nledger -> {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to your decoder (omit for --demo synthetic)")
    ap.add_argument("--demo", action="store_true", help="run with the built-in synthetic model")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="avatar_gate_ledger.csv")
    run(ap.parse_args())
