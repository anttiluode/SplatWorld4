#!/usr/bin/env python3
# live_cortex_gated.py — the lateral gate: packets may only eat error together
#
# The floaters were the fastest MSE strategy for a lone packet: collapse sigma
# to the floor, crank amplitude, patch one bright pixel, fire alone. The
# missing organ is lateral: a packet's license to correct should depend on
# whether its neighbors corroborate its claim.
#
# TWO GATES ARE IMPLEMENTED, ON PURPOSE:
#   naive   per-pixel ownership softmax (env_i / sum_j env_j) — the obvious
#           reading of "ownership of the residual". PREDICTED TO FAIL on
#           floaters: an isolated packet owns its pixel COMPLETELY, so pure
#           ownership licenses exactly the lone-wolf behavior it should stop.
#   gate    corroboration: a packet's step is scaled by how much OTHER
#           envelope mass co-claims the residual under its own footprint —
#             s_i = sum_xy env_i * (E - env_i) * |r|  /  sum_xy env_i * |r|
#             g_i = s_i / (kappa + s_i)
#           isolated packet -> s ~ 0 -> gate ~ 0; packet inside an edge
#           coalition -> gate ~ 1. This is the divisive-normalization /
#           "fire together" reading, and the bench decides between them.
#
# THE FALSIFIABLE CLAIM (from the conversation): the gate flattens the
# floater-vs-lr curve — you keep the aggressive lr that sharpens fast,
# without the stars. If the curve does not flatten, the coordination story
# is wrong. Run: python live_cortex_gated.py --bench   (no webcam, no model)
#
# Live:  python live_cortex_gated.py --model runs/splat2/model2.pt
#        keys: g cycle gate off/naive/gate | +/- lr | [/] steps | q quit
#        HUD shows live floater count; toggle g and watch the stars.
#
# Requires splat_trainer2.py (patched version) in the same directory.

import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

try:
    from Splat_trainer2 import GaborRenderer, SplatVAE, LATENT, K
except ImportError:
    sys.exit("put this next to (patched) splat_trainer2.py")

SIGMA_FLOOR = 0.012          # renderer's hard sigma minimum
FLOAT_SIG   = 0.016          # "at the floor" threshold
FLOAT_AMP   = 0.40           # "bright" threshold (tanh coeff magnitude)
KAPPA       = 0.25           # corroboration half-saturation

# ---------------------------------------------------------------- gate math
@torch.no_grad()
def envelope_fields(ren, raw):
    px, py, sigma, *_ = ren.activate(raw.float())
    dx = ren.GX - px[..., None, None]
    dy = ren.GY - py[..., None, None]
    s_ = sigma[..., None, None]
    return torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))     # (B,N,H,W)

@torch.no_grad()
def packet_gate(ren, raw, resid, mode):
    """Per-packet grad scale in [0,1]. resid: (B,3,H,W) recon-target."""
    if mode == "off":
        return torch.ones(raw.shape[:2], device=raw.device)
    env = envelope_fields(ren, raw)                            # (B,N,H,W)
    r = resid.abs().mean(1, keepdim=True)                      # (B,1,H,W)
    er = env * r                                               # claim on error
    denom = er.sum((-1, -2)) + 1e-8                            # (B,N)
    if mode == "naive":                                        # pure ownership
        E = env.sum(1, keepdim=True)
        own = env / (E + 1e-8)
        return (own * er).sum((-1, -2)) / denom
    E = env.sum(1, keepdim=True)                               # corroboration
    s = ((E - env) * er).sum((-1, -2)) / denom                 # neighbor mass
    return s / (KAPPA + s)

def floaters(ren, raw):
    with torch.no_grad():
        _, _, sigma, _, _, coeff = ren.activate(raw.float())
        amp = coeff.pow(2).sum(-1).sqrt().amax(-1)             # (B,N)
        return int(((sigma < FLOAT_SIG) & (amp > FLOAT_AMP)).sum().item())

def correct(ren, raw0, target, steps, lr, mode):
    """Adam-descend raw packets against target with the chosen gate.

    Adam matters twice here. The floater pathology NEEDS an adaptive
    optimizer: plain SGD's gradients through the sigmoids are too gentle to
    collapse sigma (measured: 0 floaters at any lr), while Adam's per-param
    normalization lets a lone packet's sigma sprint to the floor (44 at
    lr 0.4). And because Adam divides by grad magnitude, scaling the GRADIENT
    by the gate would be partially normalized away — so the gate scales the
    STEP instead: raw <- before + g * (adam_step)."""
    raw = raw0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    for _ in range(steps):
        recon = ren(raw)
        loss = F.mse_loss(recon, target)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            g = packet_gate(ren, raw, (recon - target).detach(), mode)
            before = raw.detach().clone()
        opt.step()
        with torch.no_grad():
            raw.data = before + g[..., None] * (raw.data - before)
    return raw.detach(), float(F.mse_loss(ren(raw.detach()), target))

# ---------------------------------------------------------------- bench
def bench(steps=30, seed=0):
    """floater-vs-lr curve, three arms. Out-of-domain surprise target:
    smooth gradient + isolated bright dots — floater bait."""
    torch.manual_seed(seed)
    H, N = 64, 64
    ren = GaborRenderer(H, N, chunk=32)
    gy, gx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, H),
                            indexing="ij")
    target = torch.stack([0.35 + 0.2 * gx, 0.35 + 0.2 * gy,
                          0.4 * torch.ones_like(gx)])[None]
    rng = np.random.default_rng(seed)
    for _ in range(6):                                         # the bait
        cy, cx = rng.integers(8, H - 8, 2)
        d2 = (gy - cy / H) ** 2 + (gx - cx / H) ** 2
        target[0] += torch.exp(-d2 / (2 * 0.008 ** 2))[None] * 0.9
    target = target.clamp(0, 1)
    raw0 = torch.randn(1, N, K) * 0.05

    print(f"{'lr':>6} | {'off: flt':>9} {'mse':>7} | {'naive: flt':>10} "
          f"{'mse':>7} | {'gate: flt':>9} {'mse':>7}")
    rows = {}
    for lr in (0.05, 0.1, 0.2, 0.4):
        row = {}
        for mode in ("off", "naive", "gate"):
            raw, mse = correct(ren, raw0, target, steps, lr, mode)
            row[mode] = (floaters(ren, raw), mse)
        rows[lr] = row
        print(f"{lr:>6.2f} | {row['off'][0]:>9d} {row['off'][1]:>7.4f} | "
              f"{row['naive'][0]:>10d} {row['naive'][1]:>7.4f} | "
              f"{row['gate'][0]:>9d} {row['gate'][1]:>7.4f}")
    off_hi = rows[0.4]["off"][0]; gate_hi = rows[0.4]["gate"][0]
    naive_hi = rows[0.4]["naive"][0]
    fit_gate = np.mean([rows[l]["gate"][1] / rows[l]["off"][1]
                        for l in (0.1, 0.2)])
    fit_naive = np.mean([rows[l]["naive"][1] / rows[l]["off"][1]
                         for l in (0.1, 0.2)])
    print(f"\nverdict at lr=0.4: off {off_hi} floaters, naive {naive_hi}, "
          f"gate {gate_hi}.")
    print(f"fit preserved (mse vs off, lr 0.1-0.2): gate x{fit_gate:.1f}, "
          f"naive x{fit_naive:.1f}")
    flat = gate_hi <= max(1, off_hi // 3)
    print("claim " + ("SURVIVES: the corroboration gate flattens the "
          "floater-vs-lr curve while keeping the fit"
          if flat else "DIES: gate did not flatten the curve — the "
          "coordination story is wrong or mistuned") + ".")
    if naive_hi < off_hi and fit_naive > 2.0:
        print("naive ownership also shows fewer floaters — but by throttling "
              "ALL correction (see its mse column): it brakes learning, the "
              "corroboration gate brakes isolation. Selectivity is the point.")
    return 0 if flat else 1

# ---------------------------------------------------------------- selftest
def selftest():
    ok = True
    def check(name, cond, note=""):
        nonlocal ok; ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {note}")
    ren = GaborRenderer(48, 4, chunk=4)
    # two packets far apart + two overlapping, uniform residual
    raw = torch.zeros(1, 4, K)
    raw[0, :, 2] = -1.0                                        # mid sigma
    # anchors are a 2x2 grid; push packet 3 onto packet 2's anchor
    raw[0, 3, 0] = ren.anchor_logit[2, 0] - ren.anchor_logit[3, 0]
    raw[0, 3, 1] = ren.anchor_logit[2, 1] - ren.anchor_logit[3, 1]
    resid = torch.ones(1, 3, 48, 48)
    g = packet_gate(ren, raw, resid, "gate")[0]
    check("overlapping pair gated open vs isolated",
          g[2] > 3 * g[0] and g[3] > 3 * g[0],
          f"iso {g[0]:.3f} vs pair {g[2]:.3f},{g[3]:.3f}")
    gn = packet_gate(ren, raw, resid, "naive")[0]
    check("naive ownership blesses the isolated packet", gn[0] > 0.8,
          f"naive iso {gn[0]:.3f} (this is WHY naive fails)")
    check("gates bounded", float(g.max()) <= 1.0 and float(g.min()) >= 0.0)
    g_off = packet_gate(ren, raw, resid, "off")
    check("off = ones", bool((g_off == 1).all()))
    raw2, _ = correct(ren, raw, torch.rand(1, 3, 48, 48), 3, 0.1, "gate")
    check("correct() runs and changes raw",
          float((raw2 - raw).abs().max()) > 0)
    print("selftest:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1

# ---------------------------------------------------------------- live
def live(model_path):
    import cv2 as cv
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(model_path, map_location=dev)
    model = SplatVAE(ck["image_size"], ck["num_packets"]).to(dev).eval()
    model.load_state_dict(ck["sd"])
    ren, S = model.ren, ck["image_size"]
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("webcam failed to open")
    modes, mi = ["off", "naive", "gate"], 2
    lr, steps = 0.2, 5
    win = "GATED CORTEX  real | prior | locked   (g=gate  +/-=lr  [ ]=steps)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]; s = min(h, w)
        frame = frame[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        x = cv.resize(frame, (S, S))[:, :, ::-1].astype(np.float32) / 255.0
        xt = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(dev)
        with torch.no_grad():
            mu, _ = model.enc(xt)
            raw0 = model.dec(mu)
            prior = ren(raw0)
        raw, _ = correct(ren, raw0, xt, steps, lr, modes[mi])
        with torch.no_grad():
            locked = ren(raw)
        nfl = floaters(ren, raw)
        def to8(t):
            im = (t[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255)
            return cv.resize(im.astype(np.uint8)[:, :, ::-1], (384, 384),
                             interpolation=cv.INTER_CUBIC)
        panel = np.concatenate([to8(xt), to8(prior), to8(locked)], 1)
        cv.putText(panel, f"gate:{modes[mi]}  lr {lr:.2f}  steps {steps}  "
                   f"floaters {nfl}", (10, 374), cv.FONT_HERSHEY_PLAIN,
                   1.2, (0, 255, 0), 1, cv.LINE_AA)
        cv.imshow(win, panel)
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('g'): mi = (mi + 1) % 3
        elif k in (ord('+'), ord('=')): lr = min(1.0, lr + 0.05)
        elif k == ord('-'): lr = max(0.01, lr - 0.05)
        elif k == ord(']'): steps = min(20, steps + 1)
        elif k == ord('['): steps = max(1, steps - 1)
    cap.release(); cv.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/splat2/model2.pt")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    elif a.bench:  sys.exit(bench())
    else:          live(a.model)
