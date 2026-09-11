#!/usr/bin/env python3
"""
gram_direct.py -- gram_spectrum.py, after reading _chunk().

WHAT THIS DELETES AND WHY
gram_spectrum.py treated the renderer as a black box and probed it: detect which
params columns are amplitudes, remove one packet at a time, invert the output
sigmoid, measure a float32 precision floor, gate on additivity.  ~1300 lines.
Every one of those layers existed to DISCOVER that the pre-activation field is
linear in the coefficients.  splat_trainer5._chunk states it:

    env = exp(-(dx*dx + dy*dy) / (2*s*s))
    ec  = env * cos(2*pi*f*xr)
    es  = env * sin(2*pi*f*xr)
    chans[c] = (a[:,:,c]*ec).sum(1) - (b[:,:,c]*es).sum(1)

ec and es do not depend on coeff.  They are the basis, built from px, py, sigma,
theta, freq alone.  So R is five lines of arithmetic off activate(), in float64,
in one pass.  No amplitude detection, no link inversion, no precision floor, no
A0 -- linearity is structural and visible, not something to measure.

ONE CHECK REPLACES ALL OF IT: sigmoid(R @ c) must reproduce ren(raw) to float32.
If the reimplementation is right, that is exact; if it is wrong, nothing else
runs.  That is the whole verification budget.

THREE THINGS READING IT CORRECTED
  1  THE FRAME IS 2N, NOT N.  Each packet contributes a QUADRATURE PAIR
     (ec, es), so for 409 packets the frame has 818 functions and the Gram is
     818x818.  Analysing a 409x409 Gram measures a different object.
  2  THE CHANNELS SHARE THE FRAME.  ec/es are channel-independent; only (a,b)
     differ.  So the full 2454-dim coefficient Gram (409 x 6) is block diagonal
     with THREE IDENTICAL 818x818 blocks and every eigenvalue is exactly 3-fold
     degenerate.  The previous file's claim that "the three colour channels are
     different bases, not rescalings" was WRONG, and its --channel flag was
     measuring a less fundamental operator.
  3  THE ENVELOPE IS ISOTROPIC.  theta enters only the carrier (xr), never the
     Gaussian.  So orientation rotates the fringes inside a circular window; the
     atoms have one width, not two.

AND THE THING THAT IS ACTUALLY WORTH MEASURING
    px = sigmoid(anchor_logit[:,0][None] + raw[..., 0])
anchor_logit is a LEARNED PER-PACKET PARAMETER, (N,2), in the state dict.  The
decoder supplies only a perturbation on top of it, through a sigmoid.  So
    dpx/draw = sigmoid'(anchor + raw) = px(1-px)
and wherever the learned anchor is large, that derivative collapses and the
packet is STRUCTURALLY PINNED however much the latent moves.

This bears directly on the rank-2 arc.  The standing diagnosis was a
credit-assignment split: "nothing forbids richer positions -- 409x2 free numbers
per frame -- training simply found it cheaper to explain motion with appearance."
If |anchor_logit| is typically large, that is wrong, and the correct statement is
that the position channel is throttled by a saturated sigmoid before training
ever chooses anything.  Different claim, different fix.

    python3 gram_direct.py --anchors room.pt          # torch.load + histogram
    python3 gram_direct.py --model room.pt
    python3 gram_direct.py --selftest
"""

import argparse
import sys

import numpy as np

GOE_R, POISSON_R = 0.5307, 0.3863


# ------------------------------------------------------------------ basis ---

def atoms_from_activate(px, py, sigma, theta, freq, size, grid=(0.0, 1.0)):
    """Reproduce _chunk's ec/es for every packet, vectorised, float64.

    grid must match the renderer's GX/GY span.  px,py come out of a sigmoid, so
    the default is [0,1]; --span overrides it and the verification gate will
    fail loudly if it is wrong, which is the point of having the gate."""
    g = np.linspace(grid[0], grid[1], size)
    X, Y = np.meshgrid(g, g)
    dx = X[None] - px[:, None, None]
    dy = Y[None] - py[:, None, None]
    xr = dx * np.cos(theta)[:, None, None] + dy * np.sin(theta)[:, None, None]
    env = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma[:, None, None] ** 2))
    ph = 2.0 * np.pi * freq[:, None, None] * xr
    return env * np.cos(ph), env * np.sin(ph)


def build_R(px, py, sigma, theta, freq, size, grid=(0.0, 1.0)):
    """R = [ec_1..ec_N, es_1..es_N], shape (size^2, 2N).

    The 2N frame is the honest operator: the loop iterates on COEFFICIENTS, and
    a packet's two coefficients multiply two different functions."""
    ec, es = atoms_from_activate(px, py, sigma, theta, freq, size, grid)
    n = len(px)
    R = np.empty((size * size, 2 * n))
    R[:, :n] = ec.reshape(n, -1).T
    R[:, n:] = es.reshape(n, -1).T
    return R


# ------------------------------------------------------------ statistics ---

def theta_sigma(G):
    n = G.shape[0]
    th = float(np.trace(G)) / n
    dev = G - th * np.eye(n)
    return th, float(np.linalg.norm(dev, 'fro')) / max(abs(th) * np.sqrt(n), 1e-30)


def norm_gaps(lam, w=15):
    """Locally unfolded gaps, geometric local mean (log domain).

    Arithmetic windowing is dominated by its largest term on an octave-graded
    spectrum and read 0.905 instead of 1.000 on a perfectly dyadic one."""
    lam = np.sort(np.asarray(lam, float))
    s = np.diff(lam)          # gaps are positive after sorting; do NOT filter
                              # lam itself -- that silently threw away half of
                              # any signed spectrum (e.g. a GOE test matrix)
    s = s[s > 1e-12 * max(np.median(s), 1e-30)]
    if len(s) < 3 * w:
        return None
    ker = np.ones(w) / w
    ls = np.log(s)
    loc = np.convolve(ls, ker, mode='same')
    edge = np.convolve(np.ones_like(ls), ker, mode='same')
    out = np.exp(ls - loc / np.maximum(edge, 1e-30))
    return out[w:-w] if len(out) > 3 * w else out


def r_sigma(m):
    """Sampling spread of the r estimator on m levels, measured not guessed.

    24 GOE draws each at m = 300/600/1200/1636 gave std 0.0151/0.0112/0.0082/
    0.0073, i.e. sigma ~ 0.28/sqrt(m).  This matters: a single GOE draw at m=600
    landed on 0.4999 while another gave 0.5464, so ANY r value read without an
    error bar is uninterpretable -- and the previous file printed exactly that.
    GOE 0.5307 and Poisson 0.3863 are 0.145 apart, which is ~15 sigma at m=818,
    so the classes are cleanly separable; differences of 0.02 are not."""
    return 0.28 / np.sqrt(max(m, 1))


def r_stat(s):
    """<min(r,1/r)>.  Poisson 0.386, GOE 0.531, GUE 0.603, picket fence -> 1.

    RAW gaps are NOT the statistic to use here: on a geometrically graded
    deterministic spectrum the raw value is exactly 1/ratio, which is 0.500 for a
    dyadic frame -- a hair from GOE.  An octave-graded Gabor Gram is that shape,
    so the raw number impersonates GOE out of pure determinism."""
    if s is None:
        return np.nan
    s = np.asarray(s, float)
    s = s[s > 0]
    if len(s) < 20:
        return np.nan
    r = s[1:] / s[:-1]
    return float(np.mean(np.minimum(r, 1.0 / r)))


def loop_spectrum(G, leak, inject):
    """K = 1/4 D^-1 G, D = diag(G).  Similar to D^-1/2 G D^-1/2, symmetric PSD,
    so the spectrum is real and >= 0 and rho / i_crit are closed forms."""
    d = np.diag(G).copy()
    d[d <= 0] = 1e-30
    s = 1.0 / np.sqrt(d)
    ev = np.linalg.eigvalsh(0.25 * (s[:, None] * G * s[None, :]))
    ev = np.clip(ev, 0.0, None)
    lm = float(ev[-1])
    return ev, lm, leak + inject * lm, (1.0 - leak) / lm if lm > 0 else np.inf


def report(R, leak, inject, label):
    G = R.T @ R
    ev, lm, rho, ic = loop_spectrum(G, leak, inject)
    th, ratio = theta_sigma(G)
    lamG = np.clip(np.linalg.eigvalsh(G), 0.0, None)
    nz = lamG[lamG > lamG.max() * 1e-12]
    print(f"\n{label}    frame {G.shape[0]} functions ({G.shape[0] // 2} packets "
          f"x 2 quadrature)")
    print(f"  A1 theta {th:.5g}   |sigma|/|theta| {ratio:.3f}"
          f"   {'shear' if ratio > 1 else 'breathing'}-dominated")
    m = len(ev)
    sg = r_sigma(m)
    rv = r_stat(norm_gaps(ev))
    print(f"  A3 r {rv:.4f} +- {sg:.4f}   [loop spectrum, unfolded, {m} levels]")
    print(f"     Poisson .3863 | GOE .5307 | picket -> 1 ;"
          f" {abs(rv - GOE_R) / sg:.1f} sigma from GOE,"
          f" {abs(rv - POISSON_R) / sg:.1f} from Poisson")
    print(f"  A6 lam_max {lm:.6g}   rho {rho:.6f}   i_crit {ic:.6f}"
          f"   {'CONTRACTING' if rho < 1 else 'ACTIVE'}")
    print(f"     eff rank {float(lamG.sum() ** 2 / np.sum(lamG ** 2)):.3f}"
          f"   cond {float(nz.max() / nz.min()):.3e}   nonzero {len(nz)}"
          f"/{len(lamG)}")
    return dict(r=r_stat(norm_gaps(ev)), ratio=ratio, lam=lm, rho=rho, ic=ic)


# ------------------------------------------------------------------ model ---

def anchors(path):
    """torch.load and a histogram.  No model construction, no rendering.

    dpx/draw = px(1-px) <= 0.25, and it collapses as |anchor| grows.  If the
    typical |anchor_logit| is large, the position channel is throttled before
    training chooses anything, and the credit-assignment story is the wrong one."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('sd', ck) if isinstance(ck, dict) else ck
    key = next((k for k in sd if k.endswith('anchor_logit')), None)
    if key is None:
        sys.exit(f"no anchor_logit in {path}; keys like: "
                 f"{[k for k in list(sd)[:8]]}")
    a = sd[key].detach().cpu().numpy()
    px = 1.0 / (1.0 + np.exp(-a))
    gain = px * (1.0 - px)
    print(f"\nANCHOR SATURATION  {key}  {a.shape}")
    print(f"  |anchor_logit| percentiles "
          + "  ".join(f"{q}%={np.percentile(np.abs(a), q):.2f}"
                      for q in (5, 25, 50, 75, 95)))
    print(f"  dpx/draw = px(1-px), max possible 0.25")
    print(f"    percentiles "
          + "  ".join(f"{q}%={np.percentile(gain, q):.4f}"
                      for q in (5, 25, 50, 75, 95)))
    frac = float((gain < 0.025).mean())
    print(f"  fraction with gain < 0.025 (i.e. <10% of maximum): {frac:.1%}")
    print(f"  anchor positions span px in [{px.min():.3f}, {px.max():.3f}]")
    print("\n  READ: a large throttled fraction means positions are pinned by\n"
          "  ARCHITECTURE, not by training preferring appearance.  That is a\n"
          "  different claim from the standing rank-2 diagnosis and it has a\n"
          "  different fix (re-centre the anchors / scale the perturbation),\n"
          "  so it is worth settling before any further loss-term work.")
    return a


def load_model_R(path, zscale=0.0, seed=0, span=(0.0, 1.0), size=None,
                 return_params=False):
    """activate() -> five lines -> R.  Verified against the renderer's output."""
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck if isinstance(ck, dict) else {}
    name = meta.get('trainer', 'splat_trainer5')
    mod = __import__(name)
    model = mod.load_splatvae(path)
    model = model[0] if isinstance(model, (tuple, list)) else model
    model.eval()
    sd = meta.get('sd', {})
    nz = int(sd['enc.fc_mu.bias'].shape[0]) if 'enc.fc_mu.bias' in sd else 128
    S = size or int(meta.get('image_size', 192))
    g = torch.Generator().manual_seed(seed)
    z = (torch.randn(1, nz, generator=g) * zscale if zscale > 0
         else torch.zeros(1, nz))
    print(f"  {name}: {S}px, latent {nz}, |z| = {float(z.norm()):.4f}")

    with torch.no_grad():
        raw = model.dec(z)
        raw = raw[0] if isinstance(raw, (tuple, list)) else raw
        vals = model.ren.activate(raw.float())
        px, py, sigma, theta, freq, coeff = [v.detach().cpu().numpy()
                                             for v in vals]
        ref = model.ren(raw).detach().cpu().numpy()

    px, py = px[0].astype(float), py[0].astype(float)
    sigma, theta, freq = (sigma[0].astype(float), theta[0].astype(float),
                          freq[0].astype(float))
    n = len(px)
    R = build_R(px, py, sigma, theta, freq, S, span)

    # THE ONLY GATE.  a*ec - b*es summed over packets, then sigmoid, must equal
    # the renderer's own output.  Exact if the reimplementation is right.
    c0 = coeff[0]                                   # (N, 3, 2)
    errs = []
    for ch in range(c0.shape[1]):
        vec = np.concatenate([c0[:, ch, 0], -c0[:, ch, 1]]).astype(float)
        u = R @ vec
        y = 1.0 / (1.0 + np.exp(-u))
        errs.append(float(np.abs(y - ref[0, ch].ravel()).max()))
    err = max(errs)
    ok = err < 2e-5
    print(f"  V1 [{'V' if ok else 'K'}]  sigmoid(R @ coeff) reproduces "
          f"ren(raw): max err {err:.3e}  (per channel "
          f"{', '.join(f'{e:.1e}' for e in errs)})")
    if not ok:
        sys.exit("V1 [K] -- the reimplementation does not match the renderer, so\n"
                 "  R is not its synthesis operator and nothing else may be read.\n"
                 "  Most likely the GX/GY span is not [0,1]: try --span -1 1.\n"
                 "  Printing nothing rather than a plausible wrong spectrum.")
    print(f"  packets {n}  ->  frame {2 * n} functions")
    if return_params:
        return R, (px, py, sigma, theta, freq), S
    return R


def shuffled(px, py, sigma, theta, freq, seed=0):
    """MATCHED control: same 409 packets, same (sigma, freq) attached to each,
    same canvas, same quadrature construction -- only the learned ARRANGEMENT is
    destroyed.  Positions are permuted as (x,y) pairs so the spatial layout is
    scrambled without changing the marginal distribution of positions; theta gets
    its own permutation.

    THIS IS THE COMPARISON A3 NEEDS AND DID NOT HAVE.  The earlier control was a
    fresh constant-Q draw at 256 atoms on a 96px canvas with ONE function per
    atom, so it differed from the trained frame in level count, resolution and
    quadrature structure all at once -- three confounds on a single number."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(px))
    q = rng.permutation(len(px))
    return px[p], py[p], sigma, theta[q], freq


# --------------------------------------------------------------- selftest ---

def selftest():
    ok = True

    def chk(name, cond, d=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {d}")

    print("SELFTEST")
    rng = np.random.default_rng(0)

    # V1 on a faithful numpy replica of _chunk: the reimplementation must be
    # exact, and must FAIL on a wrong grid span (the one real risk).
    N, S = 60, 40
    px, py = rng.uniform(.15, .85, N), rng.uniform(.15, .85, N)
    sg = rng.uniform(.04, .25, N)
    th = rng.normal(0, 1.5, N)
    fr = rng.uniform(1, 14, N)
    a, b = rng.normal(0, .5, N), rng.normal(0, .5, N)

    def chunk_replica(span):
        g = np.linspace(span[0], span[1], S)
        X, Y = np.meshgrid(g, g)
        out = np.zeros((S, S))
        for k in range(N):
            dx, dy = X - px[k], Y - py[k]
            xr = dx * np.cos(th[k]) + dy * np.sin(th[k])
            env = np.exp(-(dx * dx + dy * dy) / (2 * sg[k] ** 2))
            out += a[k] * env * np.cos(2 * np.pi * fr[k] * xr) \
                - b[k] * env * np.sin(2 * np.pi * fr[k] * xr)
        return out.ravel()

    R = build_R(px, py, sg, th, fr, S, (0.0, 1.0))
    vec = np.concatenate([a, -b])
    chk("V1 R @ [a,-b] reproduces the _chunk replica exactly",
        np.allclose(R @ vec, chunk_replica((0.0, 1.0)), atol=1e-12),
        f"max err {np.abs(R @ vec - chunk_replica((0., 1.))).max():.2e}")
    chk("V1 and DISAGREES on the wrong grid span (so the gate can fail)",
        not np.allclose(R @ vec, chunk_replica((-1.0, 1.0)), atol=1e-6))

    chk("frame is 2N, not N", R.shape[1] == 2 * N, f"{R.shape}")

    # the three channels share ec/es, so a 3-channel coefficient Gram is
    # exactly 3-fold degenerate -- the correction this file exists for
    G1 = R.T @ R
    G3 = np.kron(np.eye(3), G1)
    e3 = np.linalg.eigvalsh(G3)
    gaps = np.diff(np.sort(e3))
    chk("3-channel coefficient Gram is exactly 3-fold degenerate",
        (gaps < 1e-9 * max(e3.max(), 1e-30)).sum() >= 2 * len(e3) // 3 - 2,
        f"{int((gaps < 1e-9 * e3.max()).sum())} of {len(gaps)} gaps are zero")

    # statistics carried over, with their own two-sided checks
    # m=1200: at m=600 the finite-size mean is 0.520, low enough that a +-0.03
    # gate around 0.5307 fails on ordinary draws.  Measured spread at m=1200 is
    # 0.008, so 0.03 is ~3.7 sigma.  Each value is computed ONCE -- the previous
    # version drew a fresh sample for the printout, so the number shown was not
    # the number tested.
    A = rng.normal(size=(1200, 1200))
    goe = np.linalg.eigvalsh((A + A.T) / np.sqrt(2))
    r_goe = r_stat(norm_gaps(goe))
    chk("GOE reads 0.531 (m=1200, measured sigma 0.008)",
        abs(r_goe - GOE_R) < 0.03, f"{r_goe:.4f}")
    r_poi = r_stat(norm_gaps(np.sort(rng.uniform(0, 1, 1200))))
    chk("Poisson reads 0.386", abs(r_poi - POISSON_R) < 0.03, f"{r_poi:.4f}")
    chk("the two classes are separated by many sigma, so the statistic can "
        "actually decide", abs(r_goe - r_poi) > 8 * r_sigma(1200),
        f"{abs(r_goe - r_poi) / r_sigma(1200):.1f} sigma")

    # the sigma formula itself, against measured spread
    spread = np.std([r_stat(norm_gaps(np.linalg.eigvalsh(
        (lambda M: (M + M.T) / np.sqrt(2))(rng.normal(size=(400, 400))))))
        for _ in range(8)])
    chk("r_sigma(m) predicts the measured spread within 2x",
        0.5 * r_sigma(400) < spread < 2.0 * r_sigma(400),
        f"measured {spread:.4f} vs predicted {r_sigma(400):.4f}")
    geo = np.sort(2.0 ** -np.arange(1, 301, dtype=float))
    chk("dyadic spectrum: RAW r impersonates GOE at exactly 0.500",
        abs(r_stat(np.diff(geo)) - 0.5) < 1e-6, f"{r_stat(np.diff(geo)):.4f}")
    chk("and unfolding unmasks it as a picket fence",
        r_stat(norm_gaps(geo)) > 0.97, f"{r_stat(norm_gaps(geo)):.4f}")

    _, lm, rho, ic = loop_spectrum(G1, 0.95, 0.03)
    chk("i_crit puts rho exactly at 1",
        abs(loop_spectrum(G1, 0.95, ic)[2] - 1.0) < 1e-9)
    chk("loop spectrum is PSD (no autonomous rotation)",
        loop_spectrum(G1, .95, .03)[0].min() >= -1e-9)
    sp = shuffled(px, py, sg, th, fr, seed=1)
    chk("matched control preserves the (sigma,freq) marginals exactly",
        np.allclose(np.sort(sp[2]), np.sort(sg)) and
        np.allclose(np.sort(sp[4]), np.sort(fr)))
    Rs = build_R(*sp, S, (0.0, 1.0))
    chk("and changes the Gram", not np.allclose(Rs.T @ Rs, R.T @ R))

    _, ri = theta_sigma(np.eye(50))
    chk("identity is pure breathing", ri < 1e-12)
    _, rg = theta_sigma(np.diag(2.0 ** np.arange(50)))
    chk("geometric grading is shear-dominated", rg > 1.0, f"{rg:.3f}")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--anchors', type=str, default=None)
    ap.add_argument('--zscale', type=float, default=0.0)
    ap.add_argument('--span', type=float, nargs=2, default=(0.0, 1.0))
    ap.add_argument('--size', type=int, default=None)
    ap.add_argument('--leak', type=float, default=0.95)
    ap.add_argument('--inject', type=float, default=0.03)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--control', action='store_true',
                    help='matched control: same packets, arrangement destroyed')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.anchors:
        anchors(a.anchors)
        return
    if a.model:
        out = load_model_R(a.model, a.zscale, a.seed, tuple(a.span), a.size,
                           return_params=a.control)
        if a.control:
            R, prm, S = out
            tr = report(R, a.leak, a.inject, f"TRAINED  {a.model}")
            ct = report(build_R(*shuffled(*prm, seed=a.seed), S, tuple(a.span)),
                        a.leak, a.inject,
                        "CONTROL  (same packets, arrangement destroyed)")
            sg = r_sigma(len(prm[0]) * 2)
            d = tr['r'] - ct['r']
            print(f"\n  S1  r trained {tr['r']:.4f} vs control {ct['r']:.4f}"
                  f"   d {d:+.4f} = {abs(d) / (sg * np.sqrt(2)):.1f} sigma")
            print(f"      {'MORE delocalised than the control -- training MIXED '
                          'the octave bands' if d > 0 else 'MORE localised -- '
                          'training DECOUPLED the bands' if d < 0 else 'no change'}"
                  f"  (significant above ~2 sigma)")
            print(f"  S2  shear {tr['ratio']:.3f} vs {ct['ratio']:.3f}"
                  f"   |   lam_max {tr['lam']:.4f} vs {ct['lam']:.4f}"
                  f"   |   i_crit {tr['ic']:.5f} vs {ct['ic']:.5f}")
        else:
            report(out, a.leak, a.inject, f"TRAINED  {a.model}")
        return
    ap.print_help()


if __name__ == '__main__':
    main()