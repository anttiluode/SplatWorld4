#!/usr/bin/env python3
"""
flatworld.py -- 2D substrate, 3D appearance.  How many numbers does the fake cost?

THE TRAP IN THE SPEC, CAUGHT BEFORE BUILDING
The proposed benchmark was: "can a moving camera orbit a flat packet scene and
produce consistent parallax, occlusion and apparent depth without any stored
world-space z?"  As stated that cannot fail, and passing it would mean nothing.
If you write a coefficient function f_i(q, theta, L) that produces correct
parallax and occlusion, you have STORED z -- inside the function instead of in
a variable.  Parallax is defined by depth (du ~ f dX / Z); to move packet i by
the right amount the function must know packet i's depth.  Whether that number
sits in a float named z_i or in f_i's parameters is bookkeeping, not ontology.
It is the same smuggle as the vector delay line carrying an angle theta_n that
lives in a space already assumed.

SO THE QUESTION IS NOT WHETHER, IT IS HOW MANY.
Flat geometry + view-dependent appearance can obviously fake anything if the
appearance is allowed K free numbers per frame.  The measurement that means
something is the SCALING:

    appearance DOF required ~ O(K)   -> one number per packet, which behaves
                                        exactly like depth.  "No stored z" is
                                        an accounting fiction and the flat
                                        claim is empty.
    appearance DOF required ~ O(1)   -> the whole scene's viewpoint dependence
                                        lives in a handful of shared modes.
                                        That is a real compression claim and
                                        the flat substrate is doing work.

PRE-REGISTERED PREDICTION, made before running (this is the falsifier)
  P1  rigid-body position geometry reads rank 3.  This is the POSITIVE CONTROL
      sfm_rank never had -- it reported rank 2 for model2 but was never shown a
      scene that genuinely IS 3D, so "rank 2 means no depth" rested on an
      uncalibrated instrument.  If P1 fails, every earlier rank verdict is void.
  P2  unclamped Lambertian shading reads rank 3.  Known result (Shashua 1997):
      n_i . l_f is bilinear, so the matrix is rank 3 exactly.
  P3  clamping at zero (attached shadows) raises it to about 9.  Known result
      (Basri & Jacobs 2003): 9 spherical harmonics capture ~98% of Lambertian
      reflectance under arbitrary illumination.
  P4  OCCLUSION IS THE ONE THAT BREAKS IT.  Binary visibility is a viewpoint-
      dependent total ORDER on the packets, which is what depth IS.  Predicted
      to need DOF growing with K, i.e. z by another name.

P2 and P3 are checks against published theory: if this code cannot reproduce 3
and 9 it is wrong, and no result it produces about P4 should be believed.

The 3D scene here is a TEACHER, not a claim about substrate.  It generates the
target image sequence.  The FlatWorld model that must reproduce it stores only
2D positions, a per-frame 2x2 affine, and rank-r coefficients.

    python3 flatworld.py --selftest
    python3 flatworld.py --measure
    python3 flatworld.py --scaling          (the O(K) vs O(1) question)
    python3 flatworld.py --render out.png
"""

import argparse
import sys

import numpy as np


# ------------------------------------------------------------------ scene ---

def make_scene(K=180, seed=0):
    """Points on three spheres plus a ground plane.  Returns positions and
    outward normals -- normals are what shading and self-occlusion need."""
    rng = np.random.default_rng(seed)
    P, Nrm = [], []
    for c, r, n in (((-1.1, 0.2, 0.0), 0.55, K // 3),
                    ((0.9, -0.1, 0.6), 0.45, K // 4),
                    ((0.3, 0.7, -0.7), 0.35, K // 5)):
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        P.append(np.asarray(c) + r * v)
        Nrm.append(v)
    n_g = K - sum(len(p) for p in P)
    g = np.column_stack([rng.uniform(-2, 2, n_g),
                         np.full(n_g, -0.9),
                         rng.uniform(-2, 2, n_g)])
    P.append(g)
    Nrm.append(np.tile([0.0, 1.0, 0.0], (n_g, 1)))
    return np.vstack(P), np.vstack(Nrm)


def orbit(F=36):
    """Camera azimuths and the orthographic projection rows for each."""
    th = np.linspace(0, 2 * np.pi, F, endpoint=False)
    Rs = []
    for t in th:
        ct, st = np.cos(t), np.sin(t)
        R = np.array([[ct, 0, -st], [0, 1, 0], [st, 0, ct]])
        Rs.append(R)
    return th, np.array(Rs)


def observe(P, Nrm, Rs, headlight=True, clamp=True, occlude=True):
    """Generate what a camera on the orbit actually sees.

    Returns
      W    (2F, K)  projected positions, per-frame centred  -- the geometry
      C    (K, F)   per-packet brightness                   -- the appearance
      V    (K, F)   binary visibility                       -- the occlusion
      D    (K, F)   camera-frame depth (ground truth, never given to the model)
    """
    F, K = len(Rs), len(P)
    W = np.zeros((2 * F, K))
    C = np.zeros((K, F))
    V = np.zeros((K, F))
    D = np.zeros((K, F))
    l_cam = np.array([0.3, 0.4, 1.0])
    l_cam = l_cam / np.linalg.norm(l_cam)
    for f, R in enumerate(Rs):
        Pc = P @ R.T                      # into camera frame
        Nc = Nrm @ R.T
        uv = Pc[:, :2]
        W[2*f:2*f+2, :] = (uv - uv.mean(0)).T
        D[:, f] = Pc[:, 2]
        lam = Nc @ (l_cam if headlight else np.array([0.3, 0.4, 1.0]))
        C[:, f] = np.maximum(lam, 0.0) if clamp else lam
        V[:, f] = (Nc[:, 2] > 0).astype(float) if occlude else 1.0
    return W, C, V, D


# ----------------------------------------------------------------- ranks ---

def spectrum(M):
    """Singular values, energy counts and participation ratio."""
    s = np.linalg.svd(M, compute_uv=False)
    e = s ** 2 / max((s ** 2).sum(), 1e-30)
    n95 = int(np.searchsorted(np.cumsum(e), 0.95) + 1)
    n99 = int(np.searchsorted(np.cumsum(e), 0.99) + 1)
    part = float(np.exp(-(e * np.log(e + 1e-30)).sum()))
    return s, n95, n99, part


def show(name, M, predict=None):
    s, n95, n99, part = spectrum(M)
    p = f"   predicted {predict}" if predict is not None else ""
    print(f"  {name:<34} 95% in {n95:>3}   99% in {n99:>3}   "
          f"participation {part:6.2f}{p}")
    return n95, n99, part


def truncate(M, r):
    """Best rank-r approximation."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


# ------------------------------------------------------------- measurement ---

def measure(K=180, F=36, seed=0, quiet=False):
    P, Nrm = make_scene(K, seed)
    th, Rs = orbit(F)
    W, C, V, D = observe(P, Nrm, Rs)
    _, C_lin, _, _ = observe(P, Nrm, Rs, clamp=False)
    A = C * V                                        # what is actually seen

    if not quiet:
        print(f"scene: {K} packets, {F} camera azimuths, orthographic orbit\n")
        print("GEOMETRY -- the positive control the earlier rank tests lacked")
    n95_g, _, part_g = (show("position matrix W (2F x K)", W, "rank 3")
                        if not quiet else spectrum(W)[1:])
    if not quiet:
        print("\nAPPEARANCE -- checked against published theory")
        show("Lambertian, no clamp  n.l", C_lin, "rank 3   (Shashua 1997)")
        show("Lambertian, clamped   max(0,n.l)", C, "~9  (Basri-Jacobs 2003)")
        show("visibility V (binary occlusion)", V, "P4: large")
        show("what the camera sees  A = C*V", A, "P4: large")
    return dict(W=W, C=C, C_lin=C_lin, V=V, A=A, D=D, P=P, Nrm=Nrm, Rs=Rs)


def flat_error(d, ranks=(1, 2, 3, 6, 9, 16, 32)):
    """Reproduce the seen appearance with rank-r coefficients.

    Geometry is held FLAT throughout: the model gets the rank-2 affine family
    only, never the rank-3 truth.  So the geometric residual below is the price
    of flatness, and the coefficient sweep is the price of faking the rest."""
    W, A, V = d['W'], d['A'], d['V']
    W2 = truncate(W, 2)
    geo = np.linalg.norm(W - W2) / np.linalg.norm(W)
    print(f"\nGEOMETRY HELD FLAT: rank-2 affine leaves "
          f"{100*geo:.1f}% positional residual")
    print("  (a rigid 3D scene needs 3; forcing 2 is exactly the flat constraint)")
    print(f"\n{'rank r':>7} {'appearance err':>15} {'occlusion flips wrong':>23}")
    flips = (np.abs(np.diff(V, axis=1)) > 0.5)
    out = []
    for r in ranks:
        Ar = truncate(A, r)
        err = np.linalg.norm(A - Ar) / np.linalg.norm(A)
        Vr = (Ar > 0.5 * A[A > 0].mean()).astype(float)
        fr = (np.abs(np.diff(Vr, axis=1)) > 0.5)
        wrong = float((fr != flips).sum()) / max(flips.sum(), 1)
        print(f"{r:>7} {100*err:>14.1f}% {100*wrong:>22.1f}%")
        out.append((r, err, wrong))
    return out


def budget(d, target=0.10):
    """THE ARITHMETIC THAT DECIDES IT.

    Rank was the wrong statistic: rank(K x F) <= F, so with a fixed number of
    camera frames it can never grow with packet count no matter what the
    appearance is doing.  The quantity that matters is PER-PACKET STORAGE.

    A rank-r appearance model gives every packet r loading coefficients.  So:

        FlatWorld   2 (x,y) + r (loadings)          per packet
        depth model 3 (x,y,z) + 3 (normal)          per packet

    The flat representation is cheaper only if r < 4.  That is the whole
    question, and it is decided by one measured number."""
    A, C = d['A'], d['C']
    out = {}
    for nm, M in (("shading only (no occlusion)", C), ("what is seen (C*V)", A)):
        s = np.linalg.svd(M, compute_uv=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        r = int(np.searchsorted(e, 1 - target ** 2) + 1)
        out[nm] = r
        verdict = ("FLAT WINS" if 2 + r < 6 else
                   "TIE" if 2 + r == 6 else "DEPTH WINS")
        print(f"  {nm:<30} r = {r:>3}   flat costs {2+r:>3}/packet "
              f"vs 6 for depth   -> {verdict}")
    return out


def scaling(Ks=(60, 120, 240, 480), F=36, target=0.10):
    """Kept for reference; see budget() -- rank is capped by F, not K."""
    print(f"\nSCALING -- rank needed to get seen-appearance error below "
          f"{100*target:.0f}%")
    print(f"{'K':>6} {'rank needed':>13} {'rank/K':>9}   verdict")
    rows = []
    for K in Ks:
        d = measure(K=K, F=F, quiet=True)
        A = d['A']
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        need = int(np.searchsorted(e, 1 - target ** 2) + 1)
        rows.append((K, need))
        print(f"{K:>6} {need:>13} {need/K:>9.3f}")
    ks = np.array([r[0] for r in rows], float)
    ns = np.array([r[1] for r in rows], float)
    slope = float(np.polyfit(np.log(ks), np.log(ns), 1)[0])
    print(f"\n  log-log slope of rank vs K: {slope:+.2f}")
    if slope > 0.6:
        print("  => O(K).  The appearance carries one number per packet, which "
              "is what depth is.")
        print("     'No stored z' is bookkeeping; the flat claim is empty.")
    elif slope < 0.25:
        print("  => O(1).  Viewpoint dependence lives in a few shared modes "
              "regardless of scene size.")
        print("     The flat substrate is doing real work.")
    else:
        print("  => intermediate; neither reading is supported yet.")
    return slope


# ---------------------------------------------------------------- render ---

def render(d, path, size=96, r_list=(3, 9, 32, None)):
    """Contact sheet: same flat scene, increasing appearance budget."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    W, A = d['W'], d['A']
    W2 = truncate(W, 2)
    frames = [0, len(d['Rs']) // 4, len(d['Rs']) // 2]
    fig, ax = plt.subplots(len(r_list), len(frames),
                           figsize=(2.1 * len(frames), 2.1 * len(r_list)))
    g = np.linspace(-2.4, 2.4, size)
    X, Y = np.meshgrid(g, g)
    for i, r in enumerate(r_list):
        Ar = A if r is None else truncate(A, r)
        for j, f in enumerate(frames):
            img = np.zeros((size, size))
            xs, ys = W2[2*f], W2[2*f+1]
            for k in range(A.shape[0]):
                c = max(Ar[k, f], 0.0)
                if c <= 0.01:
                    continue
                img += c * np.exp(-((X - xs[k])**2 + (Y - ys[k])**2) / (2*0.075**2))
            a = ax[i, j]
            a.imshow(img, cmap='magma', vmin=0, vmax=max(img.max(), 1e-6))
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel("full" if r is None else f"rank {r}", fontsize=9)
            if i == 0:
                a.set_title(f"azimuth {360*f//len(d['Rs'])} deg", fontsize=9)
    fig.suptitle("FlatWorld: rank-2 geometry, appearance budget increasing",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


# -------------------------------------------------------------- selftest ---

def selftest():
    ok = True

    def chk(n, c, dtl=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}  {dtl}")

    print("SELFTEST")
    th, Rs = orbit(36)

    # S1 two-sided: rigid 3D reads 3, a PLANAR scene reads 2
    P, Nrm = make_scene(150, 0)
    W, _, _, _ = observe(P, Nrm, Rs)
    _, n99, _ = spectrum(W)[1:]
    chk("S1 rigid 3D scene -> position rank 3", n99 == 3, f"99% in {n99}")
    Pp = P.copy(); Pp[:, 2] = 0.0
    Wp, _, _, _ = observe(Pp, Nrm, Rs)
    _, n99p, _ = spectrum(Wp)[1:]
    chk("S1 planar scene -> position rank 2 (not 3)", n99p <= 2,
        f"99% in {n99p}")

    # S2 against Shashua: unclamped Lambertian is exactly rank 3
    _, C_lin, _, _ = observe(P, Nrm, Rs, clamp=False)
    s = np.linalg.svd(C_lin, compute_uv=False)
    chk("S2 unclamped Lambertian is rank 3 (Shashua)",
        s[3] / s[0] < 1e-10, f"s4/s1 = {s[3]/s[0]:.2e}")

    # S3 clamping must RAISE it -- otherwise the instrument is blind to shadows
    _, C_cl, _, _ = observe(P, Nrm, Rs, clamp=True)
    _, n99c, _ = spectrum(C_cl)[1:]
    chk("S3 clamped Lambertian needs more than 3", n99c > 3, f"99% in {n99c}")

    # S4 truncation error is monotone decreasing in r
    A = C_cl * observe(P, Nrm, Rs)[2]
    errs = [np.linalg.norm(A - truncate(A, r)) for r in (1, 2, 4, 8, 16)]
    chk("S4 rank-r error decreases monotonically",
        all(errs[i] >= errs[i+1] for i in range(len(errs)-1)),
        " ".join(f"{e:.2f}" for e in errs))

    # S5 two-sided on the scaling estimator itself: planted O(K) and O(1)
    rng = np.random.default_rng(0)
    def need(M, t=0.10):
        s = np.linalg.svd(M, compute_uv=False)
        e = np.cumsum(s**2)/(s**2).sum()
        return int(np.searchsorted(e, 1-t**2)+1)
    # F must exceed K or rank is capped by frames and nothing can scale.
    # That cap is exactly what the first version of this test walked into.
    sl_full = np.polyfit(np.log([60., 240.]),
                         np.log([need(rng.standard_normal((k, 600))) for k in (60, 240)]), 1)[0]
    lowr = [(rng.standard_normal((k, 4)) @ rng.standard_normal((4, 600)))
            for k in (60, 240)]
    sl_low = np.polyfit(np.log([60., 240.]),
                        np.log([need(m) for m in lowr]), 1)[0]
    chk("S5 planted full-rank data -> slope near 1", sl_full > 0.6,
        f"slope {sl_full:+.2f}")
    chk("S5 planted rank-4 data -> slope near 0", abs(sl_low) < 0.25,
        f"slope {sl_low:+.2f}")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, default=180)
    p.add_argument('--F', type=int, default=36)
    p.add_argument('--measure', action='store_true')
    p.add_argument('--scaling', action='store_true')
    p.add_argument('--render', type=str, default='')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.scaling:
        scaling(F=a.F); return
    if a.render:
        render(measure(a.K, a.F), a.render); return
    if a.measure:
        d = measure(a.K, a.F)
        flat_error(d)
        print("\nPER-PACKET STORAGE -- the question rank could not answer")
        budget(d)
        return
    p.print_help()


if __name__ == '__main__':
    main()
