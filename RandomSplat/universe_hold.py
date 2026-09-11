#!/usr/bin/env python3
"""
universe_hold.py -- CAN THE UNIVERSE HOLD ANYTHING?

Before building something to explore, measure whether there is anything to
explore.  A world you can populate needs three properties at once:

    MANY      more than one object
    PERSISTENT the same objects are there when you come back
    RECEPTIVE  a thing you put in stays where you put it

Every sweep in universe_v2 so far has one of these and never two.  This file
measures all three on the same run and reports them together, so a parameter
set can be rejected in 30 seconds instead of being explored for an evening.

WHAT WAS ALREADY MEASURED (his four CSVs + controls, N=40-48, lam0=8)
    lam=0            1 object at 29.5% of the box by step ~1000, then flat
                     for 4000 more.  Track lifetime 41/41 -- perfectly
                     persistent, and it is one thing.
    lam=-2, tau=5    56-72 objects, biggest 3.7% of box.  Coarsening arrested.
                     But 1660 tracks for ~70 objects and median lifetime 1
                     record of 41 -- a boiling foam, nothing persists.
    seeding          lam=0 absorbs an injected blob entirely (nearest object
                     stays 16 cells away and never approaches).  lam=-2 gives
                     a distance that wanders 3.9-10.5, which is the nearest of
                     70 churning objects, not the seed.
    subcritical      a<0 + quintic gives a genuinely stable background
                     (max|phi| holds at 1e-2), but seeds die --
                     BOTH shapes, so spectral mismatch is not the explanation.
                     A Gabor packet at k0 with amp 1.2 decayed after ~600
                     steps at a=-0.1, b=1, c=1, gamma=0.05.

CORRECTION TO MY OWN FIRST EXPLANATION (S5 caught it).  I said a smooth blob
dies because it has no power in the unstable band, quoting exp(-k0^2 sig^2/2)
= 0.06.  That is the AMPLITUDE AT k0, not the power fraction in the band.
Measured: a sigma=3 Gaussian puts 0.145 of its power in |k-k0|/k0 <= 0.30 --
a minority, but not exclusion -- because the band is a SHELL in 3D and shells
have volume.  A Gabor packet puts 0.815 there, 5.6x more.  The shape claim
survives as a ratio and dies as an exclusion, and it is therefore NOT the
whole reason the seeds decayed; damping is doing most of the work.

SO THE OPEN QUESTION IS A SEARCH, NOT A GUESS: does ANY (a,b,c,gamma,lam,tau)
pass U1+U2+U4 together?  --scan runs the grid and prints the table.

TWO THINGS THIS FILE FIXES IN THE EXISTING INSTRUMENT
  1. count_objects() thresholds at 0.5 x the field's OWN 99.5th percentile, so
     it reports 27-67 "objects" in a dead field whose max|phi| is 0.003.  Every
     object count on a decaying run is meaningless.  Here the threshold has an
     absolute floor and a dead field reports zero.  (Selftest S2.)
  2. tracking at 250-step intervals rebuilds IDs every record, because objects
     drift further than the match radius between samples.  Lifetimes measured
     that way are all 1 and mean nothing.  Sampling interval is a parameter
     here and the tracker uses one-to-one greedy matching.  (Selftest S4.)

GATES (pre-registered, two-sided)
  U0  positive control -- the DEFAULT config must coarsen.  <=3 objects AND
      biggest >=20% of box.  If U0 fails, the instrument is broken and nothing
      below is admissible.
  U1  multiplicity   >= 20 objects, median over the final half of the run
  U2  persistence    median track lifetime >= 0.8 x n_records
                     AND n_tracks <= 1.5 x n_objects   (no churn)
  U3  scale held     dominant wavelength within 20% of lam0 throughout
  U4  receptive      inject a Gabor packet at k0 at a chosen point; after
                     hold_steps an object sits within one wavelength of it
                     AND the field's normalised overlap with the injected
                     packet is >= 0.20

    python3 universe_hold.py --selftest
    python3 universe_hold.py --gates                 (default config)
    python3 universe_hold.py --gates --lam -2 --tau 5
    python3 universe_hold.py --scan --out scan.csv
"""

import argparse
import csv
import itertools
import sys
import time

import numpy as np
from scipy.ndimage import label, center_of_mass

from universe_v2 import Universe


# ------------------------------------------------------------- measuring ---

def objects(phi, absthr, minv=8):
    """Connected components above an ABSOLUTE threshold.

    The absolute floor is the whole point.  A percentile threshold rescales
    with the field, so a field decaying to 1e-3 still reports its noise as
    dozens of objects."""
    lab, k = label(phi > absthr, structure=np.ones((3, 3, 3)))
    if k == 0:
        return []
    vols = np.bincount(lab.ravel())
    out = []
    for i in range(1, k + 1):
        if vols[i] >= minv:
            out.append((np.asarray(center_of_mass(lab == i)), int(vols[i])))
    return out


def track(records, radius):
    """One-to-one greedy nearest matching between consecutive records.

    `records` is a list of lists of (centroid, volume).  Returns lifetimes in
    records.  Greedy-with-exclusion matters: assigning several components to
    one previous id silently deletes tracks and drives every lifetime to 1."""
    ids, nxt, life = {}, 0, {}
    for cs in records:
        pairs = sorted(
            (float(np.linalg.norm(c - ids[p][0])), j, p)
            for j, (c, _) in enumerate(cs) for p in ids)
        used_j, used_p, new = set(), set(), {}
        for d, j, p in pairs:
            if d > radius or j in used_j or p in used_p:
                continue
            new[p] = cs[j]
            used_j.add(j)
            used_p.add(p)
            life[p] = life.get(p, 0) + 1
        for j, cv in enumerate(cs):
            if j not in used_j:
                new[nxt] = cv
                life[nxt] = 1
                nxt += 1
        ids = new
    return np.array(sorted(life.values())) if life else np.array([0])


def gabor_packet(N, centres, amp, sigma, k0):
    """A localised wave packet AT the preferred wavenumber.

    A plain Gaussian is the wrong seed and this is why: its spectrum sits at
    k=0, and -(lap+k0^2)^2 damps k=0 by k0^4.  The amplitude a Gaussian of
    width sigma places in the unstable band is exp(-k0^2 sigma^2/2), which for
    sigma=3, lam0=8 is 0.06.  The only shape this operator accepts is an
    envelope times a carrier at k0 -- i.e. a Gabor atom.  That is a spectral
    fact about the equation, not an analogy with the splat work."""
    g = np.arange(N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    b = np.zeros((N, N, N))
    for c in centres:
        r = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
        b += amp * np.exp(-r ** 2 / (2.0 * sigma ** 2)) * np.cos(k0 * r)
    return b


def overlap(phi, template):
    """Normalised inner product; 1.0 = field is exactly the injected packet."""
    a = phi.ravel() - phi.mean()
    b = template.ravel() - template.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(a @ b / (na * nb))


# ----------------------------------------------------------------- gates ---

def run_gates(N=40, lam0=8.0, a=0.5, b=1.0, c=1.0, quintic=False, gamma=0.05,
              lam=0.0, tau=200.0, alpha=1.0, dt=0.05, seed=1,
              settle=1000, nrec=40, every=50, absthr=0.30,
              hold_steps=2000, seed_amp=1.2, seed_sigma=6.0, quiet=False):
    """One parameter set, all four gates, one dict back."""
    box = N ** 3
    k0 = 2 * np.pi / lam0
    kw = dict(N=N, lam0=lam0, a=a, b=b, c=c, quintic=quintic, gamma=gamma,
              lam=lam, tau=tau, alpha=alpha, dt=dt, amp=0.01, seed=seed)

    # --- U1/U2/U3 : let it settle, then watch it -------------------------
    u = Universe(**kw)
    u.step(settle)
    recs, counts, bigs, lams = [], [], [], []
    for r in range(nrec + 1):
        if r:
            u.step(every)
        cs = objects(u.phi, absthr)
        recs.append(cs)
        counts.append(len(cs))
        bigs.append(max([v for _, v in cs], default=0) / box)
        lams.append(u.peak_wavelength())
    life = track(recs, radius=5.0)
    half = counts[len(counts) // 2:]
    n_med = float(np.median(half))
    big_med = float(np.median(bigs[len(bigs) // 2:]))
    life_med = float(np.median(life))
    n_tracks = int(len(life))
    lam_ok = all(abs(L - lam0) / lam0 <= 0.20 for L in lams if np.isfinite(L))

    u0 = (n_med <= 3) and (big_med >= 0.20)
    u1 = n_med >= 20
    u2 = (life_med >= 0.8 * (nrec + 1)) and (n_tracks <= 1.5 * max(n_med, 1))
    u3 = bool(lam_ok)

    # --- U4 : put something in and come back -----------------------------
    u = Universe(**kw)
    u.step(settle)
    ctr = [(N // 4, N // 4, N // 4)]
    tmpl = gabor_packet(N, ctr, seed_amp, seed_sigma, k0)
    u.phi = u.phi + tmpl
    u.phi_o = u.phi_o + tmpl
    ov0 = overlap(u.phi, tmpl)
    u.step(hold_steps)
    cs = objects(u.phi, absthr)
    dist = min([float(np.linalg.norm(cc - np.asarray(ctr[0]))) for cc, _ in cs],
               default=np.inf)
    ov1 = overlap(u.phi, tmpl)
    u4 = (dist <= lam0) and (ov1 >= 0.20)

    res = dict(a=a, b=b, c=c, quintic=quintic, gamma=gamma, lam=lam, tau=tau,
               n_med=n_med, big_med=big_med, life_med=life_med,
               n_tracks=n_tracks, lam_end=lams[-1],
               seed_dist=dist, ov0=ov0, ov1=ov1,
               U0=u0, U1=u1, U2=u2, U3=u3, U4=u4)
    if not quiet:
        print(f"\n  a={a:+.2f} b={b:.1f} c={c:.1f} quintic={int(quintic)} "
              f"gamma={gamma:.3f} lam={lam:+.2f} tau={tau:g}")
        print(f"    objects (median, 2nd half)   {n_med:.0f}")
        print(f"    biggest as share of box      {100*big_med:.1f}%")
        print(f"    median track life            {life_med:.0f} / {nrec+1} "
              f"records   ({n_tracks} tracks)")
        print(f"    dominant wavelength (end)    {lams[-1]:.2f}  (lam0 {lam0})")
        print(f"    seeded packet: overlap       {ov0:.3f} -> {ov1:.3f}   "
              f"nearest object {dist:.1f} cells away")
        print(f"    U0 [{'V' if u0 else 'K'}] coarsens   "
              f"U1 [{'V' if u1 else 'K'}] many   "
              f"U2 [{'V' if u2 else 'K'}] persistent   "
              f"U3 [{'V' if u3 else 'K'}] scale   "
              f"U4 [{'V' if u4 else 'K'}] receptive")
    return res


# -------------------------------------------------------------- selftest ---

def selftest():
    ok = True

    def chk(n, cond, d=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {n}  {d}")

    print("SELFTEST")
    N = 32
    g = np.arange(N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')

    # S1 positive: a planted lattice of separated blobs is counted
    lat = np.zeros((N, N, N))
    ctrs = [(8, 8, 8), (8, 8, 24), (8, 24, 8), (24, 8, 8),
            (24, 24, 8), (24, 8, 24), (8, 24, 24), (24, 24, 24)]
    for cx, cy, cz in ctrs:
        lat += np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) / (2*2.0**2))
    n = len(objects(lat, 0.30))
    chk("S1 counter finds 8 planted blobs", n == 8, f"got {n}")

    # S2 negative: a DEAD field must report zero -- the bug being fixed
    dead = 3e-3 * np.random.default_rng(0).standard_normal((N, N, N))
    n_abs = len(objects(dead, 0.30))
    hi = np.percentile(dead, 99.5)
    n_pct = len(objects(dead, 0.5 * hi))
    chk("S2 dead field: absolute thr reports 0", n_abs == 0, f"got {n_abs}")
    chk("S2 dead field: percentile thr reports many (the old bug)",
        n_pct > 5, f"percentile thr got {n_pct}")

    # S3 positive: a rigidly drifting blob keeps ONE track
    recs = []
    for t in range(10):
        f = np.exp(-((X-(8+0.8*t))**2 + (Y-16)**2 + (Z-16)**2) / (2*2.0**2))
        recs.append(objects(f, 0.30))
    L = track(recs, radius=5.0)
    chk("S3 drifting blob = 1 track, full life",
        len(L) == 1 and L[0] == 10, f"{len(L)} tracks, life {L.tolist()}")

    # S4 negative: a blob that teleports each record must NOT keep one track
    rng = np.random.default_rng(1)
    recs = []
    for t in range(10):
        cx, cy, cz = rng.integers(6, 26, 3)
        f = np.exp(-((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) / (2*2.0**2))
        recs.append(objects(f, 0.30))
    L = track(recs, radius=5.0)
    chk("S4 teleporting blob breaks the track", len(L) > 3,
        f"{len(L)} tracks (a lenient matcher would say 1)")

    # S5 the seed-shape claim.  The load-bearing quantity is the fraction of
    # the seed's POWER that lands in the band the operator amplifies -- not
    # which discrete bin the argmax falls in.  At N=32 the k-grid spacing is
    # 2*pi/32 = 0.196 against k0 = 0.785, so an argmax test is quantisation-
    # limited and would need a tolerance wide enough to be meaningless.
    k0 = 2*np.pi/8.0
    kx = np.fft.fftfreq(N, 1.0)*2*np.pi
    kz = np.fft.rfftfreq(N, 1.0)*2*np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
    kmag = np.sqrt(KX**2+KY**2+KZ**2)
    band = np.abs(kmag - k0) / k0 <= 0.30

    def band_share(f):
        P = np.abs(np.fft.rfftn(f))**2
        return float(P[band].sum() / max(P.sum(), 1e-30))

    g2 = np.exp(-((X-16)**2 + (Y-16)**2 + (Z-16)**2) / (2*3.0**2))
    pk = gabor_packet(N, [(16, 16, 16)], 1.0, 6.0, k0)
    sg, sp = band_share(g2), band_share(pk)
    chk("S5 a plain Gaussian puts a minority of its power in the band",
        sg < 0.25, f"share {sg:.4f}")
    chk("S5 a Gabor packet puts most of its power there",
        sp > 0.50, f"share {sp:.4f}")
    chk("S5 packet beats Gaussian by >3x", sp > 3*sg,
        f"{sp:.4f} vs {sg:.4f}  ({sp/max(sg,1e-9):.1f}x)")

    # S6 overlap is sane two-sided
    chk("S6 overlap(x,x)=1", abs(overlap(pk, pk)-1.0) < 1e-9)
    chk("S6 overlap(noise,pk) ~ 0",
        abs(overlap(rng.standard_normal(pk.shape), pk)) < 0.15)

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


# ------------------------------------------------------------------ main ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=40)
    p.add_argument('--lam0', type=float, default=8.0)
    p.add_argument('--a', type=float, default=0.5)
    p.add_argument('--b', type=float, default=1.0)
    p.add_argument('--c', type=float, default=1.0)
    p.add_argument('--quintic', action='store_true')
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--lam', type=float, default=0.0)
    p.add_argument('--tau', type=float, default=200.0)
    p.add_argument('--settle', type=int, default=1000)
    p.add_argument('--nrec', type=int, default=40)
    p.add_argument('--every', type=int, default=50)
    p.add_argument('--hold', type=int, default=2000)
    p.add_argument('--absthr', type=float, default=0.30)
    p.add_argument('--gates', action='store_true')
    p.add_argument('--scan', action='store_true')
    p.add_argument('--out', type=str, default='')
    p.add_argument('--selftest', action='store_true')
    A = p.parse_args()

    if A.selftest:
        sys.exit(0 if selftest() else 1)

    common = dict(N=A.N, lam0=A.lam0, settle=A.settle, nrec=A.nrec,
                  every=A.every, hold_steps=A.hold, absthr=A.absthr)

    if A.gates:
        run_gates(a=A.a, b=A.b, c=A.c, quintic=A.quintic, gamma=A.gamma,
                  lam=A.lam, tau=A.tau, **common)
        return

    if A.scan:
        # supercritical arm: does inhibition ever buy count AND permanence?
        # subcritical arm: is there a window where a seeded packet survives?
        grid = list(itertools.product([0.5], [False], [0.0, -0.5, -1.0, -2.0],
                                      [200.0, 20.0, 5.0], [0.05]))
        grid += list(itertools.product([-0.05, -0.1, -0.3], [True], [0.0],
                                       [200.0], [0.02, 0.05]))
        rows = []
        t0 = time.time()
        print(f"scanning {len(grid)} parameter sets "
              f"(~{25*len(grid)/60:.0f} min at N={A.N})")
        for a, q, lam, tau, gam in grid:
            r = run_gates(a=a, b=A.b, c=A.c, quintic=q, gamma=gam,
                          lam=lam, tau=tau, quiet=True, **common)
            rows.append(r)
            flags = "".join('V' if r[f'U{i}'] else '.' for i in range(5))
            print(f"  a={a:+.2f} q={int(q)} lam={lam:+.1f} tau={tau:>5g} "
                  f"gam={gam:.2f} | n={r['n_med']:>4.0f} "
                  f"big={100*r['big_med']:>4.1f}% life={r['life_med']:>3.0f} "
                  f"ov={r['ov1']:+.2f} d={r['seed_dist']:>5.1f} | {flags}")
        print(f"\ndone in {time.time()-t0:.0f}s")
        win = [r for r in rows if r['U1'] and r['U2'] and r['U4']]
        if win:
            print(f"{len(win)} parameter set(s) pass U1+U2+U4 together:")
            for r in win:
                print(f"   a={r['a']:+.2f} quintic={int(r['quintic'])} "
                      f"lam={r['lam']:+.2f} tau={r['tau']:g} "
                      f"gamma={r['gamma']:.3f}")
            print("Those are the ones worth building an explorer on.")
        else:
            print("NO parameter set passes U1+U2+U4 together.")
            print("That is a result, not a failure of the search: this "
                  "equation, closed and damped, does not hold a population.")
            print("The next move is a DRIVE, not another parameter -- every "
                  "system in this ecosystem that stayed alive had one.")
        if A.out:
            with open(A.out, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {A.out}")
        return

    p.print_help()


if __name__ == '__main__':
    main()
