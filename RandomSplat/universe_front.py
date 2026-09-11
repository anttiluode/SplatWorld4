#!/usr/bin/env python3
"""
universe_front.py -- WHERE IS THE WORLD?  Front speed as the ruler.

WHAT KILLED THE LAST THREE ANSWERS
Three claims were made today from bad statistics, and one probe killed all
three.  Recording them because the mistake is the useful part:

  CLAIM  "a=-0.02 holds a seeded packet: overlap flat 0.30, distance 0.0"
  CLAIM  "the subcritical world retains what you put in (U4 [V] at a=-0.05)"
  CLAIM  "capacity is fine: K=1,2,4,8 packets, all 8 alive at 1800 steps"

  KILLED BY  measuring the envelope amplitude BETWEEN the objects as well as
  at them.  At K=8, contrast (amp at objects / amp between) went 26.8 at t=0
  to 0.9 at t=800 and stayed there -- the amplitude between the packets was
  HIGHER than at them.  The box had filled.  "All 8 alive" was true only in
  the sense that the field is nonzero everywhere.

  WHY THE OLD METRICS LIED.  Global overlap with a frozen template decays for
  a HEALTHY oscillating object (the object oscillates, the template does not)
  and also for a uniformly filled box.  It cannot tell those apart, so it
  cannot be the gate.  Connected-component counts fail differently: a radial
  carrier cos(k0 r) makes concentric shells, so one packet registers as 6-9
  components.  CONTRAST -- inside versus outside -- separates every case.

WHAT IS ACTUALLY HAPPENING, and it is completely standard
  a=-0.02 amp=1.2   far corner 0.00 -> 0.91 by step 700          INVADES
  a=-0.05 amp=1.2   far corner 0.00 -> 0.03 -> 0.12 -> 0.75      INVADES, slower
  a=-0.02 amp=0.6   0.32 -> 0.01 -> 0.00                          dies (sub-threshold)
  a=-0.10 amp=1.5   0.79 -> 0.06 -> 0.00                          dies (state unfavourable)

That is bistable front propagation.  A seed below the nucleation threshold
decays; a seed above it nucleates the patterned state, and then a FRONT
between patterned and uniform sweeps outward until the box is full.  There is
no third outcome at these settings -- the medium supports one thing, and that
thing wants to be everything.  Same wall as the supercritical arm reaching one
blob at 30% of the box, arriving from the other side.

WHERE THE WORLD IS
Between "invades" and "retreats" the front speed passes through ZERO.  In that
window the front is PINNED: a patch of pattern neither grows nor shrinks, so a
seeded object keeps the size you gave it, indefinitely, and several of them can
sit side by side without merging.  That is the localised-state (snaking) regime
and it is the only place a populated world can exist in this equation.

Bracketed by the runs above it lies in a in (-0.10, -0.05), possibly narrow.

WHY FRONT SPEED IS A BETTER RULER THAN ANY GATE I HAVE WRITTEN TODAY
It has a natural zero.  Every threshold in universe_hold.py was invented by me
(and one, the band-share ratio, had to be corrected against its own control).
Front speed needs no cutoff: positive is invasion, negative is retreat, zero is
the world.  The search is one-dimensional and bisection works on the sign.

    python3 universe_front.py --selftest
    python3 universe_front.py --speed --a -0.05
    python3 universe_front.py --bisect --lo -0.12 --hi -0.02
    python3 universe_front.py --hold --a <the pinned value> --K 8
"""

import argparse
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

from universe_v2 import Universe

ACTIVE = 0.20          # envelope amplitude counted as "patterned"
                       # background sits at 0.01-0.02, packets at 0.6-0.9,
                       # so this is a valley, not a tuned cutoff


# ------------------------------------------------------------ primitives ---

def packet(N, centres, amp, sigma, k0):
    """Gabor atom(s) with periodic (minimum-image) distance."""
    g = np.arange(N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    b = np.zeros((N, N, N))
    for c in centres:
        d2 = sum(np.minimum(np.abs(A - c[i]), N - np.abs(A - c[i])) ** 2
                 for i, A in enumerate((X, Y, Z)))
        r = np.sqrt(d2)
        b += amp * np.exp(-r ** 2 / (2 * sigma ** 2)) * np.cos(k0 * r)
    return b


def envelope(phi, sigma=2.0):
    """Local amplitude.  phi oscillates at k0; its ENVELOPE is what an object
    is.  Everything below is measured on this, never on phi directly."""
    return np.sqrt(gaussian_filter(phi ** 2, sigma=sigma))


def radius(e, N):
    """Effective radius of the patterned region, in cells.

    Volume-based rather than a ray cast, so a ragged or anisotropic front
    still gives a smooth monotone number to fit a slope to."""
    V = float((e > ACTIVE).sum())
    return (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0) if V > 0 else 0.0


def contrast(e, centres, N):
    """amp at the objects / amp at the point furthest from all of them.

    This is the probe that killed the three claims above.  A filled box scores
    ~1.  A genuine population scores >> 1."""
    at = float(np.mean([e[c] for c in centres]))
    g = np.arange(N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    far = np.full((N, N, N), 1e9)
    for c in centres:
        d2 = sum(np.minimum(np.abs(A - c[i]), N - np.abs(A - c[i])) ** 2
                 for i, A in enumerate((X, Y, Z)))
        far = np.minimum(far, d2)
    out = float(e.ravel()[np.argmax(far.ravel())])
    return at / max(out, 1e-9), at, out


# ------------------------------------------------------------ front speed ---

def front_speed(a, N=40, lam0=8.0, b=1.0, c=1.0, gamma=0.05, amp=1.2,
                sigma=5.0, settle=300, every=350, nrec=6, seed=1, quiet=False):
    """Seed one packet at the centre; fit dR/dt over the records.

    Returns (speed in cells/1000 steps, radii, final contrast).
    Records stop counting once R exceeds N/2.5, because past that the front has
    met its own periodic image and the slope means nothing."""
    k0 = 2 * np.pi / lam0
    u = Universe(N=N, lam0=lam0, a=a, b=b, c=c, quintic=True, gamma=gamma,
                 amp=0.005, seed=seed)
    u.step(settle)
    ctr = (N // 2, N // 2, N // 2)
    t = packet(N, [ctr], amp, sigma, k0)
    u.phi = u.phi + t
    u.phi_o = u.phi_o + t

    steps, radii = [], []
    for r in range(nrec + 1):
        if r:
            u.step(every)
        e = envelope(u.phi)
        R = radius(e, N)
        steps.append(r * every)
        radii.append(R)
        if R > N / 2.5:
            break
    e = envelope(u.phi)
    con, at, out = contrast(e, [ctr], N)

    # Fit AFTER the transient.  A seed is not the object's natural size, so
    # records 0-1 contain the relaxation from seed radius to preferred radius.
    # Including them reads a stable pinned object as "retreating" -- that is
    # exactly what happened at a=-0.0544 (R: 9.0 5.9 5.4 5.3 5.3 5.3 5.3,
    # dead flat for 2100 steps, reported as -1.29 cells/1000).
    s = np.array(steps, float)
    R = np.array(radii, float)
    fit = slice(2, None) if len(s) >= 5 else slice(0, None)
    speed = (float(np.polyfit(s[fit], R[fit], 1)[0] * 1000.0)
             if len(s[fit]) > 1 else np.nan)
    if not quiet:
        print(f"  a={a:+.4f}  R(t) " + " ".join(f"{x:4.1f}" for x in R))
        print(f"            speed {speed:+7.2f} cells/1000 steps   "
              f"contrast {con:6.2f}  (at {at:.2f} / out {out:.2f})   "
              f"{'dead' if at < ACTIVE else 'INVADES' if speed > 0.5 else
                  'RETREATS' if speed < -0.5 else 'PINNED -- an object'}")
    return speed, R, con


def bisect(lo, hi, tol=0.005, iters=7, **kw):
    """Find a where the front speed changes sign.

    Sign, not magnitude: bisection on a monotone-in-sign quantity needs no
    threshold at all, which is the point of using speed as the ruler."""
    print(f"bracketing the pinning window in a in ({lo}, {hi})")
    s_lo, _, _ = front_speed(lo, **kw)
    s_hi, _, _ = front_speed(hi, **kw)
    if np.sign(s_lo) == np.sign(s_hi):
        print("\nNO SIGN CHANGE across the bracket -- both ends do the same "
              "thing, so the window is not here (or does not exist).")
        print("Widen the bracket, or vary gamma/amp: pinning also depends on "
              "damping and on how big the seed was.")
        return None
    for i in range(iters):
        mid = 0.5 * (lo + hi)
        s, _, con = front_speed(mid, **kw)
        if abs(hi - lo) < tol:
            break
        if np.sign(s) == np.sign(s_lo):
            lo, s_lo = mid, s
        else:
            hi, s_hi = mid, s
    a_star = 0.5 * (lo + hi)
    print(f"\nsign change at a = {a_star:+.4f}  (bracket width {abs(hi-lo):.4f})")
    print("That is where a patch of pattern neither grows nor shrinks.")
    print(f"Next: python3 universe_front.py --hold --a {a_star:.4f} --K 8")
    return a_star


# ------------------------------------------------------------------ hold ---

def hold(a, K=8, N=40, lam0=8.0, gamma=0.05, amp=1.2, sigma=5.0,
         settle=300, steps=6000, every=750, seed=1):
    """THE TEST THAT MATTERS: K seeded objects, do they stay K separate things?

    Reports contrast at every record.  Contrast ~1 means the box filled and
    there are no objects however healthy the amplitudes look."""
    k0 = 2 * np.pi / lam0
    q = N // 4
    grid = [(q, q, q), (3*q, 3*q, q), (3*q, q, 3*q), (q, 3*q, 3*q),
            (q, q, 3*q), (3*q, 3*q, 3*q), (3*q, q, q), (q, 3*q, q)][:K]
    u = Universe(N=N, lam0=lam0, a=a, b=1.0, c=1.0, quintic=True, gamma=gamma,
                 amp=0.005, seed=seed)
    u.step(settle)
    t = packet(N, grid, amp, sigma, k0)
    u.phi = u.phi + t
    u.phi_o = u.phi_o + t
    print(f"  K={K} objects at a={a:+.4f}")
    print(f"  {'step':>6} {'contrast':>9} {'at':>7} {'between':>8} {'R':>6}  verdict")
    ok = None
    for r in range(steps // every + 1):
        if r:
            u.step(every)
        e = envelope(u.phi)
        con, at, out = contrast(e, grid, N)
        R = radius(e, N)
        v = "FILLED" if con < 2.0 else ("dead" if at < ACTIVE else "objects")
        print(f"  {r*every:>6} {con:>9.2f} {at:>7.3f} {out:>8.3f} {R:>6.1f}  {v}")
        ok = (con >= 2.0 and at >= ACTIVE)
    print(f"\n  W1 [{'V' if ok else 'K'}]  {K} placed objects still distinct "
          f"after {steps} steps")
    return ok


# -------------------------------------------------------------- selftest ---

def selftest():
    ok = True

    def chk(n, cond, d=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {n}  {d}")

    print("SELFTEST")
    N = 32
    k0 = 2 * np.pi / 8.0

    # T1 two-sided: contrast separates a population from a filled box
    sep = packet(N, [(8, 8, 8), (24, 24, 24)], 1.2, 4.0, k0)
    e = envelope(sep)
    c_sep, _, _ = contrast(e, [(8, 8, 8), (24, 24, 24)], N)
    filled = np.cos(k0 * np.arange(N))[:, None, None] * np.ones((N, N, N))
    c_fill, _, _ = contrast(envelope(filled), [(8, 8, 8), (24, 24, 24)], N)
    chk("T1 contrast is high for separated packets", c_sep > 3.0,
        f"{c_sep:.1f}")
    chk("T1 contrast is ~1 for a filled box", c_fill < 2.0, f"{c_fill:.2f}")

    # T2 two-sided: radius tracks a growing region and ignores amplitude
    small = envelope(packet(N, [(16, 16, 16)], 1.2, 3.0, k0))
    big = envelope(packet(N, [(16, 16, 16)], 1.2, 7.0, k0))
    chk("T2 radius grows with envelope width",
        radius(big, N) > 1.5 * radius(small, N),
        f"{radius(small,N):.1f} -> {radius(big,N):.1f}")
    faint = envelope(packet(N, [(16, 16, 16)], 0.05, 7.0, k0))
    chk("T2 radius is 0 for a sub-threshold field", radius(faint, N) == 0.0)

    # T3 the metric that lied, kept as a regression so it is not reused
    g = np.arange(N)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    rr = np.sqrt((X-16)**2 + (Y-16)**2 + (Z-16)**2)
    env = 1.2 * np.exp(-rr**2 / (2*5.0**2))
    tmpl = env * np.cos(k0*rr)               # the object
    shifted = env * np.cos(k0*rr + 1.3)      # SAME object, carrier phase moved
    def ov(p, t):
        A, B = p.ravel() - p.mean(), t.ravel() - t.mean()
        return float(A @ B / max(np.linalg.norm(A) * np.linalg.norm(B), 1e-12))
    chk("T3 template overlap falls for a merely PHASE-SHIFTED object "
        "(why it cannot be the gate)", ov(shifted, tmpl) < 0.99,
        f"overlap {ov(shifted, tmpl):.3f} for an object that never moved")

    # T4 speed sign on a planted, non-PDE synthetic front
    rs = [3.0, 5.0, 7.0, 9.0]
    sp = float(np.polyfit(np.arange(4) * 350.0, rs, 1)[0] * 1000)
    chk("T4 growing radii give positive speed", sp > 0, f"{sp:+.1f}")
    sp = float(np.polyfit(np.arange(4) * 350.0, rs[::-1], 1)[0] * 1000)
    chk("T4 shrinking radii give negative speed", sp < 0, f"{sp:+.1f}")
    sp = float(np.polyfit(np.arange(4) * 350.0, [6.0] * 4, 1)[0] * 1000)
    chk("T4 constant radii give zero speed", abs(sp) < 1e-6, f"{sp:+.2e}")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=40)
    p.add_argument('--lam0', type=float, default=8.0)
    p.add_argument('--a', type=float, default=-0.05)
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--amp', type=float, default=1.2)
    p.add_argument('--sigma', type=float, default=5.0)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--lo', type=float, default=-0.12)
    p.add_argument('--hi', type=float, default=-0.02)
    p.add_argument('--speed', action='store_true')
    p.add_argument('--bisect', action='store_true')
    p.add_argument('--hold', action='store_true')
    p.add_argument('--selftest', action='store_true')
    A = p.parse_args()

    if A.selftest:
        sys.exit(0 if selftest() else 1)
    kw = dict(N=A.N, lam0=A.lam0, gamma=A.gamma, amp=A.amp, sigma=A.sigma)
    if A.speed:
        front_speed(A.a, **kw)
    elif A.bisect:
        bisect(A.lo, A.hi, **kw)
    elif A.hold:
        hold(A.a, K=A.K, N=A.N, lam0=A.lam0, gamma=A.gamma, amp=A.amp,
             sigma=A.sigma, steps=A.steps)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
