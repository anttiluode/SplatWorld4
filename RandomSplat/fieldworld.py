#!/usr/bin/env python3
"""
fieldworld.py -- splats as neurons.  Can the field occlude in a way DEPTH CANNOT?

THE TRAP, THIRD TIME, AND WHY THIS TEST IS DIFFERENT
The proposed benchmark was "can the system learn viewpoint-consistent hiding
without assigning every object a stored depth coordinate?"  Train a decoder on
3D-derived data and it will learn depth into its weights: same smuggle as
f_i(q,theta,L) in flatworld and theta_n in the vector delay line.  Where the
number lives is bookkeeping.

So this file does not ask whether the field can IMITATE depth.  It asks whether
the field can do something depth is INCAPABLE of.  That question has an answer
that cannot be faked, because it is about the structure of the relation itself.

THE STRUCTURAL FACT DEPTH IMPLIES
A depth assignment gives every object a real number z_i.  "A occludes B" is then
z_A < z_B, which is a TOTAL ORDER, and total orders are TRANSITIVE:
    A occludes B  and  B occludes C   =>   A occludes C
An occlusion CYCLE (A over B, B over C, C over A) cannot be produced by any
assignment of depths whatsoever.  For simple convex objects it does not happen in
a 3D world.  So:

    cycles impossible  ->  ownership is a total order  ->  it IS depth, renamed
    cycles observed    ->  the field computes something no z-buffer can

REGISTERED BEFORE RUNNING (two-sided by construction)
  C0  AMPLITUDE-ONLY control must give ZERO cycles.  If ownership is decided by
      one scalar per packet, sorting by that scalar is transitive and cycles are
      impossible a priori.  If C0 shows cycles the instrument is broken and
      nothing below counts.
  C1  FULL configuration (orientation, carrier phase and scale all free) is
      predicted to give cycles at a nonzero rate, because ownership is then
      decided LOCALLY in each overlap region -- A may win the A/B overlap by
      phase alignment while losing the A/C overlap elsewhere.  Pairwise-local,
      not global-scalar.
  C2  cycle rate must RISE with phase/orientation diversity and FALL to the C0
      floor as the packets are made identical in everything but amplitude.
      A rate that does not respond to the driver is not the claimed mechanism.

WHAT MAKES OWNERSHIP HAPPEN AT ALL (measured in occlusion.py, this repo)
Packets enter one PRE-SIGMOID sum.  Where it saturates, sigma'(u) -> 0 and
weaker structure stops reaching the output.  Measured: local gain falls
0.950 -> 0.200 as a coarse packet grows; the nonlinearity removes a further
~58% of the weaker packet on top of plain loudness masking (ratio 1.00 -> 0.42);
and a coarse packet hides a fine one ~3x better than a same-scale packet does
(0.03 vs 0.10 surviving).  Occlusion here is saturation, not depth ordering.

HONEST SCOPE.  This measures the OWNERSHIP RELATION in a single static field.
It does not establish viewpoint-consistent hiding over an observer trajectory,
which is a strictly stronger requirement and is not tested here.

    python3 fieldworld.py --selftest
    python3 fieldworld.py --cycles
    python3 fieldworld.py --sweep
"""

import argparse
import itertools
import sys

import numpy as np

S = 96
_g = np.linspace(-1, 1, S)
X, Y = np.meshgrid(_g, _g)


def packet(cx, cy, sigma, freq, theta, phase, amp):
    xr = (X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    env = np.exp(-r2 / (2 * sigma ** 2))
    return amp * env * np.cos(2 * np.pi * freq * xr + phase), env


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def owns(pi, ei, pj, ej):
    """Who owns the overlap of packets i and j?

    Ownership = whose structure SURVIVES the joint nonlinear readout better,
    each measured against how well it survives alone.  Normalising by the solo
    case is what makes this a comparison of SUPPRESSION rather than of raw
    loudness -- without it the louder packet trivially 'wins' and the relation
    is a scalar sort by construction."""
    m = (ei * ej) > 0.10                       # the region they actually share
    if m.sum() < 30:
        return None
    def keep(target, field):
        a = field[m] - field[m].mean()
        b = target[m] - target[m].mean()
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / n) if n > 1e-12 else 0.0
    both = sigmoid(pi + pj)
    si = keep(pi, both) / max(abs(keep(pi, sigmoid(pi))), 1e-9)
    sj = keep(pj, both) / max(abs(keep(pj, sigmoid(pj))), 1e-9)
    if abs(si - sj) < 0.02:                    # too close to call
        return None
    return 0 if si > sj else 1                 # 0 = i owns, 1 = j owns


def draw(rng, diversity=1.0, amp_only=False):
    """Three packets placed so that all three pairs overlap.

    `diversity` scales how much orientation, phase and scale are allowed to
    differ.  amp_only pins every packet identical except amplitude -- the C0
    control, where ownership can only be a scalar sort."""
    R = 0.34
    P = []
    for k in range(3):
        a = 2 * np.pi * k / 3 + rng.uniform(-0.2, 0.2)
        cx, cy = R * np.cos(a), R * np.sin(a)
        amp = rng.uniform(2.0, 9.0)
        if amp_only:
            P.append(packet(cx, cy, 0.30, 4.0, 0.0, 0.0, amp))
        else:
            P.append(packet(cx, cy,
                            0.30 * (1 + diversity * rng.uniform(-0.45, 0.45)),
                            4.0 * (1 + diversity * rng.uniform(-0.5, 0.9)),
                            diversity * rng.uniform(0, np.pi),
                            diversity * rng.uniform(0, 2 * np.pi),
                            amp))
    return P


def cycle_rate(n=300, diversity=1.0, amp_only=False, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    cycles = complete = 0
    for _ in range(n):
        P = draw(rng, diversity, amp_only)
        rel = {}
        for i, j in itertools.combinations(range(3), 2):
            w = owns(P[i][0], P[i][1], P[j][0], P[j][1])
            if w is None:
                rel = None
                break
            rel[(i, j)] = i if w == 0 else j
        if rel is None:
            continue
        complete += 1
        # a cycle on three nodes: every node wins exactly one of its two pairs
        wins = [0, 0, 0]
        for w in rel.values():
            wins[w] += 1
        if sorted(wins) == [1, 1, 1]:
            cycles += 1
    r = cycles / max(complete, 1)
    if verbose:
        print(f"    {complete:>4} decidable configs, {cycles:>4} cyclic "
              f"-> rate {r:.3f}")
    return r, complete, cycles


def selftest():
    ok = True

    def chk(n, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}  {d}")

    print("SELFTEST")
    rng = np.random.default_rng(0)

    # T1 two-sided: ownership must be decidable and must be ASYMMETRIC
    pi, ei = packet(-0.15, 0, 0.32, 2.0, 0.0, 0.0, 8.0)
    pj, ej = packet(+0.15, 0, 0.20, 7.0, 1.1, 0.7, 1.5)
    w = owns(pi, ei, pj, ej)
    chk("T1 a strong coarse packet owns a weak fine one", w == 0, f"owner {w}")
    w2 = owns(pj, ej, pi, ei)
    chk("T1 the relation is antisymmetric under argument swap", w2 == 1,
        f"owner {w2}")

    # T2 two-sided: identical packets are UNDECIDABLE, not arbitrarily ordered
    pa, ea = packet(-0.10, 0, 0.30, 4.0, 0.0, 0.0, 5.0)
    pb, eb = packet(+0.10, 0, 0.30, 4.0, 0.0, 0.0, 5.0)
    chk("T2 two identical packets give no owner", owns(pa, ea, pb, eb) is None)

    # T3 the cycle DETECTOR itself, on planted relations rather than fields
    def cyc(wins):
        return sorted(wins) == [1, 1, 1]
    chk("T3 detector fires on a planted 3-cycle", cyc([1, 1, 1]))
    chk("T3 detector silent on a planted total order", not cyc([2, 1, 0]))

    # T4 the C0 control is a THEOREM, not a hope: sorting by one scalar
    #    cannot cycle.  Verified numerically on planted scalar orderings.
    bad = 0
    for _ in range(2000):
        z = rng.random(3)
        wins = [0, 0, 0]
        for i, j in itertools.combinations(range(3), 2):
            wins[i if z[i] > z[j] else j] += 1
        if sorted(wins) == [1, 1, 1]:
            bad += 1
    chk("T4 a scalar sort never cycles (2000 draws)", bad == 0, f"{bad} cycles")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--cycles', action='store_true')
    p.add_argument('--sweep', action='store_true')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.cycles or a.sweep:
        print("C0 CONTROL -- amplitude only, everything else identical")
        print("   (a scalar sort is transitive, so this MUST read 0.000)")
        r0, _, _ = cycle_rate(a.n, amp_only=True, verbose=True)
        print(f"   C0 [{'V' if r0 == 0.0 else 'K'}]  rate {r0:.3f}\n")

        print("C1 -- full configuration, orientation/phase/scale free")
        r1, _, _ = cycle_rate(a.n, diversity=1.0, verbose=True)
        print(f"   C1 [{'V' if r1 > 0.0 else 'K'}]  rate {r1:.3f}")
        if r1 > 0:
            print("   => the field produced occlusion relations that NO "
                  "assignment of depths can produce.")
        else:
            print("   => ownership is a total order here; it is depth, renamed.")

    if a.sweep:
        print("\nC2 -- does the cycle rate track the driver?")
        print(f"  {'diversity':>10} {'cycle rate':>12}")
        rates = []
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            r, _, _ = cycle_rate(a.n, diversity=d, seed=1)
            rates.append(r)
            print(f"  {d:>10.2f} {r:>12.3f}")
        rising = rates[-1] > rates[0]
        print(f"  C2 [{'V' if rising else 'K'}]  "
              f"{'rate responds to phase/orientation diversity' if rising else 'no response — mechanism not demonstrated'}")

    if not (a.cycles or a.sweep or a.selftest):
        p.print_help()


if __name__ == '__main__':
    main()
