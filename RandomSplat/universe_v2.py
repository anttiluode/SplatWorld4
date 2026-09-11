#!/usr/bin/env python3
"""
universe_v2.py -- oscillatory field with a SELECTED wavelength, instrumented.

    d2phi/dt2 = -(lap + k0^2)^2 phi + a*phi + b*phi^3 - c*phi^5
                - gamma dphi/dt + lam*M*phi
    tau dM/dt = -M + alpha*phi^2

WHY THIS DOES NOT NaN, and why there is no clamp anywhere in it
---------------------------------------------------------------
The operator -(lap+k0^2)^2 is FOURTH order.  Explicit stepping needs
dt < 2/|k^2-k0^2|_max, which at the Nyquist edge is dt < 2/(pi^2-k0^2)^2.
v1 used physical units L=1.0 with N=64, putting k_Nyq at 201 and demanding
dt < 5e-5 while stepping at 0.05 -- overstepping by 1000x.  That is the NaN.

The cure is numerical, not physical: the stiff LINEAR part is solved
implicitly in Fourier space, where it is diagonal and free.  Only the cubic
and quintic are explicit, so dt is set by the pattern's own frequency ~sqrt(a),
not by the grid.  Nothing is ever clipped, so a divergence would still show
up as a divergence instead of being silently absorbed.

That distinction is the whole point.  An amplitude limiter inside the
equation (best.py's c^2 = 1/(1+T*phi^2)) does prevent overflow -- by
switching off the spatial coupling, which is exactly why its domain walls
came out one voxel wide.  A stabiliser in the physics cannot be told apart
from the physics.  Keep it in the integrator.

UNITS: everything is in grid cells.  dx = 1, box side = N cells,
k runs from 2*pi/N to pi.  lam0 is a wavelength in CELLS.
"""

import argparse, csv, sys
import numpy as np
from numpy.fft import rfftn, irfftn, fftfreq, rfftfreq
from scipy.ndimage import label


class Universe:
    def __init__(self, N=48, lam0=8.0, a=0.5, b=0.0, c=0.0, quintic=False,
                 gamma=0.05, dt=0.05, tau=200.0, alpha=1.0, lam=0.0,
                 amp=0.01, seed=0):
        self.N, self.dt, self.gamma = N, dt, gamma
        self.a, self.b, self.c, self.quintic = a, b, c, quintic
        self.tau, self.alpha, self.lam = tau, alpha, lam
        self.k0 = 2*np.pi/lam0
        self.lam0 = lam0

        kx = fftfreq(N, d=1.0)*2*np.pi
        kz = rfftfreq(N, d=1.0)*2*np.pi
        KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
        self.k2 = KX**2 + KY**2 + KZ**2
        self.kmag = np.sqrt(self.k2)

        # linear operator, treated implicitly
        self.L = self.a - (self.k2 - self.k0**2)**2

        rng = np.random.default_rng(seed)
        self.phi = (amp*rng.standard_normal((N, N, N))).astype(np.float64)
        self.phi_o = self.phi.copy()
        self.M = np.zeros_like(self.phi)

        d2 = 1.0/dt**2
        g2 = gamma/(2*dt)
        self.A = d2 + g2 - self.L/2.0
        self.B = -d2 + g2 + self.L/2.0

    # -- linear growth-rate band, computable before any stepping -----------
    def band(self):
        """(k_lo, k_hi) of linearly unstable wavenumbers, and fastest-growing k."""
        s = np.sqrt(self.a) if self.a > 0 else None
        if s is None:
            return None
        lo2, hi2 = self.k0**2 - s, self.k0**2 + s
        lo = np.sqrt(max(lo2, 0.0))
        return lo, np.sqrt(hi2), self.k0

    def step(self, n=1):
        for _ in range(n):
            nl = self.b*self.phi**3 - self.c*self.phi**5 if self.quintic \
                 else -self.b*self.phi**3
            if self.lam:
                nl = nl + self.lam*self.M*self.phi
            rhs = (2.0/self.dt**2)*rfftn(self.phi) + self.B*rfftn(self.phi_o) \
                  + rfftn(nl)
            new = irfftn(rhs/self.A, s=self.phi.shape)
            self.phi_o, self.phi = self.phi, new
            if self.lam or self.tau:
                self.M += (self.dt/self.tau)*(-self.M + self.alpha*self.phi**2)

    # -- instruments -------------------------------------------------------
    def energy(self):
        ph = rfftn(self.phi)
        w = np.ones_like(self.k2)*2.0; w[..., 0] = 1.0
        if self.N % 2 == 0:
            w[..., -1] = 1.0
        lin = float(np.sum(w*(self.k2 - self.k0**2)**2*np.abs(ph)**2))/self.N**3
        ke = float(np.sum(((self.phi - self.phi_o)/self.dt)**2))*0.5
        V = float(np.sum(-0.5*self.a*self.phi**2
                         + (-0.25*self.b*self.phi**4 + self.c/6*self.phi**6
                            if self.quintic else 0.25*self.b*self.phi**4)))
        return ke + 0.5*lin + V

    def peak_wavelength(self):
        P = np.abs(rfftn(self.phi))**2
        rb = np.round(self.kmag/(2*np.pi/self.N)).astype(int).ravel()
        nb = np.bincount(rb, minlength=self.N)
        pr = np.bincount(rb, P.ravel(), minlength=self.N)/np.maximum(nb, 1)
        pr[0] = 0.0
        m = np.argmax(pr[:self.N//2])
        return np.inf if m == 0 else self.N/m

    def count_objects(self, frac=0.5, min_vol=4):
        hi = np.percentile(self.phi, 99.5)
        if hi <= 0:
            return 0, 0.0
        m = self.phi > frac*hi
        lab, n = label(m, structure=np.ones((3, 3, 3)))
        vols = np.bincount(lab.ravel())[1:]
        keep = vols[vols >= min_vol]
        return len(keep), float(keep.mean()) if len(keep) else 0.0


def selftest():
    """Two-sided controls. Every one must pass before any result is believed."""
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    print("SELFTEST")
    # positive control: the instrument must report back the wavelength we set
    for L0 in (6.0, 8.0, 12.0):
        u = Universe(N=48, lam0=L0, a=0.5, b=1.0, amp=0.01, seed=1)
        u.step(4000)
        meas = u.peak_wavelength()
        chk(f"lam0={L0:>4} recovered", abs(meas - L0)/L0 < 0.15,
            f"measured {meas:.2f}")
    # negative control: a<0 has no unstable band -> field must die
    u = Universe(N=48, lam0=8.0, a=-0.5, b=1.0, amp=0.01, seed=1)
    a0 = np.abs(u.phi).max(); u.step(4000)
    chk("a<0 decays (no band)", np.abs(u.phi).max() < a0,
        f"{a0:.3e} -> {np.abs(u.phi).max():.3e}")
    # stability with NO clamp anywhere, across dt and initial amplitude
    for dt in (0.02, 0.05, 0.1):
        for amp in (0.01, 1.0):
            u = Universe(N=32, lam0=8.0, a=0.5, b=1.0, dt=dt, amp=amp, seed=2)
            u.step(2000)
            fin = np.abs(u.phi).max()
            chk(f"finite dt={dt} amp={amp}", np.isfinite(fin) and fin < 1e3,
                f"max|phi|={fin:.3f}")
    # object counter: a known lattice of blobs must be counted correctly
    u = Universe(N=32, lam0=8.0)
    g = np.arange(32)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    u.phi = (np.cos(2*np.pi*X/8)*np.cos(2*np.pi*Y/8)*np.cos(2*np.pi*Z/8))
    n, _ = u.count_objects(frac=0.5, min_vol=1)
    chk("counter finds 4^3/2 = 32 lobes", n == 32, f"got {n}")
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=48)
    p.add_argument('--lam0', type=float, default=8.0, help='wavelength in CELLS')
    p.add_argument('--a', type=float, default=0.5)
    p.add_argument('--b', type=float, default=1.0)
    p.add_argument('--c', type=float, default=1.0)
    p.add_argument('--quintic', action='store_true',
                   help='subcritical a*phi + b*phi^3 - c*phi^5 (localised states)')
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--amp', type=float, default=0.01, help='initial noise amplitude')
    p.add_argument('--lam', type=float, default=0.0, help='memory feedback')
    p.add_argument('--tau', type=float, default=200.0)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--record', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='')
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    u = Universe(N=args.N, lam0=args.lam0, a=args.a, b=args.b, c=args.c,
                 quintic=args.quintic, gamma=args.gamma, dt=args.dt,
                 tau=args.tau, lam=args.lam, amp=args.amp, seed=args.seed)
    lo, hi, kf = u.band()
    print(f"unstable band k in ({lo:.4f}, {hi:.4f}); fastest at k0={kf:.4f} "
          f"(lambda {2*np.pi/kf:.2f} cells)")
    print(f"box holds k from {2*np.pi/args.N:.4f} to {np.pi:.4f} -- "
          f"{'band is reachable' if hi > 2*np.pi/args.N else 'BAND UNREACHABLE'}")
    rows = []
    print(f"{'step':>7} {'energy':>13} {'max|phi|':>10} {'lambda':>8} {'objects':>8}")
    for s in range(0, args.steps+1, args.record):
        if s:
            u.step(args.record)
        n, v = u.count_objects()
        rows.append([s, u.energy(), np.abs(u.phi).max(), u.peak_wavelength(), n, v])
        print(f"{s:>7} {rows[-1][1]:>13.5e} {rows[-1][2]:>10.4f} "
              f"{rows[-1][3]:>8.2f} {n:>8d}")
    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'energy', 'max_phi', 'lambda', 'objects', 'mean_vol'])
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
