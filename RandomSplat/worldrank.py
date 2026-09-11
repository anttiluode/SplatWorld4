#!/usr/bin/env python3
"""
worldrank.py -- DOES ACTION-CONDITIONING ORGANISE THE LATENT INTO GEOMETRY?

The room result in one line: room.pt reads effective rank 2.186 with
s4^2/s3^2 = 0.838 -- the "third component" is the same size as the fourth, so
the packet motion is a flat morph -- while the latent carries 3.847 effective
dimensions and the coefficients 7.823.  The information about where the camera
was IS in the latent.  The decoder spends it on appearance.

ChatGPT's proposal, implemented end to end:

    build a scene where the answer is KNOWN (sphere, cube, checkered ground),
    move a camera around it, train three models --

        A   plain autoencoder            x_t -> z -> x_t
        B   + temporal prediction        z_t -> T(z_t)      -> x_{t+1}
        C   + action input               z_t -> T(z_t, a_t) -> x_{t+1}

    -- and run the SAME diagnostics on all of them.  If only C develops
    geometry, action is what organises the latent.  If even C stays rank 2,
    the limit is architectural and no conditioning fixes it.

THE CORRECTION THAT PROMPTED THIS FILE, RECORDED BECAUSE IT WAS MINE
I wrote "the geometry channel can absorb at most 6 numbers per frame".  False
as an architectural claim: the decoder emits N x 2 free coordinates per frame
and nothing forbids them from being anything.  Rank 2 is what the LEARNED
trajectories happen to be -- empirical, not architectural.  That distinction is
the whole reason this is worth running.  An architectural ceiling cannot be
trained away; an empirical one might be.

WHAT GROUND TRUTH BUYS THAT MORE REAL FOOTAGE CANNOT
  * AN ORACLE ROW.  Push real 3D surface points through the real camera path
    and through the SAME rank instrument.  It must show a rank-3 cliff.  If it
    does not, the instrument cannot see depth in this scene at this frame count
    and every model row is meaningless.  The room run never had this.
  * EVERY GATE CALIBRATED AGAINST THE ORACLE rather than against a number I
    invented.  Ten gates in this project have now been mis-specified, several
    because I picked a threshold out of the air.  Here the scene's own ceiling
    is measured first and the model is scored as a fraction of it.
  * THE AFFINE-RESIDUAL SPLIT.  rank(W) = 2 means positions are a fixed flat
    shape under a per-frame 2x2 matrix plus translation, so the displacement
    between two frames is EXACTLY affine in position.  Fit and remove that
    affine and what is left is precisely what no rank-2 model can express.  Its
    energy fraction is parallax, measured without needing depth at all, and its
    direction answers "is right actually right" -- his complaint, as a number.

WHY THE RAW FLOW WOULD HAVE LIED, caught by this file's own selftest
The first version compared the model's displacement field against the true
flow directly.  Camera orbit makes true flow overwhelmingly a global pan, so
the shuffled-correspondence null came back at +0.649 -- i.e. a model that
merely translated everything uniformly would have scored ~0.9 and "passed".
Same for parallax: random affine fields scored corr(|flow|, 1/depth) of +0.13,
+0.61, +0.17 against the true field's +0.76.  Both metrics now run on the
affine residual, where the null measures -0.14 and a planted affine field has
residual energy 1e-28.

FOURTH ARM, AND THE WAY I GOT IT WRONG THE FIRST TIME
C has more inputs and more parameters than B, so it needs a capacity control.
v1 built that control by SHUFFLING the action across the batch.  On the first
real run (12000 steps, 96px, 160 packets) that arm did not train at all: loss
0.114 -> 0.390 -> 0.182 -> 0.363 -> 0.284, final PSNR 8.16 against 17.9-18.2
for A/B/C, packet mobility exactly 0.000, s3 = s4 = s5 = 0.000.

A wrong action does not remove information, it makes the prediction target
UNLEARNABLE -- the gradient through the shared encoder and decoder is then
pure noise, and it destroyed the reconstruction path.  So the shuffled arm was
not a capacity control; it was a poisoned trunk.  W3 "passed" for that reason
alone, and W0 failed on a 10 dB spread that came entirely from this arm.

  C0  is now a ZERO action vector: same parameters, same input width, channel
      carries nothing.  The prediction task degenerates to arm B's, which is
      learnable, so the arms stay comparable.  This is the W3 arm.
  Cx  keeps the shuffled version, not in the default arm list, so the finding
      above stays reproducible.

FIFTH ARM -- D, the credit-assignment test
Everything measured so far says the latent HAS the motion information and the
decoder spends it on coefficients rather than positions.  D is C plus a penalty
on the temporal change of the coefficient channel (--coefpen), so the cheapest
way down the loss is to MOVE PACKETS.  It is a deliberate bias, not a neutral
regulariser: if D's rank climbs toward the oracle while its PSNR stays inside
W0, the flat geometry was a training-dynamics artefact and not a ceiling.  If
D's PSNR collapses instead, the model would rather be blurry than be 3D, which
is itself the answer.

PRE-REGISTERED GATES
  W0  comparability   eval PSNR spread across arms <= 1.5 dB, else all VOID
  W1  instrument      ORACLE shows a cliff: s3^2/s2^2 >= 0.05, s4^2/s3^2 <= 0.25
  W2  geometry in C   C s3^2/s2^2 >= 0.5x ORACLE's, C s4^2/s3^2 <= 2x ORACLE's,
                      and C s3^2/s2^2 >= 3x arm A's
  W3  action did it   C s3^2/s2^2 >= 2x C0's (zero-action capacity control)
  W4  direction       C G3 >= 0.40, >= 2x its own shuffled null,
                      and >= arm A's G3 + 0.20
  W5  parallax        C's G5 field match >= 0.25 and >= 5x its shuffled null
  W6  credit          arm D's cliff >= 0.5x ORACLE and >= 3x the C/A baseline,
                      while still inside W0

HONEST SCOPE
Synthetic scene, small models, short training, one seed unless --seed is
varied.  A NEGATIVE result is weaker than a positive one -- it could be
undertraining -- so PSNR and a loss tail print for every arm and W0 refuses to
compare arms that did not reach comparable fits.  Deliberately an
AUTOENCODER, not a VAE: no beta, no posterior collapse to argue about.
This is a proxy for TinyAvatar's architecture, not TinyAvatar itself.

    python3 worldrank.py --selftest      (no training, ~20 s)
    python3 worldrank.py --smoke         (tiny end-to-end, every path)
    python3 worldrank.py --run           (the experiment)
    python3 worldrank.py --run --steps 12000 --size 96 --packets 160
    python3 worldrank.py --run --arms A,C,D --seeds 3 --steps 12000 --size 96
"""

import argparse
import math
import sys
import time

import numpy as np

# ===========================================================================
# SCENE -- analytic ray cast.  Ground truth image, depth, 3D hit points.
# Tuned near-orthographic (narrow fov, far camera) so the ORACLE shows a clean
# rank-3 cliff: at fov 0.24 / dist 8.8 the singulars run 17.5 16.2 4.3 1.4 0.4.
# A wide fov or a large floor puts perspective into a 4th component and blunts
# the very cliff the experiment is looking for.
# ===========================================================================

SPHERE_C = np.array([0.62, -0.32, 0.34])
SPHERE_R = 0.44
CUBE_C = np.array([-0.58, -0.28, -0.30])
CUBE_H = np.array([0.38, 0.52, 0.38])
FLOOR_Y = -0.82
FLOOR_EXT = 3.2
LIGHT = np.array([0.45, 0.82, 0.36]) / np.linalg.norm([0.45, 0.82, 0.36])
FOV = 0.24

PHI_R, HGT_R, DST_R = (-0.95, 0.95), (1.30, 2.20), (8.30, 9.30)
ORBIT_PHI, ORBIT_HGT, ORBIT_DST = 0.90, 1.75, 8.80


def camera(pose):
    """pose = (phi, height, dist) -> (eye, right, up, fwd)."""
    phi, hgt, dist = float(pose[0]), float(pose[1]), float(pose[2])
    eye = np.array([dist * math.sin(phi), hgt, dist * math.cos(phi)])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    return eye, right, np.cross(right, fwd), fwd


def render(pose, S=64):
    """Grayscale image, camera-space depth, 3D hit points, hit mask."""
    eye, right, up, fwd = camera(pose)
    t = math.tan(FOV)
    g = (np.arange(S) + 0.5) / S * 2.0 - 1.0
    d = fwd[None, None, :] + (g[None, :] * t)[..., None] * right \
        + (-g[:, None] * t)[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)

    INF = 1e9
    t_best = np.full((S, S), INF)
    nrm = np.zeros((S, S, 3))
    alb = np.zeros((S, S))

    oc = eye - SPHERE_C
    b = d @ oc
    disc = b ** 2 - (float(oc @ oc) - SPHERE_R ** 2)
    m = disc > 0
    ts = np.where(m, -b - np.sqrt(np.maximum(disc, 0.0)), INF)
    upd = m & (ts > 1e-4) & (ts < t_best)
    t_best = np.where(upd, ts, t_best)
    nrm = np.where(upd[..., None], (eye + ts[..., None] * d - SPHERE_C) / SPHERE_R, nrm)
    alb = np.where(upd, 0.86, alb)

    lo, hi = CUBE_C - CUBE_H, CUBE_C + CUBE_H
    dd = np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo - eye) / dd, (hi - eye) / dd
    tn = np.max(np.minimum(t1, t2), axis=-1)
    tf = np.min(np.maximum(t1, t2), axis=-1)
    m = tf > np.maximum(tn, 1e-4)
    tc = np.where(m, tn, INF)
    upd = m & (tc < t_best)
    t_best = np.where(upd, tc, t_best)
    rel = (eye + tc[..., None] * d - CUBE_C) / CUBE_H
    ax = np.argmax(np.abs(rel), axis=-1)
    n = np.zeros((S, S, 3))
    for a in range(3):
        sel = ax == a
        n[sel, a] = np.sign(rel[sel, a])
    nrm = np.where(upd[..., None], n, nrm)
    alb = np.where(upd, 0.52, alb)

    tp = np.where(np.abs(d[..., 1]) > 1e-9, (FLOOR_Y - eye[1]) / dd[..., 1], INF)
    p = eye + tp[..., None] * d
    m = (tp > 1e-4) & (np.abs(p[..., 0]) < FLOOR_EXT) & (np.abs(p[..., 2]) < FLOOR_EXT)
    tp = np.where(m, tp, INF)
    upd = m & (tp < t_best)
    t_best = np.where(upd, tp, t_best)
    chk = (np.floor(p[..., 0] * 1.6) + np.floor(p[..., 2] * 1.6)) % 2 == 0
    nrm = np.where(upd[..., None], np.array([0.0, 1.0, 0.0]), nrm)
    alb = np.where(upd, np.where(chk, 0.74, 0.30), alb)

    hit = t_best < INF / 2
    img = np.where(hit, alb * (0.26 + 0.74 * np.clip(nrm @ LIGHT, 0.0, 1.0)), 0.07)
    X = eye + np.where(hit, t_best, 0.0)[..., None] * d
    return img.astype(np.float32), np.where(hit, t_best, np.inf), X, hit


def project(pose, X, S=64):
    """3D points -> pixel coords (col, row) in [0,S).  Returns (px, py, z)."""
    eye, right, up, fwd = camera(pose)
    w = X - eye
    z = w @ fwd
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    t = math.tan(FOV)
    px = ((w @ right) / (zz * t) + 1.0) * 0.5 * S
    py = (1.0 - (w @ up) / (zz * t)) * 0.5 * S
    return px, py, z


def train_traj(n, seed=0):
    """Smooth random walk over all three pose DOF, so pose->latent is fittable."""
    rng = np.random.default_rng(seed)

    def fold(a, lo, hi):
        span = hi - lo
        a = np.abs((a - lo) % (2 * span))
        return lo + np.where(a > span, 2 * span - a, a)

    phi = np.cumsum(rng.normal(0, 0.055, n))
    hgt = np.cumsum(rng.normal(0, 0.024, n)) + np.mean(HGT_R)
    dst = np.cumsum(rng.normal(0, 0.022, n)) + np.mean(DST_R)
    return np.stack([fold(phi, *PHI_R), fold(hgt, *HGT_R), fold(dst, *DST_R)], 1)


def orbit_traj(F):
    """Held-out pure orbit: the Tomasi-Kanade case, true rank 3."""
    return np.stack([np.linspace(-ORBIT_PHI, ORBIT_PHI, F),
                     np.full(F, ORBIT_HGT), np.full(F, ORBIT_DST)], 1)


# ===========================================================================
# INSTRUMENTS -- pure numpy, so the selftest can plant known answers
# ===========================================================================

def rank_stats(P):
    """P: (F,N,2).  Tomasi-Kanade measurement matrix statistics.

    Per-frame centring removes translation; what remains is shape.  Rigid 3D
    under orthographic projection factors as rank 3 exactly; a 2D affine morph
    of a fixed shape factors as rank 2 exactly.  Read the CLIFF (s3 clear of
    s4), never the participation ratio -- that statistic is energy-weighted and
    reads 2.4 on a genuine 3-component spectrum, which is how PC1 was
    mis-specified earlier in this project."""
    F, N, _ = P.shape
    W = P - P.mean(axis=1, keepdims=True)
    W = np.concatenate([W[:, :, 0], W[:, :, 1]], axis=0)
    s = np.linalg.svd(W, compute_uv=False)[:8]
    ev = s ** 2 / max((s ** 2).sum(), 1e-30)
    return dict(s=s, eff=float(np.exp(-(ev * np.log(ev + 1e-30)).sum())),
                r32=float(s[2] ** 2 / max(s[1] ** 2, 1e-30)) if len(s) > 2 else 0.0,
                r43=float(s[3] ** 2 / max(s[2] ** 2, 1e-30)) if len(s) > 3 else 0.0,
                top2=float((s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-30)))


def dof(M):
    """Participation ratio of the sample covariance.  M: (F, D)."""
    M = np.asarray(M, dtype=np.float64)
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    ev = s ** 2 / max((s ** 2).sum(), 1e-30)
    cum = np.cumsum(ev)
    return (float(np.exp(-(ev * np.log(ev + 1e-30)).sum())),
            int(np.searchsorted(cum, 0.95) + 1), int(np.searchsorted(cum, 0.99) + 1))


def mobility(P):
    """median per-packet travel / median nearest-neighbour spacing at frame 0."""
    trav = np.linalg.norm(np.diff(P, axis=0), axis=2).sum(axis=0)
    d = np.linalg.norm(P[0][:, None, :] - P[0][None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(trav) / max(np.median(d.min(axis=1)), 1e-9))


def affine_residual(disp, pos):
    """Split a displacement field into what a 2D affine morph can produce and
    what it cannot.  Returns (residual, residual energy fraction).

    This is the exact complement of the rank test: rank 2 <=> the displacement
    between any two frames is affine in position <=> this fraction is zero."""
    A = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, disp, rcond=None)
    res = disp - A @ coef
    return res, float((res ** 2).sum() / max((disp ** 2).sum(), 1e-30))


def dir_fidelity(a_field, b_field, seed=0, n_perm=64):
    """G3: median cosine between two fields, plus a shuffled-correspondence null.

    Run this on AFFINE RESIDUALS, never on raw flow.  Raw camera-orbit flow is
    dominated by a global pan, so its shuffled null measures +0.65 and any
    uniform translation would score ~0.9."""
    a = a_field / np.maximum(np.linalg.norm(a_field, axis=1, keepdims=True), 1e-12)
    b = b_field / np.maximum(np.linalg.norm(b_field, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    nulls = [np.median((a[rng.permutation(len(a))] * b).sum(1))
             for _ in range(n_perm)]
    return float(np.median((a * b).sum(1))), float(np.median(nulls))


def field_match(a_field, b_field, seed=0, n_perm=64):
    """G5: squared GLOBAL cosine between two displacement fields, plus its
    shuffled null.  1 iff one field is a scalar multiple of the other; the
    analytic null for an unrelated field is ~1/(2n).

    This exists because W5 was first written on the non-affine ENERGY FRACTION
    alone, and --smoke passed it at 30 training steps: an untrained model's
    displacement is ~90% non-affine because random fields are, while ground
    truth is only 8%.  High non-affine energy is not parallax unless it is the
    RIGHT non-affine field.  Eleventh mis-specified gate in this project, and
    the first caught by a smoke run before shipping."""
    a, b = a_field.ravel(), b_field.ravel()
    den = float((a @ a) * (b @ b))
    g = float((a @ b) ** 2 / den) if den > 1e-30 else 0.0
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        ap = a_field[rng.permutation(len(a_field))].ravel()
        dn = float((ap @ ap) * (b @ b))
        nulls.append((ap @ b) ** 2 / dn if dn > 1e-30 else 0.0)
    return g, float(np.median(nulls))


def depth_corr(disp, depth):
    """Diagnostic only, NOT a gate: corr(|displacement|, 1/depth).

    Its sign is scene-dependent -- the affine fit absorbs the near-field motion,
    so residual magnitude can anti-correlate with inverse depth (measured -0.46
    on ground truth here).  Reported for interest; W5 uses the energy fraction,
    which has no such ambiguity."""
    m = np.isfinite(depth) & (depth > 1e-6)
    if m.sum() < 10:
        return float("nan")
    mag = np.linalg.norm(disp[m], axis=1)
    inv = 1.0 / depth[m]
    if mag.std() < 1e-12 or inv.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(mag, inv)[0, 1])


def sample_at(field, px, py, S):
    ix = np.clip(np.round(px).astype(int), 0, S - 1)
    iy = np.clip(np.round(py).astype(int), 0, S - 1)
    return field[iy, ix]


def flow_field(pose, delta, S):
    """DENSE ground-truth flow for one orbit step, plus depth and hit mask.

    Dense on purpose.  The first version sampled ground truth at 400 scattered
    points and then looked it up at the packet positions -- which almost never
    coincide, so the evaluation ran on 1-4 packets and every controllability
    number came back NaN.  Caught by --smoke, which is what --smoke is for."""
    _, dep, X, hit = render(pose, S)
    p1 = np.asarray(pose, float).copy()
    p1[0] += delta
    P3 = X.reshape(-1, 3)
    x0, y0, _ = project(pose, P3, S)
    x1, y1, _ = project(p1, P3, S)
    flow = np.stack([(x1 - x0) / S, (y1 - y0) / S], 1).reshape(S, S, 2)
    return flow, np.where(hit, dep, np.inf), hit


def oracle_row(ev_poses, S, n_pts=240, seed=0):
    """The instrument's positive control: real 3D points, real camera path."""
    mid = len(ev_poses) // 2
    _, _, X, hit = render(ev_poses[mid], S)
    pts = X[hit]
    if len(pts) > n_pts:
        pts = pts[np.random.default_rng(seed).choice(len(pts), n_pts, replace=False)]
    P = np.stack([np.stack(project(p, pts, S)[:2], 1) / S for p in ev_poses])
    r = rank_stats(P)
    r["mob"] = mobility(P)
    return r, pts


# ===========================================================================
# MODEL -- a minimal constant-Q Gabor splat autoencoder (torch)
# ===========================================================================

def build_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class Splat:
    """Renderer constants.  Packet k keeps its identity across every latent."""

    def __init__(self, torch, N, S, octaves=4, q=0.62,
                 sig_lo=0.020, sig_hi=0.26, device="cpu"):
        self.N, self.S, self.q, self.torch = N, S, q, torch
        side = int(math.ceil(math.sqrt(N)))
        k = np.arange(N)
        ax, ay = ((k % side) + 0.5) / side, ((k // side) + 0.5) / side
        self.anchor = torch.tensor(
            np.stack([np.log(ax / (1 - ax)), np.log(ay / (1 - ay))], 1),
            dtype=torch.float32, device=device)
        band = k % octaves
        self.slo = torch.tensor(sig_lo * (sig_hi / sig_lo) ** (band / octaves),
                                dtype=torch.float32, device=device)
        self.shi = torch.tensor(sig_lo * (sig_hi / sig_lo) ** ((band + 1) / octaves),
                                dtype=torch.float32, device=device)
        g = (np.arange(S) + 0.5) / S
        Y, X = np.meshgrid(g, g, indexing="ij")
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=device)

    def activate(self, raw):
        """raw (B,N,6) -> params.  Positions are FREE in [0,1], anchored only
        by initialisation -- nothing here restricts them to an affine family."""
        t = self.torch
        sg = self.slo * (self.shi / self.slo) ** t.sigmoid(raw[..., 2])
        return dict(cx=t.sigmoid(raw[..., 0] + self.anchor[:, 0]),
                    cy=t.sigmoid(raw[..., 1] + self.anchor[:, 1]),
                    sigma=sg, freq=self.q / sg, amp=raw[..., 3],
                    theta=raw[..., 4], phase=raw[..., 5])

    def render(self, p, chunk=32):
        t = self.torch
        u = t.zeros(p["cx"].shape[0], self.S, self.S, device=p["cx"].device)
        for i in range(0, self.N, chunk):
            sl = slice(i, i + chunk)
            dx = self.X[None, None] - p["cx"][:, sl, None, None]
            dy = self.Y[None, None] - p["cy"][:, sl, None, None]
            sg = p["sigma"][:, sl, None, None]
            th = p["theta"][:, sl, None, None]
            env = t.exp(-(dx ** 2 + dy ** 2) / (2 * sg ** 2))
            car = t.cos(2 * math.pi * p["freq"][:, sl, None, None]
                        * (dx * t.cos(th) + dy * t.sin(th))
                        + p["phase"][:, sl, None, None])
            u = u + (p["amp"][:, sl, None, None] * env * car).sum(1)
        return t.sigmoid(u)


def make_models(torch, nn, zdim, N, adim, device):
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(
                nn.Conv2d(1, 24, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(),
                nn.Conv2d(48, 96, 4, 2, 1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(4))
            self.f = nn.Linear(96 * 16, zdim)

        def forward(self, x):
            return self.f(self.c(x).flatten(1))

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = nn.Sequential(nn.Linear(zdim, 256), nn.SiLU(),
                                   nn.Linear(256, 256), nn.SiLU(),
                                   nn.Linear(256, N * 6))
            self.m[-1].weight.data *= 0.1
            self.m[-1].bias.data.zero_()

        def forward(self, z):
            return self.m(z).view(-1, N, 6)

    class Trans(nn.Module):
        """z_{t+1} = z_t + T(z_t, a_t).  adim = 0 for arm B."""

        def __init__(self):
            super().__init__()
            self.adim = adim
            self.m = nn.Sequential(nn.Linear(zdim + adim, 128), nn.SiLU(),
                                   nn.Linear(128, 128), nn.SiLU(),
                                   nn.Linear(128, zdim))
            self.m[-1].weight.data *= 0.1
            self.m[-1].bias.data.zero_()

        def forward(self, z, a=None):
            return z + self.m(z if self.adim == 0 else torch.cat([z, a], 1))

    return Enc().to(device), Dec().to(device), Trans().to(device)


# ===========================================================================
# TRAINING
# ===========================================================================

def train_arm(arm, cfg, frames, actions, log=True):
    """Train one arm, retrying at a lower learning rate if it diverges.

    WHY THIS EXISTS.  Two full runs were destroyed by divergence before this
    guard was written -- first the shuffled-action arm, then, on the very next
    run, plain arm A on seed 1 (loss 0.113 -> 0.361 -> 0.368 -> 0.353 -> 0.352).
    That second failure falsified my own explanation of the first: it is not
    that a wrong action makes the target unlearnable, it is that this renderer
    is unstable at lr 2e-3.  Amplitudes are unbounded and the finest packets
    carry a 31 cycle/canvas carrier, so one bad step saturates the sigmoid
    everywhere and the gradient dies.  A collapsed model then averages into the
    arm mean and blows W0 on a spread that has nothing to do with the question.

    THE FAILURE THRESHOLD IS NOT INVENTED, AND IT IS TWO-SIDED.  A model that
    outputs the dataset mean scores exactly the data variance, so a run that
    ends above that has learned nothing.  But undertraining looks the same by
    that test alone, and a 30-step smoke run would be wrongly binned as
    diverged.  So a run is FAILED only if it ends both above the data variance
    AND no better than where it started -- which is divergence, not slowness."""
    torch, nn = build_torch()
    dev = cfg["device"]
    var = float(np.var(frames))
    adim = 3 if arm in ("C", "C0", "Cx", "D") else 0
    Xd = torch.tensor(frames, dtype=torch.float32, device=dev)[:, None]
    Ad = torch.tensor(actions, dtype=torch.float32, device=dev)
    n = len(frames) - 1

    def attempt(lr, tag):
        torch.manual_seed(cfg["seed"])
        enc, dec, tr = make_models(torch, nn, cfg["zdim"], cfg["packets"], adim, dev)
        sp = Splat(torch, cfg["packets"], cfg["size"], device=dev)
        params = list(enc.parameters()) + list(dec.parameters())
        if arm != "A":
            params += list(tr.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["steps"])
        rng = np.random.default_rng(cfg["seed"] + 991)
        tail, t0 = [], time.time()
        for step in range(cfg["steps"]):
            idx = torch.tensor(rng.integers(0, n, cfg["batch"]), device=dev)
            x0, x1, a = Xd[idx], Xd[idx + 1], Ad[idx]
            if arm == "C0":        # capacity control: the channel carries nothing
                a = torch.zeros_like(a)
            elif arm == "Cx":      # the SHUFFLED arm -- see the docstring
                a = a[torch.randperm(len(a), device=dev)]
            z = enc(x0)
            raw0 = dec(z)
            loss = ((sp.render(sp.activate(raw0)) - x0[:, 0]) ** 2).mean()
            if arm != "A":
                z1 = tr(z) if adim == 0 else tr(z, a)
                raw1 = dec(z1)
                loss = loss + ((sp.render(sp.activate(raw1)) - x1[:, 0]) ** 2).mean()
                if arm == "D":   # force the motion through POSITIONS, not appearance
                    loss = loss + cfg["coefpen"] * ((raw1[..., 2:] - raw0[..., 2:])
                                                    ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            sched.step()
            tail.append(float(loss.detach()) / (1.0 if arm == "A" else 2.0))
            if log and step % max(cfg["steps"] // 5, 1) == 0:
                print(f"    [{arm:>2}{tag}] step {step:>6}  loss {tail[-1]:.5f}"
                      f"  {time.time()-t0:5.0f}s", flush=True)
        return (enc, dec, sp, float(np.mean(tail[-50:])),
                float(np.mean(tail[:20])))

    lr = cfg["lr"]
    for k in range(cfg["retries"] + 1):
        enc, dec, sp, tail, start = attempt(lr, "" if k == 0 else f"r{k}")
        if not (tail >= var and tail >= start):
            return enc, dec, sp, tail, k
        if k < cfg["retries"]:
            lr /= 3.0
            print(f"    [{arm:>2}] DIVERGED (tail {tail:.4f} >= data variance "
                  f"{var:.4f} and >= its own start {start:.4f}); retrying at "
                  f"lr {lr:.1e}", flush=True)
    print(f"    [{arm:>2}] STILL DIVERGED after {cfg['retries']} retries -- this "
          f"run is marked FAILED and excluded from the arm mean", flush=True)
    return enc, dec, sp, tail, -1


# ===========================================================================
# EVALUATION
# ===========================================================================

def eval_arm(enc, dec, sp, cfg, ev_frames, ev_poses, tr_frames, tr_poses):
    torch, _ = build_torch()
    dev, S = cfg["device"], cfg["size"]

    def latents(fr):
        with torch.no_grad():
            return enc(torch.tensor(fr, dtype=torch.float32,
                                    device=dev)[:, None]).cpu().numpy()

    def packets(z):
        with torch.no_grad():
            p = sp.activate(dec(torch.tensor(z, dtype=torch.float32, device=dev)))
            pos = torch.stack([p["cx"], p["cy"]], -1).cpu().numpy()
            coef = torch.stack([p["amp"], p["theta"], p["phase"],
                                p["sigma"]], -1).cpu().numpy()
            img = sp.render(p).cpu().numpy()
        return pos, coef.reshape(len(z), -1), img

    Z = latents(ev_frames)
    P, C, R = packets(Z)
    mse = float(((R - ev_frames) ** 2).mean())
    out = dict(psnr=10 * math.log10(1.0 / max(mse, 1e-12)),
               mob=mobility(P), **rank_stats(P))
    out["z_dof"], out["c_dof"] = dof(Z), dof(C)
    # did the coefficient penalty actually bite?  RMS temporal change of each
    # channel across the held-out sequence.  Without this, W6 [K] cannot be told
    # apart from "the penalty was too weak to do anything".
    out["dcoef"] = float(np.sqrt((np.diff(C, axis=0) ** 2).mean()))
    out["dpos"] = float(np.sqrt((np.diff(P, axis=0) ** 2).mean()))

    # pose -> latent, fitted on the TRAINING trajectory (all 3 DOF vary there)
    Zt = latents(tr_frames)
    A = np.concatenate([tr_poses, np.ones((len(tr_poses), 1))], 1)
    B, *_ = np.linalg.lstsq(A, Zt, rcond=None)
    out["pose_r2"] = float(1.0 - ((Zt - A @ B) ** 2).sum()
                           / max(((Zt - Zt.mean(0)) ** 2).sum(), 1e-12))

    # G3 / G4 at the middle of the held-out orbit
    mid = len(ev_poses) // 2
    delta = 0.16
    Pm, _, _ = packets(np.stack([Z[mid], Z[mid] + delta * B[0]]))
    dmodel = Pm[1] - Pm[0]
    flow, depth, hitm = flow_field(ev_poses[mid], delta, S)
    ppx, ppy = Pm[0][:, 0] * S, Pm[0][:, 1] * S
    dtrue = sample_at(flow, ppx, ppy, S)
    dpt = sample_at(depth, ppx, ppy, S)
    keep = sample_at(hitm, ppx, ppy, S) & np.isfinite(dpt)
    out["npk"] = int(keep.sum())
    if keep.sum() >= 12:
        pp = Pm[0][keep]
        res_t, frac_t = affine_residual(dtrue[keep], pp)
        res_m, frac_m = affine_residual(dmodel[keep], pp)
        out["g3"], out["g3null"] = dir_fidelity(res_m, res_t)
        out["g5"], out["g5null"] = field_match(res_m, res_t)
        out["nonaff"], out["nonaff_oracle"] = frac_m, frac_t
        out["dcorr"] = depth_corr(res_m, dpt[keep])
    else:
        for k in ("g3", "g3null", "g5", "g5null", "nonaff", "nonaff_oracle",
                  "dcorr"):
            out[k] = float("nan")
    return out


# ===========================================================================
# REPORTING
# ===========================================================================

def print_row(name, r):
    s = " ".join(f"{x:7.3f}" for x in r["s"][:5])
    print(f"  {name:<10} {s}   eff {r['eff']:5.3f}  s3/s2 {r['r32']:.4f}"
          f"  s4/s3 {r['r43']:.4f}  top2 {r['top2']*100:5.2f}%")


def verdict(tag, ok, msg):
    print(f"  {tag} [{'V' if ok else 'K'}]  {msg}")
    return bool(ok)


def run(cfg):
    print(f"scene: sphere + cube + checkered ground, {cfg['size']}px, "
          f"fov {FOV:.2f} rad, dist ~{ORBIT_DST}  (near-orthographic on purpose)")
    tr_poses = train_traj(cfg["frames"], cfg["seed"])
    ev_poses = orbit_traj(cfg["evframes"])
    t0 = time.time()
    tr_frames = np.stack([render(p, cfg["size"])[0] for p in tr_poses])
    ev_frames = np.stack([render(p, cfg["size"])[0] for p in ev_poses])
    act = np.diff(tr_poses, axis=0)
    act = np.concatenate([act, act[-1:]], 0)
    act = act / (act.std(0, keepdims=True) + 1e-9)
    print(f"training footage {len(tr_poses)} frames (3-DOF walk), held-out "
          f"{len(ev_poses)}-frame pure orbit; rendered in {time.time()-t0:.1f}s\n")

    orc, _ = oracle_row(ev_poses, cfg["size"])
    print("ORACLE -- true 3D surface points through the true camera path")
    print_row("oracle", orc)
    w1 = verdict("W1", orc["r32"] >= 0.05 and orc["r43"] <= 0.25,
                 f"instrument shows a rank-3 cliff on ground truth "
                 f"(s3/s2 {orc['r32']:.4f}, s4/s3 {orc['r43']:.4f})")
    if not w1:
        print("\n  VOID: the rank instrument cannot resolve depth even on ground"
              " truth here.  Widen the orbit or narrow the fov before believing"
              " any model row below.\n")

    seeds = [cfg["seed"] + i for i in range(cfg["seeds"])]
    per, res, fails = {a: [] for a in cfg["arms"]}, {}, []
    for sd in seeds:
        for arm in cfg["arms"]:
            print(f"  training arm {arm}  seed {sd} ...", flush=True)
            c = dict(cfg); c["seed"] = sd
            enc, dec, sp, tail, tries = train_arm(arm, c, tr_frames, act)
            if tries < 0:
                print(f"  arm {arm} seed {sd} FAILED to train -- excluded\n")
                fails.append((arm, sd))
                continue
            r = eval_arm(enc, dec, sp, c, ev_frames, ev_poses, tr_frames, tr_poses)
            r["tail"], r["tries"] = tail, tries
            per[arm].append(r)
            print()
    if fails:
        print("FAILED RUNS (excluded from every number below):")
        for a, sd in fails:
            print(f"  arm {a} seed {sd}")
        print()
    cfg["arms"] = [a for a in cfg["arms"] if per[a]]
    SCAL = ("psnr", "tail", "mob", "eff", "r32", "r43", "top2", "pose_r2",
            "g3", "g3null", "g5", "g5null", "nonaff", "nonaff_oracle", "dcorr",
            "dcoef", "dpos")
    for arm in cfg["arms"]:
        r = dict(per[arm][0])
        for k in SCAL:
            r[k] = float(np.nanmean([q[k] for q in per[arm]]))
        r["s"] = np.mean([q["s"] for q in per[arm]], axis=0)
        r["z_dof"] = (float(np.mean([q["z_dof"][0] for q in per[arm]])),
                      int(np.median([q["z_dof"][1] for q in per[arm]])), 0)
        r["c_dof"] = (float(np.mean([q["c_dof"][0] for q in per[arm]])), 0, 0)
        r["npk"] = int(np.mean([q["npk"] for q in per[arm]]))
        res[arm] = r
    if len(seeds) > 1:
        print("PER-SEED SPREAD  (gates below run on the mean)")
        print(f"  {'arm':<5} " + " ".join(f"{'s3/s2':>8}" for _ in seeds)
              + "   " + " ".join(f"{'G3':>7}" for _ in seeds))
        for arm in cfg["arms"]:
            print(f"  {arm:<5} "
                  + " ".join(f"{q['r32']:>8.4f}" for q in per[arm]) + "   "
                  + " ".join(f"{q['g3']:>7.3f}" for q in per[arm]))
        print()

    print("MEASUREMENT MATRIX  (per-frame centred packet positions, held-out orbit)")
    print_row("oracle", orc)
    for a in cfg["arms"]:
        print_row(f"arm {a}", res[a])
    print()

    print("CHANNEL BUDGET and FIT")
    print(f"  {'arm':<5} {'PSNR':>6} {'loss/term':>9} {'mobility':>9} {'z eff':>7} "
          f"{'95%':>4} {'coef eff':>9} {'pose R2':>8} {'d.coef':>8} {'d.pos':>8}")
    for a in cfg["arms"]:
        r = res[a]
        print(f"  {a:<5} {r['psnr']:>6.2f} {r['tail']:>9.5f} {r['mob']:>9.3f} "
              f"{r['z_dof'][0]:>7.3f} {r['z_dof'][1]:>4d} {r['c_dof'][0]:>9.3f} "
              f"{r['pose_r2']:>8.4f} {r['dcoef']:>8.4f} {r['dpos']:>8.4f}")
    print()

    print("CONTROLLABILITY  (move the latent one orbit step -- does it go the "
          "right way?)")
    print(f"  {'arm':<5} {'G3 cos':>7} {'null':>7} {'G5 match':>9} {'null':>7} "
          f"{'nonaff':>8} {'oracle':>8} {'dcorr':>7} {'pkts':>5}")
    for a in cfg["arms"]:
        r = res[a]
        print(f"  {a:<5} {r['g3']:>7.3f} {r['g3null']:>7.3f} {r['g5']:>9.4f} "
              f"{r['g5null']:>7.4f} {r['nonaff']:>8.4f} {r['nonaff_oracle']:>8.4f} "
              f"{r['dcorr']:>7.3f} {r['npk']:>5d}")
    print("  (all computed on the AFFINE RESIDUAL.  A HIGH nonaff with a low G5"
          " is noise, not parallax --")
    print("   an untrained model scores ~0.90 nonaff against ground truth's"
          " ~0.08.  dcorr is a diagnostic.)")
    if cfg["arms"] and min(res[a]["npk"] for a in cfg["arms"]) < 12:
        print("  NOTE: fewer than 12 packets landed on the scene rather than the"
              " sky, so the controllability row is NaN.  Raise --packets or"
              " --size; this is a sampling floor, not a result.")
    print()

    print("GATES")
    if not cfg["arms"]:
        print("  every arm failed to train -- nothing to gate")
        return res
    ps = [res[a]["psnr"] for a in cfg["arms"]]
    w0 = verdict("W0", max(ps) - min(ps) <= 1.5,
                 f"arms comparably fitted (PSNR spread {max(ps)-min(ps):.2f} dB)")
    if not w0:
        print("       arms are NOT comparably fitted -- W2..W5 are confounded by"
              " fit quality and must be read as VOID, not as results.")
    if "C" in res and "A" in res:
        C, A = res["C"], res["A"]
        verdict("W2", C["r32"] >= 0.5 * orc["r32"] and C["r43"] <= 2 * orc["r43"]
                and C["r32"] >= 3 * A["r32"],
                f"arm C has a rank-3 cliff and beats A  (C {C['r32']:.4f}/"
                f"{C['r43']:.4f}, oracle {orc['r32']:.4f}/{orc['r43']:.4f}, "
                f"A {A['r32']:.4f})")
        verdict("W4", C["g3"] >= 0.40 and C["g3"] >= 2 * abs(C["g3null"])
                and C["g3"] >= A["g3"] + 0.20,
                f"C's latent step matches the true parallax field  "
                f"(cos {C['g3']:.3f}, null {C['g3null']:+.3f}, A {A['g3']:.3f})")
        verdict("W5", C["g5"] >= 0.25 and C["g5"] >= 5 * max(C["g5null"], 1e-6),
                f"C's non-affine motion IS the true parallax field  "
                f"(G5 {C['g5']:.4f}, null {C['g5null']:.4f}, "
                f"nonaff {C['nonaff']:.3f} vs oracle {C['nonaff_oracle']:.3f})")
    if "D" in res:
        D = res["D"]
        base = res.get("C", res.get("A"))
        verdict("W6", D["r32"] >= 0.5 * orc["r32"] and D["r43"] <= 2 * orc["r43"]
                and D["r32"] >= 3 * base["r32"],
                f"forcing motion into POSITIONS produces geometry  "
                f"(D {D['r32']:.4f}/{D['r43']:.4f} at {D['psnr']:.2f} dB; "
                f"baseline {base['r32']:.4f}).  d.coef {D['dcoef']:.4f} vs "
                f"{base['dcoef']:.4f} -- if these are equal the penalty never "
                f"bit and W6 is UNTESTED, not falsified")
    if "C" in res and "C0" in res:
        verdict("W3", res["C"]["r32"] >= 2 * res["C0"]["r32"],
                f"the ACTION did it, not the extra capacity  (C "
                f"{res['C']['r32']:.4f} vs zero-action C0 "
                f"{res['C0']['r32']:.4f})")
    print("\n  Read W2..W5 only if W0 and W1 both passed.")
    return res


# ===========================================================================
# SELFTEST -- every instrument two-sided, with a planted known answer
# ===========================================================================

def selftest():
    ok = True

    def chk(n, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}  {d}")

    print("SELFTEST")
    S = 64
    rng = np.random.default_rng(0)
    pose = np.array([0.3, ORBIT_HGT, ORBIT_DST])
    img, dep, X, hit = render(pose, S)

    chk("T1 image has structure", img.std() > 0.08, f"std {img.std():.3f}")
    chk("T1 a useful fraction of rays hit the scene", hit.mean() > 0.25,
        f"{hit.mean():.2f}")
    px, py, _ = project(pose, SPHERE_C[None], S)
    sd = dep[int(py[0]), int(px[0])]
    fd = np.nanmedian(np.where(hit, dep, np.nan))
    chk("T1 the sphere is nearer than the median surface", sd < fd,
        f"{sd:.2f} vs {fd:.2f}")

    g = np.arange(S) + 0.5
    PX, PY = np.meshgrid(g, g)
    rx, ry, _ = project(pose, X.reshape(-1, 3), S)
    m = hit.ravel()
    err = max(np.abs(rx[m] - PX.ravel()[m]).max(), np.abs(ry[m] - PY.ravel()[m]).max())
    chk("T2 project(render hits) returns the original pixels", err < 0.05,
        f"max err {err:.4f} px")

    ev = orbit_traj(24)
    orc, pts = oracle_row(ev, S)
    chk("T3 planted TRUE 3D shows a rank-3 cliff",
        orc["r32"] >= 0.05 and orc["r43"] <= 0.25,
        f"s3/s2 {orc['r32']:.4f}  s4/s3 {orc['r43']:.4f}  "
        f"s {np.array2string(orc['s'][:4], precision=2)}")
    b2 = np.stack(project(ev[12], pts, S)[:2], 1) / S
    aff = np.stack([b2 @ (np.eye(2) + rng.normal(0, 0.25, (2, 2))).T
                    + rng.normal(0, 0.3, 2) for _ in range(24)])
    ra = rank_stats(aff)
    chk("T3 planted 2D AFFINE morph reads rank 2, no cliff",
        ra["r32"] < 0.01 and ra["eff"] < 2.2,
        f"eff {ra['eff']:.3f}  s3/s2 {ra['r32']:.2e}")
    chk("T3 the participation ratio would have MISREAD the true 3D case",
        orc["eff"] < 2.5, f"oracle eff {orc['eff']:.3f} -- read the cliff, not this")

    fl, dpm, hm = flow_field(ev[12], 0.16, S)
    idx = rng.choice(np.where(hm.ravel())[0], 400, replace=False)
    gg = (np.arange(S) + 0.5) / S
    GY, GX = np.meshgrid(gg, gg, indexing="ij")
    pos0 = np.stack([GX.ravel()[idx], GY.ravel()[idx]], 1)
    tflow = fl.reshape(-1, 2)[idx]
    res_t, frac_t = affine_residual(tflow, pos0)
    chk("T4 true 3D flow has non-affine content", frac_t > 0.05,
        f"residual energy {frac_t:.4f}")
    fr = []
    for _ in range(3):
        A = np.eye(2) + rng.normal(0, 0.05, (2, 2))
        fr.append(affine_residual(pos0 @ A.T + rng.normal(0, 0.02, 2) - pos0, pos0)[1])
    chk("T4 a planted affine field has NONE", max(fr) < 1e-6,
        f"max residual energy {max(fr):.2e}")

    c1, n1 = dir_fidelity(res_t, res_t)
    chk("T5 identical residual fields give cos 1", c1 > 0.99, f"cos {c1:.4f}")
    chk("T5 the shuffled null on residuals is near 0", abs(n1) < 0.30,
        f"null {n1:+.3f}")
    c2, _ = dir_fidelity(-res_t, res_t)
    chk("T5 a reversed field gives cos -1", c2 < -0.99, f"cos {c2:.4f}")
    g5, g5n = field_match(res_t, res_t)
    chk("T5 G5 field match is 1 for identical fields", g5 > 0.999, f"{g5:.4f}")
    chk("T5 G5 shuffled null is ~0", g5n < 0.10, f"{g5n:.4f}")
    noise = rng.normal(size=res_t.shape)
    gn, _ = field_match(noise, res_t)
    chk("T5 G5 rejects a random field", gn < 0.10, f"{gn:.4f}")
    _, fn = affine_residual(noise, pos0)
    chk("T5 REGRESSION: a random field is ~all non-affine, which is why W5 "
        "cannot gate on the energy fraction", fn > 0.90, f"nonaff {fn:.3f}")
    craw, nraw = dir_fidelity(tflow, tflow)
    chk("T5 REGRESSION: raw flow has a badly contaminated null, which is why "
        "G3 runs on residuals", abs(nraw) > 0.4, f"raw null {nraw:+.3f}")

    M3 = rng.normal(size=(60, 3)) @ rng.normal(size=(3, 40))
    chk("T6 planted rank-3 reads ~3", 2.5 < dof(M3)[0] < 3.5, f"{dof(M3)[0]:.3f}")
    M1 = np.outer(rng.normal(size=60), rng.normal(size=40))
    chk("T6 planted rank-1 reads ~1", dof(M1)[0] < 1.2, f"{dof(M1)[0]:.3f}")

    P = np.tile(rng.random((1, 60, 2)), (10, 1, 1))
    chk("T7 frozen packets read mobility 0", mobility(P) < 1e-9)
    chk("T7 moving packets read mobility > 0",
        mobility(P + np.linspace(0, 0.3, 10)[:, None, None]) > 1.0)

    try:
        torch, nn = build_torch()
        enc, dec, tr = make_models(torch, nn, 16, 24, 3, "cpu")
        sp = Splat(torch, 24, 32, device="cpu")
        with torch.no_grad():
            z = torch.zeros(2, 16)
            p = sp.activate(dec(z))
            im = sp.render(p)
            chk("T8 render shape and range", tuple(im.shape) == (2, 32, 32)
                and float(im.min()) >= 0 and float(im.max()) <= 1)
            chk("T8 packet centres live in [0,1]", float(p["cx"].min()) >= 0
                and float(p["cx"].max()) <= 1)
            chk("T8 the action input actually reaches the transition",
                float((tr(z, torch.ones(2, 3)) - tr(z, -torch.ones(2, 3))
                       ).abs().sum()) > 0)
            chk("T8 encoder maps an image to a latent",
                tuple(enc(torch.zeros(2, 1, 32, 32)).shape) == (2, 16))
        _, _, trC = make_models(torch, nn, 16, 24, 3, "cpu")
        _, _, trB = make_models(torch, nn, 16, 24, 0, "cpu")
        nC = sum(q.numel() for q in trC.parameters())
        nB = sum(q.numel() for q in trB.parameters())
        chk("T9 the zero-action control is capacity-matched to C, not to B",
            nC > nB, f"C {nC} params vs B {nB}")
        with torch.no_grad():
            z = torch.zeros(2, 16)
            same = float((trC(z, torch.zeros(2, 3)) - trC(z, torch.zeros(2, 3))
                          ).abs().sum())
        chk("T9 a zero action is deterministic (C0's task stays learnable)",
            same == 0.0, f"{same:.2e}")
    except ImportError:
        chk("T8 torch present", False, "pip install torch to run the experiment")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    p.add_argument("--run", action="store_true")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--packets", type=int, default=144)
    p.add_argument("--zdim", type=int, default=32)
    p.add_argument("--frames", type=int, default=320)
    p.add_argument("--evframes", type=int, default=24)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--arms", type=str, default="A,B,C,C0")
    p.add_argument("--seeds", type=int, default=1, help="repeat every arm N times")
    p.add_argument("--retries", type=int, default=2,
                   help="retries at lr/3 when a run diverges")
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip")
    p.add_argument("--coefpen", type=float, default=0.05,
                   help="arm D: penalty on temporal change of the coefficient "
                        "channel, forcing motion through packet POSITIONS")
    a = p.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.run or a.smoke):
        p.print_help()
        return

    dev = a.device
    if dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            sys.exit("torch is required for --run / --smoke")
    cfg = dict(size=a.size, packets=a.packets, zdim=a.zdim, frames=a.frames,
               evframes=a.evframes, steps=a.steps, batch=a.batch, lr=a.lr,
               seed=a.seed, device=dev, seeds=max(1, a.seeds),
               coefpen=a.coefpen, retries=a.retries, clip=a.clip,
               arms=[s for s in a.arms.split(",") if s])
    if a.smoke:
        cfg.update(size=32, packets=64, zdim=12, frames=60, evframes=12,
                   steps=30, batch=4)
        print("SMOKE: not an experiment, only a check that every path runs.\n")
    print(f"device {dev}\n")
    run(cfg)


if __name__ == "__main__":
    main()
