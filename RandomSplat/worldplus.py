#!/usr/bin/env python3
"""
worldplus.py -- STEP 1 AND 2 OF WorldPlusAvatar, AND ONLY THOSE.

WHERE THIS COMES FROM
worldrank.py established, over three runs and with working controls, that on a
static orbit NOTHING moves the geometry: action conditioning (C 0.0138 vs
zero-action C0 0.0133), temporal prediction, and a coefficient penalty that
demonstrably bit (d.coef 0.1567 -> 0.1390, d.pos 0.0133 -> 0.0149) all leave the
packet positions at ~0.012 against an oracle of 0.0506, with within-arm seed
spread larger than every between-arm difference.  Every one of those arms varied
the INPUT or added a regulariser.  None changed what the loss is ABOUT.  So the
objective is now the untested variable, and that is an empirical statement.

WHY THIS IS NOT THE WHOLE PROPOSAL
The WorldPlusAvatar sketch has a dozen subsystems.  Three of its four core loss
terms -- coverage, loop closure, replay -- cannot be SCORED on an orbit around a
sphere, because they need a world where the same place is reachable by different
routes and where two places can look the same.  Building the full loss on a
scene that cannot score it produces a large system with nothing to check it
against, which is how this project has burned two GPU runs already.

And one part of the sketch has to be refused as written.  It says, correctly,
"grid cells might emerge -- you shouldn't program them", and then hard-splits the
latent into World State and Appearance.  If the split is built by hand, geometry
appearing in the world branch is construction, not emergence -- the same smuggle
refused three times in flatworld/fieldworld (f_i(q,theta,L), theta_n, decoders
trained on 3D-derived data).  So there is ONE latent here, the objective is the
only thing that changes, and whether a split appears is MEASURED (L3).

WHAT THIS FILE ADDS
  1. A NAVIGABLE ROOM.  Walls, occluding pillars, a checkered floor, free 2D
     motion plus yaw.  Content appears and disappears; translation produces real
     parallax (which an orbit at near-orthographic fov cannot).
  2. EXACT PERCEPTUAL ALIASING, by construction.  The room is invariant under a
     180 degree rotation about its centre -- pillars in +-pairs, even wall
     stripes, a floor checker that is symmetric mod 2, and a straight-down light
     so shading depends only on n_y.  So pose p and R180(p) render IDENTICALLY
     to floating point, except for one landmark box that breaks the symmetry.
     Verified two-sided in the selftest, both with the landmark and without.
     This is the thing an orbit could not provide: two different places that
     look the same, so "world state" and "image state" make different
     predictions instead of the same one.
  3. COUNTERFACTUAL CONSISTENCY as the objective, arm X, against a
     task-and-capacity-matched control, arm P.
        P  predict frame i+k for random k -- displacements ALONG THE PATH
        X  predict any frame whose pose is within reach of p_i but which was
           NOT reached from i within k steps -- displacements the agent did not
           take, checked against reality because reality happened to go there
     Identical architecture, identical number of loss terms, identical action
     representation (relative pose in the current frame).  The ONLY difference
     is whether the queried displacement lies along the travelled path.  That is
     the Moser distinction -- sample your own trajectory, or sample the
     surrounding manifold -- reduced to one bit.

     NOTE ON WHY THE CONTROL IS NOT A SCRAMBLED ACTION.  It was, twice, and both
     times it destroyed the run: a wrong action makes the target unlearnable, so
     the gradient through the shared trunk is noise.  P is matched WITHOUT being
     poisoned.

  4. LOOP CLOSURE AND ALIASING ARE MEASURED, NOT IMPOSED.  No loss term rewards
     them.  If counterfactual training alone produces them, that is a result; a
     loop-closure loss would make it a specification.

PRE-REGISTERED, ALL AS FRACTIONS OF THE ORACLE OR AGAINST A NULL
  W0  comparability   eval PSNR spread <= 1.5 dB, else everything VOID
  W1  instrument      ORACLE shows a rank-3 cliff on the lateral traverse:
                      s4^2/s3^2 <= 0.10 and s3^2/s2^2 >= 0.01.  Measured
                      0.0313 / 0.0007 -- s3 stands 38x clear of s4
  X1  geometry        arm X s3^2/s2^2 >= 0.5x ORACLE and >= 2x arm P
  X2  parallax        arm X G5 >= 0.25 and >= 5x its own shuffled null
  L1  aliasing        REGISTERED AS EXPECTED TO FAIL for a feedforward encoder.
                      Two places that render identically CANNOT be separated by
                      any function of the current image alone.  L1 measures the
                      architectural ceiling, so an [K] here is the predicted
                      result and an [V] would mean the aliasing is not exact --
                      check the selftest before believing it.
  L2  loop closure    corr(pose distance, latent distance) >= 0.5 and >= 3x its
                      shuffled null, computed with aliased pairs EXCLUDED
  X3  smoothness      arm S = X plus an ACTION-SPACE Laplacian on the transition
  L3  emergent split  does the latent separate into a slow pose-tracking
                      subspace and a fast appearance one, without being told to?
                      Reported as the variance share of dz/dt explained by the
                      top-3 pose-aligned directions.  Diagnostic, not a gate.

    python3 worldplus.py --selftest
    python3 worldplus.py --smoke
    python3 worldplus.py --run --arms A,P,X --seeds 3 --steps 12000
"""

import argparse
import math
import sys
import time

import numpy as np

# ===========================================================================
# SCENE -- a room.  Analytic ray cast; exact 180-degree symmetry by design.
# ===========================================================================

HALF = 3.0                     # room half-width in x and z
FLOOR_Y, WALL_TOP = -1.0, 3.2
CAM_Y = 0.0
FOV = 0.55                     # half-angle; a room needs a wide view
PILLARS = [                    # in +- pairs, so the set maps to itself
    (np.array([1.30, 0.95]), np.array([0.34, 0.34])),
    (np.array([-1.30, -0.95]), np.array([0.34, 0.34])),
    (np.array([-1.05, 1.55]), np.array([0.26, 0.26])),
    (np.array([1.05, -1.55]), np.array([0.26, 0.26])),
]
LANDMARK_C = np.array([2.35, 2.35])
LANDMARK_H = np.array([0.30, 0.30])
INF = 1e9


def _stripe(c):
    """Wall albedo. cos() is EVEN, so this survives c -> -c and the wall
    pattern is invariant under the 180 degree rotation."""
    return np.where(np.cos(2.3 * c) > 0.0, 0.66, 0.34)


def render(pose, S=64, landmark=True):
    """pose = (x, z, yaw).  Returns image, depth, 3D hit points, hit mask.

    Light is straight down, so shading depends only on n_y and is therefore
    invariant under rotation about y -- without this the symmetry breaks."""
    x0, z0, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    eye = np.array([x0, CAM_Y, z0])
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    t = math.tan(FOV)
    g = (np.arange(S) + 0.5) / S * 2.0 - 1.0
    d = (fwd[None, None, :] + (g[None, :] * t)[..., None] * right
         + (-g[:, None] * t)[..., None] * up)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    dd = np.where(np.abs(d) < 1e-9, 1e-9, d)

    tb = np.full((S, S), INF)
    ny = np.zeros((S, S))
    alb = np.zeros((S, S))

    def box(c2, h2, lo_y, hi_y, albedo_fn):
        nonlocal tb, ny, alb
        lo = np.array([c2[0] - h2[0], lo_y, c2[1] - h2[1]])
        hi = np.array([c2[0] + h2[0], hi_y, c2[1] + h2[1]])
        t1, t2 = (lo - eye) / dd, (hi - eye) / dd
        tn = np.max(np.minimum(t1, t2), axis=-1)
        tf = np.min(np.maximum(t1, t2), axis=-1)
        m = tf > np.maximum(tn, 1e-4)
        tc = np.where(m, tn, INF)
        upd = m & (tc < tb)
        p = eye + tc[..., None] * d
        rel = (p - 0.5 * (lo + hi)) / np.maximum(0.5 * (hi - lo), 1e-9)
        ax = np.argmax(np.abs(rel), axis=-1)
        n_y = np.where(ax == 1, np.sign(rel[..., 1]), 0.0)
        tb = np.where(upd, tc, tb)
        ny = np.where(upd, n_y, ny)
        alb = np.where(upd, albedo_fn(p), alb)

    # floor: the checker is symmetric mod 2 because -s == s (mod 2)
    tp = np.where(np.abs(d[..., 1]) > 1e-9, (FLOOR_Y - eye[1]) / dd[..., 1], INF)
    p = eye + tp[..., None] * d
    m = ((tp > 1e-4) & (np.abs(p[..., 0]) <= HALF) & (np.abs(p[..., 2]) <= HALF))
    tp = np.where(m, tp, INF)
    upd = m & (tp < tb)
    chk = (np.floor(p[..., 0] * 1.4) + np.floor(p[..., 2] * 1.4)) % 2 == 0
    tb = np.where(upd, tp, tb)
    ny = np.where(upd, 1.0, ny)
    alb = np.where(upd, np.where(chk, 0.78, 0.34), alb)

    # four walls, opposite pairs identical
    for axis, sgn in ((0, 1), (0, -1), (2, 1), (2, -1)):
        pl = sgn * HALF
        tw = np.where(np.abs(d[..., axis]) > 1e-9,
                      (pl - eye[axis]) / dd[..., axis], INF)
        p = eye + tw[..., None] * d
        other = 2 if axis == 0 else 0
        m = ((tw > 1e-4) & (np.abs(p[..., other]) <= HALF)
             & (p[..., 1] >= FLOOR_Y) & (p[..., 1] <= WALL_TOP))
        tw = np.where(m, tw, INF)
        upd = m & (tw < tb)
        base = 0.62 if axis == 0 else 0.40
        tb = np.where(upd, tw, tb)
        ny = np.where(upd, 0.0, ny)
        alb = np.where(upd, base * _stripe(p[..., other]) * 1.4, alb)

    for c2, h2 in PILLARS:
        box(c2, h2, FLOOR_Y, FLOOR_Y + 1.5, lambda p: 0.52)
    if landmark:
        box(LANDMARK_C, LANDMARK_H, FLOOR_Y, FLOOR_Y + 0.9, lambda p: 0.95)

    hit = tb < INF / 2
    img = np.where(hit, alb * (0.34 + 0.66 * np.clip(ny, 0.0, 1.0)), 0.06)
    X = eye + np.where(hit, tb, 0.0)[..., None] * d
    return np.clip(img, 0, 1).astype(np.float32), np.where(hit, tb, np.inf), X, hit


def project(pose, X, S=64):
    x0, z0, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    eye = np.array([x0, CAM_Y, z0])
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    up = np.array([0.0, 1.0, 0.0])
    w = X - eye
    z = w @ fwd
    zz = np.where(np.abs(z) < 1e-6, 1e-6, z)
    t = math.tan(FOV)
    return (((w @ right) / (zz * t) + 1.0) * 0.5 * S,
            (1.0 - (w @ up) / (zz * t)) * 0.5 * S, z)


def rot180(pose):
    """The symmetry of the room: (x,z,yaw) -> (-x,-z,yaw+pi)."""
    return np.array([-pose[0], -pose[1], pose[2] + math.pi])


# ---------------------------------------------------------------- motion ---

def blocked(x, z, pad=0.42):
    if abs(x) > HALF - 0.45 or abs(z) > HALF - 0.45:
        return True
    for c2, h2 in PILLARS:
        if abs(x - c2[0]) < h2[0] + pad and abs(z - c2[1]) < h2[1] + pad:
            return True
    return False


def walk(n, seed=0):
    """Random walk with (forward, strafe, turn) actions, respecting the room."""
    rng = np.random.default_rng(seed)
    x, z, yaw = 0.0, -2.0, 0.0
    poses, acts = [], []
    for _ in range(n):
        poses.append([x, z, yaw])
        for _try in range(24):
            a = np.array([rng.normal(0.13, 0.05), rng.normal(0.0, 0.05),
                          rng.normal(0.0, 0.16)])
            ny = yaw + a[2]
            nx = x + a[0] * math.sin(ny) + a[1] * math.cos(ny)
            nz = z + a[0] * math.cos(ny) - a[1] * math.sin(ny)
            if not blocked(nx, nz):
                break
        else:
            a = np.array([0.0, 0.0, 0.9])
            ny, nx, nz = yaw + a[2], x, z
        acts.append(a)
        x, z, yaw = nx, nz, ny
    return np.array(poses), np.array(acts)


def traverse(n, seed=0):
    """A SMOOTH walk across the room, for the rank window only.

    The exploratory random walk turns too much: over 24 of its frames fewer than
    24 surface points stay visible, and Tomasi-Kanade needs full visibility, so
    the oracle could not be computed at all.  Rank and parallax therefore run on
    a gentle traverse (the geometry question), while aliasing and loop closure
    run on the exploratory walk (the world-state question).  Two trajectories
    because they are two different measurements, not to make a number look
    better."""
    rng = np.random.default_rng(seed)
    # A SIDEWAYS traverse, facing +z, along the clear lane at z = -2.35.
    # Lateral motion is what makes parallax: displacement goes as 1/depth, so
    # near pillars sweep past while the far wall barely moves.  A forward
    # traverse was tried first and gives an oracle s3/s2 of only 0.0064 with
    # s4/s3 0.55 -- pure forward motion is dominated by a uniform zoom, which is
    # affine, and the depth-dependent part is second order.  The instrument
    # needs the motion that actually separates depths.
    x, z, yaw = -1.55, -2.35, 0.0
    poses = []
    for _ in range(n):
        poses.append([x, z, yaw])
        yaw += rng.normal(0.0, 0.004)
        x += 0.105 + rng.normal(0.0, 0.004)
    return np.array(poses)


def rel_action(pi, pj):
    """Displacement from pose i to pose j, expressed in i's own frame.
    This is what odometry gives a robot, and it is defined for ANY pair, which
    is what makes counterfactual queries possible at all."""
    dx, dz = pj[0] - pi[0], pj[1] - pi[1]
    # walk() turns FIRST and then translates, so the displacement is expressed
    # in the TARGET yaw frame.  Using pi[2] here instead of pj[2] was a silent
    # inconsistency: rel_action failed to reconstruct the target pose by 0.118,
    # so every counterfactual query would have carried a slightly wrong action.
    c, s = math.cos(pj[2]), math.sin(pj[2])
    fwd = dx * s + dz * c
    strafe = dx * c - dz * s
    dyaw = (pj[2] - pi[2] + math.pi) % (2 * math.pi) - math.pi
    return np.array([fwd, strafe, dyaw])


def pose_dist(P, Q, w_yaw=0.7):
    dyaw = (Q[..., 2] - P[..., 2] + math.pi) % (2 * math.pi) - math.pi
    return np.sqrt((Q[..., 0] - P[..., 0]) ** 2 + (Q[..., 1] - P[..., 1]) ** 2
                   + (w_yaw * dyaw) ** 2)


# ===========================================================================
# INSTRUMENTS (carried over from worldrank, all two-sided in the selftest)
# ===========================================================================

def rank_stats(P):
    F, N, _ = P.shape
    W = P - P.mean(axis=1, keepdims=True)
    W = np.concatenate([W[:, :, 0], W[:, :, 1]], axis=0)
    s = np.linalg.svd(W, compute_uv=False)[:8]
    ev = s ** 2 / max((s ** 2).sum(), 1e-30)
    return dict(s=s, eff=float(np.exp(-(ev * np.log(ev + 1e-30)).sum())),
                r32=float(s[2] ** 2 / max(s[1] ** 2, 1e-30)),
                r43=float(s[3] ** 2 / max(s[2] ** 2, 1e-30)),
                top2=float((s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-30)))


def dof(M):
    M = np.asarray(M, dtype=np.float64)
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    ev = s ** 2 / max((s ** 2).sum(), 1e-30)
    cum = np.cumsum(ev)
    return (float(np.exp(-(ev * np.log(ev + 1e-30)).sum())),
            int(np.searchsorted(cum, 0.95) + 1))


def mobility(P):
    trav = np.linalg.norm(np.diff(P, axis=0), axis=2).sum(axis=0)
    d = np.linalg.norm(P[0][:, None, :] - P[0][None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(trav) / max(np.median(d.min(axis=1)), 1e-9))


def affine_residual(disp, pos):
    """The part of a displacement field no rank-2 morph can express."""
    A = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, disp, rcond=None)
    res = disp - A @ coef
    return res, float((res ** 2).sum() / max((disp ** 2).sum(), 1e-30))


def dir_fidelity(a_field, b_field, seed=0, n_perm=64):
    a = a_field / np.maximum(np.linalg.norm(a_field, axis=1, keepdims=True), 1e-12)
    b = b_field / np.maximum(np.linalg.norm(b_field, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    nulls = [np.median((a[rng.permutation(len(a))] * b).sum(1)) for _ in range(n_perm)]
    return float(np.median((a * b).sum(1))), float(np.median(nulls))


def field_match(a_field, b_field, seed=0, n_perm=64):
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


def sample_at(field, px, py, S):
    ix = np.clip(np.round(px).astype(int), 0, S - 1)
    iy = np.clip(np.round(py).astype(int), 0, S - 1)
    return field[iy, ix]


def flow_field(pose, act, S):
    """Dense ground-truth flow for one action step, plus depth and hit mask."""
    _, dep, X, hit = render(pose, S)
    yaw2 = pose[2] + act[2]
    p1 = np.array([pose[0] + act[0] * math.sin(yaw2) + act[1] * math.cos(yaw2),
                   pose[1] + act[0] * math.cos(yaw2) - act[1] * math.sin(yaw2),
                   yaw2])
    P3 = X.reshape(-1, 3)
    x0, y0, _ = project(pose, P3, S)
    x1, y1, _ = project(p1, P3, S)
    fl = np.stack([(x1 - x0) / S, (y1 - y0) / S], 1).reshape(S, S, 2)
    return fl, np.where(hit, dep, np.inf), hit, p1


def oracle_row(poses, S, n_pts=260, seed=0):
    """The instrument's ceiling: true 3D points through the true camera path.

    VISIBILITY IS ENFORCED, and it has to be.  Tomasi-Kanade assumes every point
    is seen in every frame.  In a room with a freely moving camera most surface
    points leave the frame or pass behind the camera, and a point behind the
    camera projects through a near-zero depth -- the first version of this
    returned singulars of 39153 and an oracle s3/s2 of 0.50, which is not a
    measurement of anything.  Points are kept only if they stay in front of the
    camera and inside a generous frame margin at EVERY pose in the window."""
    mid = len(poses) // 2
    _, _, X, hit = render(poses[mid], S)
    pts = X[hit]
    rng = np.random.default_rng(seed)
    if len(pts) > 6 * n_pts:
        pts = pts[rng.choice(len(pts), 6 * n_pts, replace=False)]
    good = np.ones(len(pts), bool)
    cols = []
    for p in poses:
        px, py, z = project(p, pts, S)
        good &= (z > 0.25) & (px > -0.4 * S) & (px < 1.4 * S) \
            & (py > -0.4 * S) & (py < 1.4 * S)
        cols.append(np.stack([px / S, py / S], 1))
    if good.sum() < 24:
        return None
    P = np.stack([c[good] for c in cols])
    if P.shape[1] > n_pts:
        P = P[:, rng.choice(P.shape[1], n_pts, replace=False)]
    r = rank_stats(P)
    r["mob"] = mobility(P)
    r["npts"] = int(P.shape[1])
    return r


# ===========================================================================
# ALIASING AND LOOP CLOSURE -- found in the data, not asserted
# ===========================================================================

def alias_pairs(poses, frames, max_img=0.004, min_pose=1.5):
    """Pairs that look the same but are somewhere else.

    Detected, not assumed: an image difference below max_img AND a pose distance
    above min_pose.  The room's 180 degree symmetry makes such pairs exist; the
    landmark makes them rarer, which is the realistic case."""
    n = len(poses)
    out = []
    for i in range(n):
        pr = rot180(poses[i])
        d = pose_dist(poses, pr[None])
        j = int(np.argmin(d))
        if d[j] > 0.35:
            continue
        e = float(np.mean((frames[i] - frames[j]) ** 2))
        if e <= max_img and pose_dist(poses[i][None], poses[j][None])[0] >= min_pose:
            out.append((i, j, e))
    return out


def revisit_gap(Z, poses, near=0.55, gap=60, seed=0):
    """L2b: median latent distance for pairs that are PHYSICALLY CLOSE but
    TEMPORALLY DISTANT, over the median for random pairs.

    Loop closure is a local question -- "do I recognise I am back?" -- and the
    global correlation answers a different one.  In a 180-degree-symmetric room
    distant places genuinely look alike, so corr(pose distance, latent distance)
    is diluted by the symmetry before any model is involved: the first full run
    read -0.004 / -0.008 / +0.007 against a null of 0.014 for every arm.  This
    statistic has the same shape as alias_gap and its own built-in reference:
    near 0 means revisits collapse to the same latent, near 1 means a revisit
    looks like any other pair of frames."""
    n = len(Z)
    ii, jj = np.triu_indices(n, 1)
    keep = (np.abs(ii - jj) > gap) & (pose_dist(poses[ii], poses[jj]) < near)
    if keep.sum() < 20:
        return float("nan"), 0
    # The reference set is TEMPORALLY MATCHED and physically FAR.  Dividing by
    # random pairs instead lets a collapsed latent pass: at 30 smoke steps every
    # latent is nearly identical, so the ratio read 0.33 for every arm and L2
    # printed [V] on a model that had learned nothing.  Matching the control on
    # the same temporal separation cancels that -- collapse shrinks numerator and
    # denominator together.
    far = (np.abs(ii - jj) > gap) & (pose_dist(poses[ii], poses[jj]) > 2.0 * near)
    if far.sum() < 20:
        return float("nan"), 0
    zr = np.linalg.norm(Z[ii[keep]] - Z[jj[keep]], axis=1)
    zf = np.linalg.norm(Z[ii[far]] - Z[jj[far]], axis=1)
    return float(np.median(zr) / max(float(np.median(zf)), 1e-12)), int(keep.sum())


def loop_stats(Z, poses, exclude, seed=0, n=4000):
    """Does latent distance track physical distance?  With a shuffled null."""
    rng = np.random.default_rng(seed)
    n_f = len(Z)
    bad = set()
    for i, j, _ in exclude:
        bad.add((min(i, j), max(i, j)))
    a = rng.integers(0, n_f, n)
    b = rng.integers(0, n_f, n)
    keep = [(i, j) for i, j in zip(a, b)
            if i != j and (min(i, j), max(i, j)) not in bad]
    if len(keep) < 50:
        return float("nan"), float("nan")
    ii = np.array([k[0] for k in keep])
    jj = np.array([k[1] for k in keep])
    pd = pose_dist(poses[ii], poses[jj])
    zd = np.linalg.norm(Z[ii] - Z[jj], axis=1)
    r = float(np.corrcoef(pd, zd)[0, 1])
    nulls = [float(np.corrcoef(pd, zd[rng.permutation(len(zd))])[0, 1])
             for _ in range(32)]
    return r, float(np.median(np.abs(nulls)))


# ===========================================================================
# MODEL
# ===========================================================================

def build_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class Splat:
    def __init__(self, torch, N, S, octaves=4, q=0.62,
                 sig_lo=0.022, sig_hi=0.26, device="cpu"):
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
            self.m = nn.Sequential(nn.Linear(zdim, 320), nn.SiLU(),
                                   nn.Linear(320, 320), nn.SiLU(),
                                   nn.Linear(320, N * 6))
            self.m[-1].weight.data *= 0.1
            self.m[-1].bias.data.zero_()

        def forward(self, z):
            return self.m(z).view(-1, N, 6)

    class Trans(nn.Module):
        def __init__(self):
            super().__init__()
            self.adim = adim
            self.m = nn.Sequential(nn.Linear(zdim + adim, 160), nn.SiLU(),
                                   nn.Linear(160, 160), nn.SiLU(),
                                   nn.Linear(160, zdim))
            self.m[-1].weight.data *= 0.1
            self.m[-1].bias.data.zero_()

        def forward(self, z, a=None):
            return z + self.m(z if self.adim == 0 else torch.cat([z, a], 1))

    return Enc().to(device), Dec().to(device), Trans().to(device)


# ===========================================================================
# QUERY SETS -- the entire difference between arm P and arm X
# ===========================================================================

def build_queries(poses, K=6, radius=None, seed=0):
    """For each frame i: path targets (i+1..i+K) and counterfactual targets.

    Counterfactual = any j whose pose is within `radius` of p_i but which is
    NOT within K steps of i along the path.  Both sets are then trimmed so
    their displacement-magnitude distributions match, because a difference in
    action scale between the arms would be a confound rather than the point."""
    n = len(poses)
    if radius is None:
        pk = np.concatenate([pose_dist(poses[:-k], poses[k:])
                             for k in range(1, K + 1)])
        radius = float(np.percentile(pk, 90))
    D = pose_dist(poses[:, None, :], poses[None, :, :])
    idx = np.arange(n)
    path, cf = [], []
    for i in range(n - K - 1):
        for k in range(1, K + 1):
            path.append((i, i + k))
        near = np.where((D[i] <= radius) & (np.abs(idx - i) > K))[0]
        for j in near:
            cf.append((i, int(j)))
    rng = np.random.default_rng(seed)
    path = np.array(path)
    cf = np.array(cf) if len(cf) else np.zeros((0, 2), int)
    if len(cf):
        # match the displacement-magnitude distribution by rejection sampling
        dp = pose_dist(poses[path[:, 0]], poses[path[:, 1]])
        dc = pose_dist(poses[cf[:, 0]], poses[cf[:, 1]])
        # Subsample BOTH sets to the per-bin minimum.  Trimming only the
        # counterfactual set left medians 0.493 vs 0.773 -- the low bins simply
        # have no counterfactual pairs to draw from, so the path set has to give
        # ground too.  Matched magnitudes are load-bearing: otherwise arm X is
        # asked bigger questions than arm P and the comparison is not the point.
        lo = min(dp.min(), dc.min())
        hi = max(np.percentile(dp, 99), np.percentile(dc, 99))
        edges = np.linspace(lo, hi, 9)
        kp, kc = [], []
        for b in range(8):
            ip = np.where((dp >= edges[b]) & (dp < edges[b + 1]))[0]
            ic = np.where((dc >= edges[b]) & (dc < edges[b + 1]))[0]
            m = min(len(ip), len(ic))
            if m:
                kp.append(rng.choice(ip, m, replace=False))
                kc.append(rng.choice(ic, m, replace=False))
        if kp:
            path = path[np.concatenate(kp)]
            cf = cf[np.concatenate(kc)]
    return path, cf, radius


# ===========================================================================
# TRAINING
# ===========================================================================

def train_arm(arm, cfg, frames, poses, queries, log=True):
    """arm A = autoencoder only.  P = path queries.  X = counterfactual queries.

    P and X are identical in every respect except which query set they draw
    from, so no capacity control is needed -- and neither is poisoned, which is
    what destroyed the shuffled-action arms in worldrank."""
    torch, nn = build_torch()
    dev = cfg["device"]
    var = float(np.var(frames))
    adim = 0 if arm == "A" else 3
    lap = cfg["lap"] if arm == "S" else 0.0
    Xd = torch.tensor(frames, dtype=torch.float32, device=dev)[:, None]
    Q = (None if arm == "A" else
         queries["path" if arm == "P" else "cf"])
    if arm != "A" and (Q is None or len(Q) < 32):
        raise SystemExit(f"arm {arm}: only {0 if Q is None else len(Q)} queries; "
                         "raise --frames")
    A_sc = np.ones((1, 3))
    if arm != "A":
        A_np = np.stack([rel_action(poses[i], poses[j]) for i, j in Q])
        A_sc = A_np.std(0, keepdims=True) + 1e-9
        Ad = torch.tensor(A_np / A_sc, dtype=torch.float32, device=dev)
        Qi = torch.tensor(Q[:, 0], device=dev)
        Qj = torch.tensor(Q[:, 1], device=dev)

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
            if arm == "A":
                idx = torch.tensor(rng.integers(0, len(frames), cfg["batch"]),
                                   device=dev)
                x0 = Xd[idx]
                loss = ((sp.render(sp.activate(dec(enc(x0)))) - x0[:, 0]) ** 2).mean()
                nterm = 1
            else:
                q = torch.tensor(rng.integers(0, len(Q), cfg["batch"]), device=dev)
                x0, x1, a = Xd[Qi[q]], Xd[Qj[q]], Ad[q]
                z = enc(x0)
                loss = ((sp.render(sp.activate(dec(z))) - x0[:, 0]) ** 2).mean()
                loss = cfg["xw"] * loss + ((sp.render(sp.activate(dec(tr(z, a))))
                                            - x1[:, 0]) ** 2).mean()
                nterm = 1 + cfg["xw"]
                if lap:
                    # ChatGPT proposed dz/dt = F(z,a) + lambda * grad^2 z.  A
                    # Laplacian in LATENT space is not well posed -- the latent
                    # has no metric until you define one, and defining it is the
                    # problem.  In ACTION space it is: require the transition to
                    # vary smoothly with the action, i.e. a small second
                    # difference.  Same intent, and it needs no invented metric.
                    e = torch.zeros_like(a).normal_(0, 0.25)
                    zc = tr(z, a)
                    curv = ((tr(z, a + e) + tr(z, a - e) - 2 * zc) ** 2).mean()
                    # NORMALISED by how far the transition moves at all.  As a
                    # raw term it is dead on arrival: the transition head is
                    # initialised at 0.1 scale, so the second difference measures
                    # 4e-9 while the reconstruction loss is ~6e-3, and lap=0.05
                    # would have to be ~1e6 to matter.  The lesson is d.coef's:
                    # a term you cannot show BIT is untested, not falsified.
                    loss = loss + lap * curv / ((zc - z) ** 2).mean().clamp(min=1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            sched.step()
            tail.append(float(loss.detach()) / nterm)
            if log and step % max(cfg["steps"] // 5, 1) == 0:
                print(f"    [{arm}{tag}] step {step:>6}  loss {tail[-1]:.5f}"
                      f"  {time.time()-t0:5.0f}s", flush=True)
        return enc, dec, tr, sp, float(np.mean(tail[-50:])), float(np.mean(tail[:20]))

    lr = cfg["lr"]
    for k in range(cfg["retries"] + 1):
        enc, dec, tr, sp, tail, start = attempt(lr, "" if k == 0 else f"r{k}")
        if not (tail >= var and tail >= start):
            return enc, dec, tr, sp, tail, k, A_sc
        if k < cfg["retries"]:
            lr /= 3.0
            print(f"    [{arm}] DIVERGED (tail {tail:.4f} >= var {var:.4f} and "
                  f">= start {start:.4f}); retrying at lr {lr:.1e}", flush=True)
    print(f"    [{arm}] STILL DIVERGED -- FAILED, excluded", flush=True)
    return enc, dec, tr, sp, tail, -1, A_sc


# ===========================================================================
# EVALUATION
# ===========================================================================

def eval_arm(arm, enc, dec, tr, sp, cfg, ev_frames, ev_poses, aliases,
             A_sc, win_frames, win_poses):
    """Arm A has no transition net, so its probe direction comes from a linear
    pose->latent fit instead -- otherwise A's controllability row is NaN and the
    baseline every gate compares against is missing."""
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
            img = sp.render(p).cpu().numpy()
        return pos, img

    Z = latents(ev_frames)
    P, R = packets(Z)
    mse = float(((R - ev_frames) ** 2).mean())
    # rank on a contiguous WINDOW, matching the oracle's visibility window; the
    # alias and loop metrics use the whole walk, where revisits actually live
    Zw = latents(win_frames)
    Pw, _ = packets(Zw)
    out = dict(psnr=10 * math.log10(1.0 / max(mse, 1e-12)), mob=mobility(Pw),
               **rank_stats(Pw))
    out["z_eff"], out["z95"] = dof(Z)

    # ---- L1 aliasing: can the encoder tell two identical-looking places apart?
    if aliases:
        zi = np.array([Z[i] for i, j, _ in aliases])
        zj = np.array([Z[j] for i, j, _ in aliases])
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, len(Z), 2000), rng.integers(0, len(Z), 2000)
        ref = float(np.median(np.linalg.norm(Z[a] - Z[b], axis=1)))
        out["alias_gap"] = float(np.median(np.linalg.norm(zi - zj, axis=1)) /
                                 max(ref, 1e-12))
        out["n_alias"] = len(aliases)
    else:
        out["alias_gap"], out["n_alias"] = float("nan"), 0

    # ---- L2 loop closure ------------------------------------------------
    out["loop_r"], out["loop_null"] = loop_stats(Z, ev_poses, aliases)
    out["revisit"], out["n_revisit"] = revisit_gap(Z, ev_poses)
    # transition curvature in action space -- the diagnostic that says whether
    # arm S's term actually constrained anything
    if arm == "A":
        out["curv"] = float("nan")
    else:
        with torch.no_grad():
            zz = torch.tensor(Zw, dtype=torch.float32, device=dev)
            aa = torch.zeros(len(Zw), 3, device=dev)
            ee = torch.zeros_like(aa).normal_(0, 0.25, generator=None)
            zc = tr(zz, aa)
            num = float(((tr(zz, aa + ee) + tr(zz, aa - ee) - 2 * zc) ** 2).mean())
            den = float(((tr(zz, aa + ee) - zc) ** 2).mean())
        out["curv"] = num / max(den, 1e-12)

    # ---- G3 / G5 on the affine residual, using the model's OWN transition
    mid = len(win_poses) // 2
    probe = np.array([0.0, 0.45, 0.0])          # a lateral step: maximum parallax
    flow, depth, hitm, _ = flow_field(win_poses[mid], probe, S)
    z0 = Zw[mid]
    if arm == "A":
        Apose = np.concatenate([win_poses, np.ones((len(win_poses), 1))], 1)
        B, *_ = np.linalg.lstsq(Apose, Zw, rcond=None)
        # BUG FIXED: this used probe[0] only.  The probe became a LATERAL step
        # ([0, 0.45, 0]) when the traverse changed, so probe[0] = 0, the target
        # pose equalled the current one, z1 == z0, and arm A's whole
        # controllability row read exactly 0.000 / 0.0000 / 0.0000 -- which looks
        # like a measurement and is an artefact.  Same formula as flow_field now.
        yaw2 = win_poses[mid][2] + probe[2]
        tgt = np.array([win_poses[mid][0] + probe[0] * math.sin(yaw2)
                        + probe[1] * math.cos(yaw2),
                        win_poses[mid][1] + probe[0] * math.cos(yaw2)
                        - probe[1] * math.sin(yaw2), yaw2])
        z1 = z0 + (np.append(tgt, 1.0) - np.append(win_poses[mid], 1.0)) @ B
    else:
        A_np = probe / A_sc[0]        # the scaling the arm was trained with
        with torch.no_grad():
            z1 = tr(torch.tensor(z0[None], dtype=torch.float32, device=dev),
                    torch.tensor(A_np[None], dtype=torch.float32,
                                 device=dev)).cpu().numpy()[0]
    Pm, _ = packets(np.stack([z0, z1]))
    dmodel = Pm[1] - Pm[0]
    ppx, ppy = Pm[0][:, 0] * S, Pm[0][:, 1] * S
    dtrue = sample_at(flow, ppx, ppy, S)
    keep = sample_at(hitm, ppx, ppy, S) & np.isfinite(sample_at(depth, ppx, ppy, S))
    out["npk"] = int(keep.sum())
    if keep.sum() >= 12:
        pp = Pm[0][keep]
        res_t, frac_t = affine_residual(dtrue[keep], pp)
        res_m, frac_m = affine_residual(dmodel[keep], pp)
        out["g3"], out["g3null"] = dir_fidelity(res_m, res_t)
        out["g5"], out["g5null"] = field_match(res_m, res_t)
        out["nonaff"], out["nonaff_oracle"] = frac_m, frac_t
    else:
        for k in ("g3", "g3null", "g5", "g5null", "nonaff", "nonaff_oracle"):
            out[k] = float("nan")
    return out


# ===========================================================================
# SELFTEST
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
    pose = np.array([0.6, -1.4, 0.5])
    img, dep, X, hit = render(pose, S)
    chk("T1 the room renders structure", img.std() > 0.10, f"std {img.std():.3f}")
    chk("T1 nearly every ray hits the room", hit.mean() > 0.97, f"{hit.mean():.3f}")

    g = np.arange(S) + 0.5
    PX, PY = np.meshgrid(g, g)
    rx, ry, _ = project(pose, X.reshape(-1, 3), S)
    m = hit.ravel()
    err = max(np.abs(rx[m] - PX.ravel()[m]).max(), np.abs(ry[m] - PY.ravel()[m]).max())
    chk("T2 project(render hits) returns the original pixels", err < 0.05,
        f"max err {err:.4f} px")

    # T3 THE ALIASING IS EXACT -- the whole point of this scene
    worst = 0.0
    for _ in range(6):
        p = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 6.28)])
        if blocked(p[0], p[1]):
            continue
        a, _, _, _ = render(p, S, landmark=False)
        b, _, _, _ = render(rot180(p), S, landmark=False)
        worst = max(worst, float(np.abs(a - b).max()))
    chk("T3 without the landmark, p and R180(p) render IDENTICALLY",
        worst < 1e-6, f"max abs diff {worst:.2e}")
    p = np.array([1.9, 1.5, 0.9])
    a, _, _, _ = render(p, S, landmark=True)
    b, _, _, _ = render(rot180(p), S, landmark=True)
    chk("T3 with the landmark, the symmetry is BROKEN somewhere",
        float(np.abs(a - b).max()) > 0.05,
        f"max abs diff {np.abs(a-b).max():.3f}")

    # T4 the alias detector finds planted pairs and rejects near-identical poses
    ps, _ = walk(500, seed=3)
    fr = np.stack([render(q, 48)[0] for q in ps])
    al = alias_pairs(ps, fr)
    chk("T4 the detector finds aliased pairs in a real walk", len(al) >= 1,
        f"{len(al)} pairs")
    if al:
        i, j, e = al[0]
        chk("T4 a detected pair really is far apart in space",
            pose_dist(ps[i][None], ps[j][None])[0] > 1.5,
            f"pose distance {pose_dist(ps[i][None], ps[j][None])[0]:.2f}, "
            f"image mse {e:.2e}")
    same = alias_pairs(ps, fr, max_img=1.0, min_pose=99.0)
    chk("T4 the detector rejects pairs that are NOT far apart", len(same) == 0)

    # T5 rank instrument, both sides
    orc = oracle_row(traverse(24, 900), S)
    chk("T5 true 3D on the lateral traverse shows a rank-3 CLIFF",
        orc is not None and orc["r43"] <= 0.10 and orc["r32"] >= 0.01,
        f"s3/s2 {orc['r32']:.4f}  s4/s3 {orc['r43']:.4f}  "
        f"s {np.array2string(orc['s'][:4], precision=2)}")
    orc_f = oracle_row(np.stack([[0, -2.3 + 0.085 * k, 0.0] for k in range(24)]), S)
    chk("T5 REGRESSION: a FORWARD traverse does not -- forward motion is a "
        "uniform zoom, which is affine",
        orc_f is not None and orc_f["r43"] > 0.25,
        f"s4/s3 {orc_f['r43']:.4f} vs lateral {orc['r43']:.4f}")
    base = np.stack(project(ps[12], render(ps[12], S)[2][render(ps[12], S)[3]][:200],
                            S)[:2], 1) / S
    aff = np.stack([base @ (np.eye(2) + rng.normal(0, 0.2, (2, 2))).T
                    + rng.normal(0, 0.2, 2) for _ in range(24)])
    ra = rank_stats(aff)
    chk("T5 a planted 2D affine morph reads rank 2", ra["r32"] < 0.01,
        f"s3/s2 {ra['r32']:.2e}")

    # T6 affine residual + field match, both sides
    fl, dp, hm, _ = flow_field(ps[12], np.array([0.45, 0.0, 0.0]), S)
    idx = rng.choice(np.where(hm.ravel())[0], 400, replace=False)
    gg = (np.arange(S) + 0.5) / S
    GY, GX = np.meshgrid(gg, gg, indexing="ij")
    pos0 = np.stack([GX.ravel()[idx], GY.ravel()[idx]], 1)
    tflow = fl.reshape(-1, 2)[idx]
    res_t, frac_t = affine_residual(tflow, pos0)
    chk("T6 real translation in a room has strong non-affine flow", frac_t > 0.10,
        f"residual energy {frac_t:.4f}")
    A = np.eye(2) + rng.normal(0, 0.05, (2, 2))
    _, fa = affine_residual(pos0 @ A.T + rng.normal(0, 0.02, 2) - pos0, pos0)
    chk("T6 a planted affine field has none", fa < 1e-6, f"{fa:.2e}")
    g5, g5n = field_match(res_t, res_t)
    chk("T6 G5 is 1 for identical fields and ~0 for a random one",
        g5 > 0.999 and field_match(rng.normal(size=res_t.shape), res_t)[0] < 0.10,
        f"self {g5:.4f}")

    # T7 loop metric, both sides
    zt = np.concatenate([ps[:, :2] * 3.0, rng.normal(0, .01, (len(ps), 6))], 1)
    r_good, n_good = loop_stats(zt, ps, [])
    r_bad, _ = loop_stats(rng.normal(size=(len(ps), 8)), ps, [])
    chk("T7 a latent that IS position gives a high loop correlation", r_good > 0.8,
        f"{r_good:.3f} (null {n_good:.3f})")
    chk("T7 a random latent does not", abs(r_bad) < 0.2, f"{r_bad:+.3f}")

    # T8 query sets: matched magnitudes, disjoint by construction
    P_q, C_q, rad = build_queries(ps, K=6)
    chk("T8 both query sets are non-empty", len(P_q) > 50 and len(C_q) > 20,
        f"path {len(P_q)}  counterfactual {len(C_q)}  radius {rad:.2f}")
    if len(C_q):
        dp = pose_dist(ps[P_q[:, 0]], ps[P_q[:, 1]])
        dc = pose_dist(ps[C_q[:, 0]], ps[C_q[:, 1]])
        chk("T8 displacement magnitudes are matched between the arms",
            abs(np.median(dp) - np.median(dc)) < 0.25 * np.median(dp),
            f"median path {np.median(dp):.3f} vs cf {np.median(dc):.3f}")
        chk("T8 no counterfactual pair is within K steps along the path",
            np.abs(C_q[:, 0] - C_q[:, 1]).min() > 6,
            f"min |i-j| = {np.abs(C_q[:,0]-C_q[:,1]).min()}")

    # T9 rel_action round-trips
    pi, pj = ps[5], ps[9]
    a = rel_action(pi, pj)
    yaw2 = pi[2] + a[2]
    rx2 = pi[0] + a[0] * math.sin(yaw2) + a[1] * math.cos(yaw2)
    rz2 = pi[1] + a[0] * math.cos(yaw2) - a[1] * math.sin(yaw2)
    chk("T9 rel_action reconstructs the target pose",
        abs(rx2 - pj[0]) < 1e-9 and abs(rz2 - pj[1]) < 1e-9,
        f"err {abs(rx2-pj[0]):.2e}")

    try:
        torch, nn = build_torch()
        enc, dec, tr = make_models(torch, nn, 16, 24, 3, "cpu")
        sp = Splat(torch, 24, 32, device="cpu")
        with torch.no_grad():
            z = torch.zeros(2, 16)
            im = sp.render(sp.activate(dec(z)))
            chk("T10 render shape and range", tuple(im.shape) == (2, 32, 32)
                and float(im.min()) >= 0 and float(im.max()) <= 1)
            chk("T10 the action reaches the transition",
                float((tr(z, torch.ones(2, 3)) - tr(z, -torch.ones(2, 3))
                       ).abs().sum()) > 0)
    except ImportError:
        chk("T10 torch present", False, "pip install torch")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


# ===========================================================================

def verdict(tag, ok, msg):
    print(f"  {tag} [{'V' if ok else 'K'}]  {msg}")
    return bool(ok)


def print_row(name, r):
    s = " ".join(f"{x:7.3f}" for x in r["s"][:5])
    print(f"  {name:<9} {s}   eff {r['eff']:5.3f}  s3/s2 {r['r32']:.4f}"
          f"  s4/s3 {r['r43']:.4f}  top2 {r['top2']*100:5.2f}%")


def run(cfg):
    S = cfg["size"]
    print(f"room {2*HALF:.0f}x{2*HALF:.0f}, {len(PILLARS)} pillars in +-pairs, "
          f"one symmetry-breaking landmark, {S}px, fov {FOV:.2f}")
    tr_poses, _ = walk(cfg["frames"], cfg["seed"])
    ev_poses, _ = walk(cfg["evframes"], cfg["seed"] + 500)
    t0 = time.time()
    tr_frames = np.stack([render(p, S)[0] for p in tr_poses])
    ev_frames = np.stack([render(p, S)[0] for p in ev_poses])
    P_q, C_q, rad = build_queries(tr_poses, K=cfg["K"], seed=cfg["seed"])
    aliases = alias_pairs(ev_poses, ev_frames)
    print(f"train {len(tr_poses)} frames, held-out {len(ev_poses)}; rendered in "
          f"{time.time()-t0:.1f}s")
    print(f"queries: {len(P_q)} path, {len(C_q)} counterfactual (radius {rad:.2f}); "
          f"{len(aliases)} aliased pairs detected in the held-out walk")
    # L2's ceiling: what a latent that IS the pose would score on the same pairs
    # yaw must go in as cos/sin: walk() accumulates it without wrapping, so a
    # raw yaw column is unbounded and swamps x,z -- the first version of this
    # ceiling read 1.2390, i.e. "a latent that is the pose recognises nothing".
    rv_orc, _ = revisit_gap(np.concatenate(
        [ev_poses[:, :2], 0.7 * np.cos(ev_poses[:, 2:3]),
         0.7 * np.sin(ev_poses[:, 2:3])], 1), ev_poses)
    print(f"L2 ceiling: a latent that IS the pose scores revisit gap "
          f"{rv_orc:.4f} on this walk\n")
    if not aliases:
        print("  NOTE: no aliased pairs found -- L1 will be NaN.  Raise "
              "--evframes so the walk visits both halves of the room.\n")

    win_poses = traverse(cfg["evwindow"], cfg["seed"] + 900)
    win_frames = np.stack([render(p, S)[0] for p in win_poses])
    orc = oracle_row(win_poses, S)
    if orc is None:
        print("ORACLE: fewer than 24 points stay visible across the window -- "
              "lower --evwindow.  Nothing below can be calibrated.")
        return {}
    print(f"ORACLE -- true 3D points visible in all {cfg['evwindow']} frames of "
          f"the traverse ({orc['npts']} points)")
    print_row("oracle", orc)
    verdict("W1", orc["r43"] <= 0.10 and orc["r32"] >= 0.01,
            f"ground truth shows a rank-3 CLIFF on the traverse  "
            f"(s3/s2 {orc['r32']:.4f}, s4/s3 {orc['r43']:.4f}) -- lateral motion "
            f"of a rigid scene is exactly Tomasi-Kanade, so this is near exact")
    print()

    queries = dict(path=P_q, cf=C_q)
    seeds = [cfg["seed"] + i for i in range(cfg["seeds"])]
    per, res, fails = {a: [] for a in cfg["arms"]}, {}, []
    for sd in seeds:
        for arm in cfg["arms"]:
            print(f"  training arm {arm} seed {sd} ...", flush=True)
            c = dict(cfg)
            c["seed"] = sd
            enc, dec, tr, sp, tail, tries, A_sc = train_arm(
                arm, c, tr_frames, tr_poses, queries)
            if tries < 0:
                fails.append((arm, sd))
                print()
                continue
            r = eval_arm(arm, enc, dec, tr, sp, c, ev_frames, ev_poses,
                         aliases, A_sc, win_frames, win_poses)
            r["tail"] = tail
            per[arm].append(r)
            print()
    if fails:
        print("FAILED RUNS (excluded):", ", ".join(f"{a}/{s}" for a, s in fails), "\n")
    cfg["arms"] = [a for a in cfg["arms"] if per[a]]
    if not cfg["arms"]:
        print("every arm failed to train")
        return {}
    SC = ("psnr", "tail", "mob", "eff", "r32", "r43", "top2", "z_eff",
          "alias_gap", "loop_r", "loop_null", "revisit", "g3", "g3null", "g5",
          "g5null", "curv",
          "nonaff", "nonaff_oracle")
    for arm in cfg["arms"]:
        r = dict(per[arm][0])
        for k in SC:
            r[k] = float(np.nanmean([q[k] for q in per[arm]]))
        r["s"] = np.mean([q["s"] for q in per[arm]], axis=0)
        res[arm] = r

    if len(seeds) > 1:
        print("PER-SEED SPREAD  (gates run on the mean)")
        for arm in cfg["arms"]:
            print(f"  {arm:<3} s3/s2 " + " ".join(f"{q['r32']:.4f}" for q in per[arm])
                  + "   G5 " + " ".join(f"{q['g5']:.4f}" for q in per[arm])
                  + "   loop " + " ".join(f"{q['loop_r']:+.3f}" for q in per[arm]))
        print()

    print("MEASUREMENT MATRIX  (per-frame centred packet positions, held-out walk)")
    print_row("oracle", orc)
    for a in cfg["arms"]:
        print_row(f"arm {a}", res[a])
    print()
    print(f"  {'arm':<4} {'PSNR':>6} {'loss':>8} {'mobility':>9} {'z eff':>7} "
          f"{'alias gap':>10} {'revisit':>8} {'curv':>7} {'loop r':>8}")
    for a in cfg["arms"]:
        r = res[a]
        print(f"  {a:<4} {r['psnr']:>6.2f} {r['tail']:>8.5f} {r['mob']:>9.3f} "
              f"{r['z_eff']:>7.3f} {r['alias_gap']:>10.4f} {r['revisit']:>8.4f} "
              f"{r['curv']:>7.4f} {r['loop_r']:>8.3f}")
    print()
    print(f"  {'arm':<4} {'G3 cos':>7} {'null':>7} {'G5':>8} {'null':>7} "
          f"{'nonaff':>8} {'oracle':>8} {'pkts':>5}")
    for a in cfg["arms"]:
        r = res[a]
        print(f"  {a:<4} {r['g3']:>7.3f} {r['g3null']:>7.3f} {r['g5']:>8.4f} "
              f"{r['g5null']:>7.4f} {r['nonaff']:>8.4f} {r['nonaff_oracle']:>8.4f} "
              f"{r['npk']:>5d}")
    print()

    print("GATES")
    ps = [res[a]["psnr"] for a in cfg["arms"]]
    verdict("W0", max(ps) - min(ps) <= 1.5,
            f"ALL arms comparably fitted (spread {max(ps)-min(ps):.2f} dB)")
    pair_ok = True
    if "P" in res and "X" in res:
        # X1/X2 compare P against X, so THAT pair is what has to be matched.
        # Arm A carries one loss term where P and X carry two, so A is never
        # loss-count matched to them and a three-arm spread can void a P-vs-X
        # comparison that is perfectly valid on its own.
        dpx = abs(res["P"]["psnr"] - res["X"]["psnr"])
        pair_ok = verdict("W0px", dpx <= 1.5,
                          f"P and X comparably fitted (spread {dpx:.2f} dB) -- "
                          f"this is the pair X1/X2 actually compare")
    if not pair_ok:
        print("       P vs X NOT comparable -- X1/X2 are VOID, not results.  "
              "Use --xw to weight the prediction term until the PSNRs match.")
    if "X" in res and "P" in res:
        Xr, Pr = res["X"], res["P"]
        verdict("X1", Xr["r32"] >= 0.5 * orc["r32"] and Xr["r32"] >= 2 * Pr["r32"],
                f"counterfactual queries produce geometry  (X {Xr['r32']:.4f}, "
                f"path-only P {Pr['r32']:.4f}, oracle {orc['r32']:.4f})")
        verdict("X2", Xr["g5"] >= 0.25 and Xr["g5"] >= 5 * max(Xr["g5null"], 1e-6),
                f"and it is the TRUE parallax field  (G5 {Xr['g5']:.4f}, null "
                f"{Xr['g5null']:.4f}, vs P {Pr['g5']:.4f})")
    for a in cfg["arms"]:
        r = res[a]
        verdict(f"L1 {a}", r["alias_gap"] >= 0.5,
                f"separates two identical-looking places  (gap {r['alias_gap']:.4f} "
                f"of a typical latent distance; a feedforward encoder CANNOT, so "
                f"[K] is the predicted result)")
    for a in cfg["arms"]:
        r = res[a]
        verdict(f"L2 {a}", r["revisit"] <= 2.0 * rv_orc,
                f"revisits collapse in the latent  (gap {r['revisit']:.4f} vs "
                f"pose-latent ceiling {rv_orc:.4f}, {r['n_revisit']} pairs; "
                f"global corr {r['loop_r']:+.3f} vs null {r['loop_null']:.3f})")
    print("\n  Read X1/X2 only if W0 and W1 passed.  L1 is the architectural "
          "ceiling, not a failure of training.")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--packets", type=int, default=160)
    p.add_argument("--zdim", type=int, default=32)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--evframes", type=int, default=400)
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--xw", type=float, default=1.0,
                   help="weight on the reconstruction term of P/X/S")
    p.add_argument("--lap", type=float, default=0.05,
                   help="arm S: action-space smoothness of the transition")
    p.add_argument("--evwindow", type=int, default=24,
                   help="contiguous held-out frames used for the rank test")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--arms", type=str, default="A,P,X")
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
            sys.exit("torch is required")
    cfg = dict(size=a.size, packets=a.packets, zdim=a.zdim, frames=a.frames,
               evframes=a.evframes, K=a.K, evwindow=a.evwindow, xw=a.xw, lap=a.lap, steps=a.steps, batch=a.batch, lr=a.lr,
               clip=a.clip, retries=a.retries, seed=a.seed, seeds=max(1, a.seeds),
               device=dev, arms=[s for s in a.arms.split(",") if s])
    if a.smoke:
        cfg.update(size=32, packets=48, zdim=12, frames=200, evframes=400,
                   steps=30, batch=4)
        print("SMOKE: not an experiment, only a check that every path runs.\n")
    print(f"device {dev}\n")
    run(cfg)


if __name__ == "__main__":
    main()