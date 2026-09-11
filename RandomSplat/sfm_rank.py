#!/usr/bin/env python3
"""
sfm_rank.py -- is the head turn a real 3D rotation, or a 2D morph that looks
like one?  Tomasi-Kanade factorisation run on packet trajectories.

THE TEST
    Sweep z along the decoder's dominant Jacobian direction (the axis that
    produces the head turn -- see rank1_probe.py) and record packet centroids
    at F samples.  Stack them into the measurement matrix

        W (2F x N),   rows = [x_1..x_N ; y_1..y_N] per sample,
                      each frame centred to remove translation.

    Tomasi & Kanade 1992: for orthographic projection of a RIGID 3D shape,

        W = R (2F x 3) . S (3 x N)      =>   rank(W) = 3, exactly.

    A 2D affine morph that merely resembles rotation gives rank 2, exactly:
    p_f = A_f q + t_f centres to W_f = A_f q, so W = [A_1;..;A_F] q, rank <= 2.

    Rank 3 vs rank 2 is therefore a clean yes/no on whether the model carries
    depth.  Single-axis yaw is sufficient: R_f = [[c,0,-s],[0,1,0]] spans all
    three columns as long as the shape has depth variation.

WHY THIS IS CHEAP HERE
    Classical SfM's hard part is correspondence -- tracking one physical point
    across frames.  There is none to solve: packet k is packet k at every
    latent, so the decoder emits N perfectly tracked points as its native
    output.

GATES
    SF0  packet mobility.  If the renderer's raster-cell anchoring pins
         centroids near their cells, they are not tracking material points and
         the whole readout is void.  Requires median centroid travel >= 0.20 x
         median nearest-neighbour spacing.  THIS GATES EVERYTHING BELOW.
    SF1  rank-3 signature.  s3^2 must carry >= 5% of the energy that s2^2 does,
         AND s4 must be small against s3.  Both, or it is a 2D morph.
    SF2  metric upgrade is consistent.  The recovered rotations must satisfy
         orthonormality; if the Gram system is not PSD the "3D" is an artefact.

    python3.13 sfm_rank.py --model model2.pt --image me.png
    python3.13 sfm_rank.py --selftest        (no model needed)
"""

import argparse
import sys

import numpy as np


# ----------------------------------------------------------- factorisation ---

def measurement_matrix(tracks):
    """tracks: (F, N, 2) -> centred W (2F, N) and the per-frame centroids."""
    F, N, _ = tracks.shape
    c = tracks.mean(axis=1, keepdims=True)          # translation per frame
    t = tracks - c
    W = np.concatenate([t[:, :, 0], t[:, :, 1]], axis=0)
    order = np.empty(2 * F, dtype=int)
    order[0::2] = np.arange(F)                       # interleave x,y rows
    order[1::2] = np.arange(F) + F
    return W[order], c[:, 0, :]


def factorise(W):
    """Rank-3 truncation. Returns (M 2Fx3, S 3xN, singular values)."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    k = min(3, len(s))
    M = U[:, :k] * np.sqrt(s[:k])
    S = (np.sqrt(s[:k])[:, None]) * Vt[:k]
    return M, S, s


def metric_upgrade(M):
    """Solve for Q with M Q orthonormal per frame (Tomasi-Kanade eq. 16).

    Unknown is the symmetric Gram L = Q Q^T (6 dof). Each frame contributes
    i.L.i = j.L.j and i.L.j = 0. If L comes back non-PSD the rank-3 structure
    is not a rotation and the 3D reading is an artefact -- that is SF2."""
    F = M.shape[0] // 2
    A, b = [], []

    def vec(u, v):      # coefficients of L in <u, L v>, L symmetric 3x3
        return np.array([u[0] * v[0], u[1] * v[1], u[2] * v[2],
                         u[0] * v[1] + u[1] * v[0],
                         u[0] * v[2] + u[2] * v[0],
                         u[1] * v[2] + u[2] * v[1]])

    for f in range(F):
        i, j = M[2 * f], M[2 * f + 1]
        A.append(vec(i, i) - vec(j, j)); b.append(0.0)
        A.append(vec(i, j)); b.append(0.0)
        A.append(vec(i, i)); b.append(1.0)           # fix the scale
    L6, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    L = np.array([[L6[0], L6[3], L6[4]],
                  [L6[3], L6[1], L6[5]],
                  [L6[4], L6[5], L6[2]]])
    w = np.linalg.eigvalsh(L)
    psd = bool(w.min() > -1e-6 * max(abs(w).max(), 1e-12))
    if not psd:
        return None, L, w
    w2, V = np.linalg.eigh(L)
    Q = V @ np.diag(np.sqrt(np.maximum(w2, 0)))
    return Q, L, w


def rank_report(s, tag=""):
    e = s ** 2 / max((s ** 2).sum(), 1e-30)
    eff = float(np.exp(-(e * np.log(e + 1e-30)).sum()))
    r2, r3 = float(e[:2].sum()), float(e[:3].sum())
    print(f"  {tag}singulars      {np.array2string(s[:6], precision=4)}")
    print(f"  {tag}energy         top-2 {r2*100:6.2f}%   top-3 {r3*100:6.2f}%   "
          f"gain from the 3rd {(r3-r2)*100:.2f}%")
    print(f"  {tag}effective rank {eff:.3f}")
    ratio32 = float(s[2] ** 2 / max(s[1] ** 2, 1e-30)) if len(s) > 2 else 0.0
    ratio43 = float(s[3] ** 2 / max(s[2] ** 2, 1e-30)) if len(s) > 3 else 0.0
    print(f"  {tag}s3^2/s2^2 {ratio32:.4f}   s4^2/s3^2 {ratio43:.4f}")
    return eff, ratio32, ratio43


def linearity(tracks):
    """SF3: fraction of motion energy NOT explained by a straight line in t.

    A near-linear family p(t) = p0 + t d is rank 2 after centring NO MATTER
    what 3D structure is behind it, so if this reads ~0 the rank test cannot
    see depth and SF1 is uninformative rather than negative.  This is exactly
    the trap a straight-line latent sweep walks into."""
    F = len(tracks)
    t = np.linspace(-1.0, 1.0, F)
    A = np.stack([np.ones(F), t], 1)
    X = tracks.reshape(F, -1)
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    resid = X - A @ coef
    Xc = X - X.mean(0)
    return float((resid ** 2).sum() / max((Xc ** 2).sum(), 1e-30))


def mobility(tracks):
    """SF0: how far do centroids travel, against how far apart they sit?"""
    trav = np.linalg.norm(tracks.max(0) - tracks.min(0), axis=1)
    p = tracks[len(tracks) // 2]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(np.median(trav)), float(np.median(nn))


# ------------------------------------------------------------------ model ---

def dof(X, tag, labels=("95%", "99%")):
    """How many dimensions does this channel actually use?

    Positions came back rank 2, so the visible head turn must live in the
    COEFFICIENTS.  This measures how many modes each channel needs, which is
    what sets the per-frame payload when the model is driven from a webcam."""
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    e = sv ** 2 / max((sv ** 2).sum(), 1e-30)
    eff = float(np.exp(-(e * np.log(e + 1e-30)).sum()))
    cum = np.cumsum(e)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"  {tag:<22} dims {X.shape[1]:>5}   effective {eff:6.3f}   "
          f"{labels[0]} in {k95:>3}   {labels[1]} in {k99:>3}")
    return eff, k95, k99


def _load(path):
    """The real API, read off splat_ragdoll.py rather than guessed:
    load_splatvae returns (model, checkpoint) and takes no map_location, and
    packet positions come from ren.activate(dec(z)) -- the raw decoder output
    is PRE-activation and its first two columns are not coordinates."""
    import torch
    try:
        from splat_trainer5 import load_splatvae
    except ImportError:
        from splat_trainer4q import load_splatvae
    m, _ck = load_splatvae(path)
    m.eval()
    with torch.no_grad():                      # latent dim from the encoder,
        mu, _ = m.enc(torch.zeros(1, 3, m.ren.H, m.ren.H))   # never hardcoded
    return m, int(mu.shape[1])


def centroids(m, z):
    """(N,2) packet centres at latent z."""
    import torch
    with torch.no_grad():
        px, py, *_ = m.ren.activate(m.dec(z[None]).float())
    return torch.stack([px[0], py[0]], dim=1).numpy()


def encode(m, path):
    import cv2, torch
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"cannot read --image {path}")
    S = m.ren.H
    img = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        mu, _ = m.enc(x)
    return mu[0]


def sweep_sequence(path, seq, stride=1, maxf=48):
    """Track packets across REAL poses: encode each frame, decode, read centres.

    This is the input Tomasi-Kanade actually wants.  A latent line is a guess
    about where pose lives; a video of you turning your head is the pose."""
    import os, glob, torch, cv2
    m, zdim = _load(path)
    if os.path.isdir(seq):
        files = sorted(sum([glob.glob(os.path.join(seq, e))
                            for e in ("*.png", "*.jpg", "*.jpeg")], []))
        frames = [cv2.imread(f) for f in files][::stride][:maxf]
    else:
        cap = cv2.VideoCapture(seq); frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release(); frames = frames[::stride][:maxf]
    if len(frames) < 8:
        sys.exit(f"need >=8 frames from {seq}, got {len(frames)}")
    S = m.ren.H
    tracks, zs, coeffs = [], [], []
    for fr in frames:
        img = cv2.cvtColor(cv2.resize(fr, (S, S)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
        with torch.no_grad():
            mu, _ = m.enc(x)
            px, py, _sg, _th, _fr, cf = m.ren.activate(m.dec(mu).float())
        zs.append(mu[0].numpy())
        tracks.append(torch.stack([px[0], py[0]], 1).numpy())
        coeffs.append(cf[0].reshape(-1).numpy())
    print(f"  {len(frames)} real frames encoded from {seq}; "
          f"latent {zdim}, {m.ren.N} packets, {S}px")
    return np.array(tracks), m, np.array(zs), np.array(coeffs)


def sweep_model(path, image=None, F=24, span=3.0):
    import torch
    m, zdim = _load(path)
    z0 = encode(m, image) if image else torch.zeros(zdim)
    print(f"  latent {zdim}, {m.ren.N} packets, {m.ren.H}px"
          + (f", z0 from {image} (|mu| {float(z0.norm()):.3f})" if image else
             ", z0 = 0"))

    # dominant Jacobian direction: the axis that produces the head turn
    eps, base = 1e-2, centroids(m, z0)
    J = np.empty((base.size, zdim))
    for i in range(zdim):
        zp = z0.clone(); zp[i] += eps
        zm = z0.clone(); zm[i] -= eps
        J[:, i] = (centroids(m, zp).ravel() - centroids(m, zm).ravel()) / (2 * eps)
    sj = np.linalg.svd(J, full_matrices=False)
    v1 = sj[2][0]
    print(f"  Jacobian singulars {np.array2string(sj[1][:4], precision=4)}"
          f"  -> sweeping v1")

    tracks = [centroids(m, z0 + torch.as_tensor(t * v1, dtype=torch.float32))
              for t in np.linspace(-span, span, F)]
    return np.array(tracks), m


def write_ply(path, S, colors=None):
    N = S.shape[1]
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {N}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n")
        for k in range(N):
            f.write(f"{S[0,k]:.6f} {S[1,k]:.6f} {S[2,k]:.6f}\n")


# --------------------------------------------------------------- selftest ---

def selftest():
    ok = True

    def chk(n, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}  {d}")

    rng = np.random.default_rng(0)
    N, F = 200, 24
    print("SELFTEST\n")

    # PC: a real 3D shape under single-axis yaw must come out rank 3
    P = rng.standard_normal((3, N))
    tr = []
    for th in np.linspace(-0.6, 0.6, F):
        R = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0]])
        tr.append((R @ P).T + rng.standard_normal((N, 2)) * 1e-4)
    W, _ = measurement_matrix(np.array(tr))
    M, S, s = factorise(W)
    print("  3D rigid yaw")
    eff, r32, r43 = rank_report(s, "  ")
    # NOT effective rank: it is energy-weighted and reads 2.39 on a genuine
    # 71/62/24 spectrum. What identifies rank 3 is the CLIFF -- a third
    # component that carries real energy, and a fourth that carries none.
    chk("PC1 3D yaw shows the rank-3 cliff", r32 > 0.05 and r43 < 0.05,
        f"s3^2/s2^2 {r32:.4f}  s4^2/s3^2 {r43:.6f}")
    Q, L, w = metric_upgrade(M)
    chk("PC2 metric upgrade is PSD", Q is not None,
        f"eigs {np.array2string(w, precision=3)}")
    if Q is not None:
        # The upgrade fixes shape only UP TO A GLOBAL 3D ROTATION (and a
        # reflection), so comparing depth axis-to-axis is meaningless.
        # Procrustes-align first, then measure what is left.
        Sm = np.linalg.inv(Q) @ S
        A = Sm - Sm.mean(1, keepdims=True); B = P - P.mean(1, keepdims=True)
        A = A / np.linalg.norm(A); B = B / np.linalg.norm(B)
        Uq, sq, Vq = np.linalg.svd(A @ B.T)
        resid = float(np.linalg.norm((Vq.T @ Uq.T) @ A - B))
        chk("PC3 shape recovered up to a rotation", resid < 0.15,
            f"Procrustes residual {resid:.4f}  (0 = exact)")
        d3 = np.linalg.norm(Sm[2] - Sm[2].mean())
        chk("PC3b the third dimension is not degenerate",
            d3 / np.linalg.norm(Sm) > 0.05, f"depth share {d3/np.linalg.norm(Sm):.4f}")

    # NC: a 2D affine morph that resembles rotation must come out rank 2
    q = rng.standard_normal((2, N))
    tr = []
    for th in np.linspace(-0.6, 0.6, F):
        A = np.array([[np.cos(th), -0.3 * np.sin(th)], [0.1 * np.sin(th), 1.0]])
        tr.append((A @ q).T + rng.standard_normal((N, 2)) * 1e-4)
    W2, _ = measurement_matrix(np.array(tr))
    _, _, s2 = factorise(W2)
    print("\n  2D affine morph (looks like rotation, has no depth)")
    eff2, r32b, _ = rank_report(s2, "  ")
    chk("NC1 2D morph reads rank 2, not 3", eff2 < 2.4 and r32b < 0.05,
        f"eff {eff2:.3f}, s3^2/s2^2 {r32b:.5f}")

    # NC: a 3D shape sampled along a LINEAR path is rank 2 no matter what --
    # this is the confound a straight latent sweep creates, made visible
    d = rng.standard_normal((2, N))
    lin = np.array([(P[:2] + t * d).T for t in np.linspace(-1, 1, F)])
    _, _, sl = factorise(measurement_matrix(lin)[0])
    _, r32l, _ = rank_report(sl, "  ")
    # CORRECTED: a generic linear-in-t motion field has row space spanned by
    # {p0x, p0y, dx, dy} -> RANK 4, not 2.  A straight latent sweep therefore
    # does NOT force rank 2, and reading rank 2 out of real data is a genuine
    # and restrictive finding, not a sampling artefact.  Rank 2 means
    # W_f = A_f S: every frame is one 2x2 linear map of a FIXED 2D shape.
    chk("NC3 linear path on 3D data reads rank 4, not 2", r32l > 0.05,
        f"s3^2/s2^2 {r32l:.4f}  -- so a linear sweep cannot fake rank 2")
    chk("NC3b SF3 catches it", linearity(lin) < 0.02,
        f"nonlinear fraction {linearity(lin)*100:.4f}%")
    rot = np.array([(np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0]]) @ P).T
                    for t in np.linspace(-.6, .6, F)])
    chk("NC3c SF3 passes a real rotation", linearity(rot) >= 0.02,
        f"nonlinear fraction {linearity(rot)*100:.4f}%")

    # NC: anchored packets must trip SF0 before anything else is believed
    anc = rng.standard_normal((N, 2)) * 1.0
    tr = np.array([anc + rng.standard_normal((N, 2)) * 1e-3 for _ in range(F)])
    trav, nn = mobility(tr)
    chk("NC2 pinned packets trip the mobility gate", trav / nn < 0.20,
        f"travel/spacing {trav/nn:.4f}")
    trav, nn = mobility(np.array([(np.array([[np.cos(t), 0, -np.sin(t)],
                                             [0, 1, 0]]) @ P).T
                                  for t in np.linspace(-.6, .6, F)]))
    chk("NC2b a real sweep passes it", trav / nn >= 0.20,
        f"travel/spacing {trav/nn:.4f}")

    print(f"\nSELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='')
    p.add_argument('--image', type=str, default='')
    p.add_argument('--frames', type=int, default=24)
    p.add_argument('--span', type=float, default=3.0)
    p.add_argument('--seq', type=str, default='',
                   help='folder of frames or a video: track REAL poses instead '
                        'of a synthetic latent line (removes the SF3 confound)')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--ply', type=str, default='packets3d.ply')
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.model:
        print("need --model (or --selftest)"); sys.exit(2)

    zs = coeffs = None
    if a.seq:
        tracks, m, zs, coeffs = sweep_sequence(a.model, a.seq, a.stride, a.frames)
    else:
        tracks, m = sweep_model(a.model, a.image or None, a.frames, a.span)
    F, N, _ = tracks.shape
    print(f"\n{a.model}: {N} packets tracked over {F} latent samples")

    trav, nn = mobility(tracks)
    ratio = trav / max(nn, 1e-12)
    print(f"\nSF0 mobility   median travel {trav:.4f}, median spacing {nn:.4f}, "
          f"ratio {ratio:.3f}")
    if ratio < 0.20:
        print("SF0 [K]  centroids barely move against their spacing -- the "
              "renderer's raster-cell anchoring is pinning them, so they are "
              "NOT tracking material points. Everything below is void; the "
              "readout would have to be optical flow on the renders instead.")
        sys.exit(1)
    print("SF0 [V]  packets are mobile enough to act as tracked points\n")

    nl = linearity(tracks)
    print(f"SF3 curvature  {nl*100:.2f}% of motion energy is NOT linear in t")
    print("SF3 is a DIAGNOSTIC, not a gate: straight trajectories give rank 4,\n"
          "     not rank 2, so low curvature cannot manufacture a rank-2 result."
          "\n")

    if zs is not None:
        print("\nCHANNEL BUDGET  (how many numbers per frame does driving "
              "this model actually take?)")
        dof(zs, "latent z")
        dof(tracks.reshape(len(tracks), -1), "packet positions")
        dof(coeffs, "packet coefficients")
        print("  geometry needs 4 (a 2x2 affine on a fixed shape) + 2 for "
              "translation; the rest of the payload is coefficients\n")

    W, _ = measurement_matrix(tracks)
    M, S, s = factorise(W)
    eff, r32, r43 = rank_report(s)
    sf1 = (r32 > 0.05) and (r43 < 0.05)
    print(f"\nSF1 [{'V' if sf1 else 'K'}]  "
          f"{'rank-3 signature: the turn carries DEPTH' if sf1 else
             'no third component: this is a 2D morph that resembles rotation'}")

    Q, L, w = metric_upgrade(M)
    print(f"SF2 [{'V' if Q is not None else 'K'}]  metric upgrade "
          f"{'PSD' if Q is not None else 'NOT PSD'}, "
          f"Gram eigenvalues {np.array2string(w, precision=4)}")

    if sf1 and Q is not None:
        S3 = np.linalg.inv(Q) @ S
        write_ply(a.ply, S3)
        d = S3[2]
        print(f"\n3D packet cloud written to {a.ply}")
        print(f"  depth spread {d.std():.4f} against in-plane "
              f"{S3[:2].std():.4f}  (ratio {d.std()/max(S3[:2].std(),1e-9):.3f})")
        print("  a 3D point cloud of Gabor packets, from a 2D model, "
              "with no 3D data and no training")


if __name__ == '__main__':
    main()