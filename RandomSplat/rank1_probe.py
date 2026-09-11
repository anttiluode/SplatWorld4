#!/usr/bin/env python3
"""
rank1_probe.py -- is the ragdoll's "it knows direction" a single latent axis?

CLAIM UNDER TEST
    The decoder Jacobian at real latents has effective rank ~1 (your FS4b).
    If so, DLS IK  dz = J^T (J J^T + lam I)^-1 dx  reduces to

        dz ~ (u1 . dx / s1) v1

    i.e. the drag direction sets SIGN and MAGNITUDE only; the SHAPE of the
    latent response is always v1, and v1 is head yaw because that is the
    dominant mode of the footage.  "It knows direction" would then mean
    "it has one thing it can do and DLS reads off how much of it you asked for".

GATES (pre-registered, two-sided; the random-init decoder is the control)
    R1  response shape is fixed      effective dim of the response set <= 1.30
                                     CEILING IS 2.00, NOT 128: the n_dirs drag
                                     requests span only a 2-D subspace (one
                                     direction tiled across the pins), so the
                                     responses cannot span more than 2 whatever
                                     the Jacobian rank is.  A full-rank decoder
                                     measures 2.00 and median |cos| 2/pi=0.637.
                                     REVISION: the first version of this gate
                                     used |cos| >= 0.85 against a null of
                                     1/sqrt(128) = 0.088, which is the null for
                                     ISOTROPIC requests, not tiled ones.  The
                                     isotropic control measured 0.608 and killed
                                     it.  Effective dim has a known ceiling and
                                     no such confound.
    R2  pin error is rank-predicted  corr(measured err/|dx|,
                                          sqrt(1 - (u1.dxhat)^2)) >= 0.80
    R3  opposing two-pin request     achieved/requested <= 0.35 for the
                                     antisymmetric part  (<= means one DOF)

Nothing here trains, renders a video, or writes a checkpoint.  It reads a
model, finite-differences the decoder, and prints numbers.

    python3.13 rank1_probe.py --model model5_constQ.pt
    python3.13 rank1_probe.py --selftest          (no model needed)
"""

import argparse
import sys

import numpy as np


# --------------------------------------------------------------- backend ---

def load_decoder(path, device="cpu"):
    """load_splatvae returns (model, checkpoint) and takes no map_location.
    Packet centres come from ren.activate(dec(z)); the raw decoder output is
    PRE-activation, so its columns are not coordinates."""
    import torch
    try:
        from splat_trainer5 import load_splatvae
    except ImportError:
        from splat_trainer4q import load_splatvae
    m, _ck = load_splatvae(path)
    m.eval()
    with torch.no_grad():
        mu, _ = m.enc(torch.zeros(1, 3, m.ren.H, m.ren.H))
    zdim = int(mu.shape[1])

    def cent(z):
        with torch.no_grad():
            px, py, *_ = m.ren.activate(m.dec(torch.as_tensor(
                z, dtype=torch.float32)[None]).float())
        return torch.stack([px[0], py[0]], dim=1).numpy()
    return cent, zdim, m


def centroids(dec, z):
    """(N,2) packet centres. `dec` here is the closure load_decoder returns."""
    return dec(z)


def jacobian(dec, z, sel, eps=1e-2):
    """d(centre_x, centre_y)/dz for the selected packets, shape (2m, D)."""
    D = len(z)
    base = centroids(dec, z)[sel].ravel()
    J = np.empty((base.size, D))
    for i in range(D):
        zp = z.copy(); zp[i] += eps
        zm = z.copy(); zm[i] -= eps
        J[:, i] = (centroids(dec, zp)[sel].ravel()
                   - centroids(dec, zm)[sel].ravel()) / (2 * eps)
    return J


def dls(J, dx, lam=1e-3):
    """Damped least squares, the same solve splat_ragdoll uses."""
    m = J @ J.T
    m[np.diag_indices_from(m)] += lam * max(np.trace(m) / len(m), 1e-9)
    return J.T @ np.linalg.solve(m, dx)


# ----------------------------------------------------------------- probe ---

def probe(dec, zdim, z0, pin_xy, radius=0.10, drag=0.15, n_dirs=8, lam=1e-3):
    """Drag one pin in n_dirs directions; report the shape of the response."""
    c = centroids(dec, z0)
    sel = np.where(np.linalg.norm(c - np.asarray(pin_xy), axis=1) < radius)[0]
    if len(sel) < 3:
        sel = np.argsort(np.linalg.norm(c - np.asarray(pin_xy), axis=1))[:8]
    J = jacobian(dec, z0, sel)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)

    dzs, errs, proj = [], [], []
    for k in range(n_dirs):
        a = 2 * np.pi * k / n_dirs
        d = np.array([np.cos(a), np.sin(a)]) * drag
        dx = np.tile(d, len(sel))
        dz = dls(J, dx, lam)
        got = centroids(dec, z0 + dz)[sel].ravel() - centroids(dec, z0)[sel].ravel()
        dzs.append(dz / max(np.linalg.norm(dz), 1e-12))
        errs.append(np.linalg.norm(got - dx) / np.linalg.norm(dx))
        dxh = dx / np.linalg.norm(dx)
        proj.append(np.sqrt(max(0.0, 1.0 - float(U[:, 0] @ dxh) ** 2)))
    dzs = np.array(dzs)
    cos = np.abs(dzs @ dzs.T)
    iu = np.triu_indices(len(dzs), 1)
    sv = np.linalg.svd(dzs, compute_uv=False)
    ev = sv ** 2 / max((sv ** 2).sum(), 1e-30)
    eff = float(np.exp(-(ev * np.log(ev + 1e-30)).sum()))
    return dict(S=S, sel=sel, J=J, cos_med=float(np.median(cos[iu])),
                cos_min=float(cos[iu].min()), eff=eff,
                err=np.array(errs), proj=np.array(proj))


def two_pin(dec, z0, pins, drag=0.15, radius=0.10, lam=1e-3):
    """R3: ask two pins to move in OPPOSITE directions.  One DOF cannot."""
    c = centroids(dec, z0)
    sel, want = [], []
    for (p, sgn) in pins:
        s = np.argsort(np.linalg.norm(c - np.asarray(p), axis=1))[:6]
        sel.append(s)
        want.append(np.tile(np.array([sgn * drag, 0.0]), len(s)))
    sel_all = np.concatenate(sel)
    dx = np.concatenate(want)
    J = jacobian(dec, z0, sel_all)
    dz = dls(J, dx, lam)
    got = centroids(dec, z0 + dz)[sel_all].ravel() - centroids(dec, z0)[sel_all].ravel()
    # antisymmetric part: the bit a single global mode cannot deliver
    n0 = len(sel[0]) * 2
    anti_w = 0.5 * (dx[:n0] - dx[n0:])
    anti_g = 0.5 * (got[:n0] - got[n0:])
    return float(np.linalg.norm(anti_g) / max(np.linalg.norm(anti_w), 1e-12))


def report(name, r):
    S = r['S']
    ev = S ** 2 / max((S ** 2).sum(), 1e-12)
    rank = float(np.exp(-(ev * np.log(ev + 1e-30)).sum()))
    print(f"\n{name}")
    print(f"  singulars      {np.array2string(S[:6], precision=4)}")
    print(f"  participation  {rank:.2f}   (1.0 = one axis; 128 = isotropic)")
    print(f"  R1 response    effective dim {r['eff']:.3f}  (ceiling 2.00)   "
          f"median |cos| {r['cos_med']:.4f}")
    c = np.corrcoef(r['err'], r['proj'])[0, 1] if r['proj'].std() > 1e-9 else np.nan
    print(f"  R2 err vs rank corr {c:+.4f}   "
          f"err/|dx| {r['err'].min():.3f}-{r['err'].max():.3f}")
    return r['eff'], c


# -------------------------------------------------------------- selftest ---

def selftest():
    """Two-sided: a planted rank-1 decoder must pass R1; an isotropic one must fail."""
    ok = True

    def chk(n, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}  {d}")

    print("SELFTEST")
    rng = np.random.default_rng(0)
    D, N = 128, 64
    base = rng.random((N, 11))

    for tag, rank in (("planted rank-1", 1), ("isotropic rank-128", 128)):
        W = rng.standard_normal((N * 2, D))
        if rank == 1:
            u, s, vt = np.linalg.svd(W, full_matrices=False)
            W = np.outer(u[:, 0], vt[0]) * s[0]

        def dec(z, W=W):
            out = base.copy()
            out[:, :2] += (W @ z).reshape(N, 2) * 0.01
            return out

        r = probe(dec, D, np.zeros(D), pin_xy=base[0, :2], radius=1e9, drag=0.05)
        eff, _ = report(f"  {tag}", r)
        if rank == 1:
            chk("R1 fires on a rank-1 decoder", eff <= 1.30, f"eff dim {eff:.3f}")
        else:
            chk("R1 does NOT fire on an isotropic decoder", eff > 1.60,
                f"eff dim {eff:.3f}  (2.00 is the analytic ceiling)")

    print(f"SELFTEST {'PASS' if ok else 'FAIL'}\n")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='')
    p.add_argument('--image', type=str, default='',
                   help='encode this frame for z0; else z0 = 0')
    p.add_argument('--drag', type=float, default=0.15)
    p.add_argument('--radius', type=float, default=0.10)
    p.add_argument('--lam', type=float, default=1e-3)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.model:
        print("need --model (or --selftest)"); sys.exit(2)

    dec, zdim, m = load_decoder(a.model)
    z0 = np.zeros(zdim)
    if a.image:
        import torch, cv2
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"cannot read --image {a.image}")
        S = m.ren.H
        img = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
        with torch.no_grad():
            z0 = m.enc(x)[0][0].numpy()
        print(f"z0 from {a.image}: |mu| = {np.linalg.norm(z0):.3f}")

    c = centroids(dec, z0)
    print(f"model {a.model}: {len(c)} packets, latent {zdim}")

    for pin, nm in ((c.mean(0) + [-0.18, 0.0], "left cheek"),
                    (c.mean(0) + [+0.18, 0.0], "right cheek"),
                    (c.mean(0) + [0.0, -0.15], "brow")):
        r = probe(dec, zdim, z0, pin, radius=a.radius, drag=a.drag, lam=a.lam)
        eff, corr = report(f"pin @ {nm}  ({len(r['sel'])} packets)", r)
        print(f"  R1 [{'V' if eff <= 1.30 else 'K'}]   "
              f"R2 [{'V' if (corr == corr and corr >= 0.80) else 'K'}]")

    ratio = two_pin(dec, z0,
                    [(c.mean(0) + [-0.18, 0.0], +1.0),
                     (c.mean(0) + [+0.18, 0.0], -1.0)],
                    drag=a.drag, radius=a.radius, lam=a.lam)
    print(f"\nR3 opposing two-pin: antisymmetric achieved/requested = {ratio:.4f}")
    print(f"R3 [{'V' if ratio <= 0.35 else 'K'}]   "
          f"[V] = one DOF, the drag only picks how much of it")


if __name__ == '__main__':
    main()
