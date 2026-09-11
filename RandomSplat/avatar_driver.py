#!/usr/bin/env python3
# =============================================================================
# avatar_driver.py — the tiny avatar, assembled from certified parts
#
# WHAT IS CERTIFIED (splat_avatar_gate --pairs 8, 2026-07-21, ledger on disk):
#   R3 8/8  transport mid-path frames are SHARPER than the decoder's own
#           latent road at every t (pointwise ratio >= 1.0)
#   R4 8/8  transport stays within 0.35*gap of the decoder road even across
#           full identity swaps (ratio 0.02-0.06 typical, 0.14 worst at the
#           largest gap) -> keyframe spacing has ~an order of magnitude of
#           headroom over any sane driving rate
#   R5 8/8  scrambling target phases breaks road agreement ~10x -> coherent
#           phase is the mechanism, not a coincidence
#   R2 4/8  HONEST NOTE: between same-model faces the lerp "fire" is mild
#           (phases correlate). Transport's in-manifold value is amplitude
#           discipline + sharpness; its out-of-manifold value is not dying.
#
# WHAT THIS SCRIPT ADDS (not yet certified — this is the demo, not the proof):
#   a PURSUIT scheme for live driving. Keyframes (encoded z -> packets) arrive
#   at low rate; every display frame, the current packet state takes a
#   fractional transport step toward the latest keyframe:
#     geometry: lerp fraction alpha        phase: shortest-arc fraction alpha
#     magnitude: lerp fraction alpha       theta: shortest-arc fraction alpha
#   This is exponential smoothing along the transport geodesic — smooth
#   pursuit, no fixed segments, no scheduling.
#
# MODES
#   webcam (default):
#     python avatar_driver.py --model runs/splat2/model2.pt
#     keys: m  cycle DIRECT (encode+render every frame) / LERP pursuit /
#              PHASE pursuit  — compare jitter & sharpness live
#           k/j  keyframe interval up/down (frames between encoder calls)
#           a/z  pursuit alpha up/down
#           n    toggle input normalization (fights the black-head domain gap)
#           q    quit
#   latent walk (no webcam needed; also the shareable-GIF generator):
#     python avatar_driver.py --walk --record 240
#     random-walk z keyframes at low rate, transport pursuit at display rate,
#     writes avatar_walk.gif. This is the "surf that never melts" artifact.
#
# HONESTY: webcam mode needs a camera and a screen — smoke-tested here only
# for import/arg wiring. --walk --record was run end-to-end in the sandbox on
# a random-weight checkpoint (pipeline verified, output GIF written). Frame
# rates and visual quality on your trained model are yours to measure.
# =============================================================================
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Splat_trainer2 as ST


# ---------------------------------------------------------------- rendering
def render_image(ren, P):
    px, py, sigma, theta, freq, coeff = P
    out = None
    for i in range(0, ren.N, ren.chunk):
        sl = slice(i, i + ren.chunk)
        c = ren._chunk(px[:, sl], py[:, sl], sigma[:, sl],
                       theta[:, sl], freq[:, sl], coeff[:, sl])
        out = c if out is None else out + c
    return torch.sigmoid(out)


# ---------------------------------------------------------------- pursuit
def _arc_step(a, b, alpha):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + alpha * d


def pursue(P, T, alpha, mode):
    """One fractional step of current state P toward target keyframe T.
    mode 'phase': transport geodesic (mag-lerp + shortest-arc phase).
    mode 'lerp' : straight line in (a,b) (the baseline, for live comparison)."""
    px, py, s, th, f, c = P
    pxT, pyT, sT, thT, fT, cT = T
    L = lambda a, b: a + alpha * (b - a)
    px2, py2, s2, f2 = L(px, pxT), L(py, pyT), L(s, sT), L(f, fT)
    th2 = _arc_step(th, thT, alpha)
    if mode == "lerp":
        c2 = L(c, cT)
    else:
        a_, b_ = c[..., 0], c[..., 1]
        aT, bT = cT[..., 0], cT[..., 1]
        m = torch.sqrt(a_ * a_ + b_ * b_ + 1e-12)
        mT = torch.sqrt(aT * aT + bT * bT + 1e-12)
        ph = torch.atan2(b_, a_)
        phT = torch.atan2(bT, aT)
        m2 = L(m, mT)
        ph2 = _arc_step(ph, phT, alpha)
        c2 = torch.stack([m2 * torch.cos(ph2), m2 * torch.sin(ph2)], dim=-1)
    return (px2, py2, s2, th2, f2, c2)


def clone_params(P):
    return tuple(t.clone() for t in P)


# ---------------------------------------------------------------- input prep
def normalize_crop(x, tgt_mean=0.52, tgt_std=0.26):
    """Fight the domain gap: push webcam crop stats toward face-dataset-ish
    brightness/contrast. x: float array HxWx3 in [0,1]."""
    m, s = x.mean(), x.std() + 1e-6
    return np.clip((x - m) / s * tgt_std + tgt_mean, 0, 1)


# ---------------------------------------------------------------- walk mode
def walk(args, model, dev):
    ren = model.ren
    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(1, ST.LATENT, generator=g).to(dev)

    def keyframe(z):
        with torch.no_grad():
            return ren.activate(model.dec(z).float())

    T = keyframe(z)
    P = clone_params(T)
    frames = []
    n = args.record if args.record else 240
    t0 = time.time()
    for f in range(n):
        if f % args.kf == 0:                       # new keyframe: step z
            step = torch.randn(1, ST.LATENT, generator=g).to(dev)
            z = z + args.walk_step * step
            z = z * min(1.0, args.z_max / (z.norm() + 1e-9))   # stay in core
            T = keyframe(z)
        with torch.no_grad():
            P = pursue(P, T, args.alpha, "phase")
            img = render_image(ren, P)
        x = (img[0].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        if args.scale > 1:
            x = np.kron(x, np.ones((args.scale, args.scale, 1), dtype=np.uint8))
        frames.append(x)
    fps = n / (time.time() - t0)
    print(f"{n} frames at {fps:.1f} fps (render+pursuit, {dev})")
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(args.gif, save_all=True, append_images=ims[1:],
                duration=40, loop=0)
    print(f"gif -> {args.gif}  (keyframe every {args.kf} frames, "
          f"alpha {args.alpha}, |z| capped at {args.z_max})")


# ---------------------------------------------------------------- webcam mode
def live(args, model, dev):
    import cv2 as cv
    ren = model.ren
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("webcam failed to open (use --walk for the no-camera demo)")
    modes, mi = ["direct", "lerp", "phase"], 2
    kf, alpha, norm = args.kf, args.alpha, True
    P = T = None
    f = 0
    win = "TINY AVATAR  cam | avatar   (m=mode k/j=keyframe a/z=alpha n=norm q=quit)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    t_last, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]; s = min(h, w)
        crop = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        x = cv.resize(crop, (ren.H, ren.W))[:, :, ::-1].astype(np.float32) / 255.0
        if norm:
            x = normalize_crop(x)
        xt = torch.from_numpy(np.ascontiguousarray(
            x.transpose(2, 0, 1)))[None].to(dev)

        need_encode = (modes[mi] == "direct") or (f % kf == 0) or (T is None)
        if need_encode:
            with torch.no_grad():
                mu, _ = model.enc(xt)
                T = ren.activate(model.dec(mu).float())
            if P is None:
                P = clone_params(T)

        with torch.no_grad():
            if modes[mi] == "direct":
                P = clone_params(T)
            else:
                P = pursue(P, T, alpha, modes[mi])
            img = render_image(ren, P)

        def to8(t_):
            im = (t_[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255)
            return cv.resize(im.astype(np.uint8)[:, :, ::-1], (384, 384),
                             interpolation=cv.INTER_CUBIC)
        cam = cv.resize(crop, (384, 384))
        panel = np.concatenate([cam, to8(img)], 1)
        now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now
        cv.putText(panel, f"{modes[mi]}  kf:{kf}  alpha:{alpha:.2f}  "
                   f"norm:{'on' if norm else 'off'}  {fps:.0f}fps",
                   (10, 374), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1,
                   cv.LINE_AA)
        cv.imshow(win, panel)
        f += 1
        k = cv.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('m'): mi = (mi + 1) % 3
        elif k == ord('k'): kf = min(60, kf + 1)
        elif k == ord('j'): kf = max(1, kf - 1)
        elif k == ord('a'): alpha = min(1.0, alpha + 0.05)
        elif k == ord('z'): alpha = max(0.05, alpha - 0.05)
        elif k == ord('n'): norm = not norm
    cap.release(); cv.destroyAllWindows()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./runs/splat2/model2.pt")
    ap.add_argument("--walk", action="store_true",
                    help="no-webcam latent-walk demo")
    ap.add_argument("--record", type=int, default=0,
                    help="walk mode: number of frames to write to --gif")
    ap.add_argument("--gif", default="avatar_walk.gif")
    ap.add_argument("--kf", type=int, default=8,
                    help="frames between keyframes (encoder calls)")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="pursuit fraction per frame")
    ap.add_argument("--walk_step", type=float, default=2.5,
                    help="walk mode: z step size per keyframe")
    ap.add_argument("--z_max", type=float, default=12.0,
                    help="walk mode: |z| cap (stay in the core, out of the fire)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location="cpu")
    model = ST.SplatVAE(ck["image_size"], ck["num_packets"])
    model.load_state_dict(ck["sd"]); model.eval().to(dev)
    print(f"model {ck['image_size']}px / {ck['num_packets']} packets on {dev}")

    if args.walk:
        walk(args, model, dev)
    else:
        live(args, model, dev)


if __name__ == "__main__":
    main()
