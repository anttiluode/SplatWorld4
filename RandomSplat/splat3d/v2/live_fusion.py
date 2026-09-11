"""
live_fusion.py — a depth BELIEF: motion sharpens it, stillness can't erase it
================================================================================
The two depth organs fail in opposite ways:

  the_splat_3d.py  (motion residual)  — REAL depth, but only while you move; it
                                        flickers to nothing the instant you hold
                                        still (no motion -> no parallax -> flat).
  splat3d / model.pt (learned prior)  — STABLE every frame, but on a real scene
                                        it hallucinates (the webcam frame: a near-
                                        uniform orange map — no real 3D).

This fuses them the way the flower fused blur and reality: prediction + a
precision-weighted correction. It maintains a depth BELIEF D and, each frame:

  1. PREDICT   warp D forward by the estimated global motion (the_anchor's boil),
               so the belief stays registered to the scene as the camera pans.
  2. CORRECT   measure motion-parallax depth M (the_splat_3d residual) and pull D
               toward it, weighted by CONFIDENCE c = (how much the camera moved)
               x (how much texture is here). Still camera -> c~0 -> D is HELD, not
               wiped. Moving camera -> c up -> D sharpens toward real parallax.
  3. ANCHOR    (optional) when confidence is chronically low, pull D gently toward
               the learned prior P from model.pt — the "eyes-closed" fallback.

So: stable through stillness (the property you liked), real structure from your
own motion (the property the sim prior can't give), and the learned model is a
soft attractor, not the load-bearing part. It does NOT depend on the weak prior.

HONEST NEIGHBOURHOOD (used, not claimed new): this is a Kalman-flavoured temporal
depth filter / monocular depth fusion (cf. DTAM-style dense tracking, depth
filtering). The contribution is only the framing and wiring: a leaky predictive
depth belief in the predictive-coding line, joined to the_anchor (global motion),
the_video_tensor (leaky hold), the_splat_3d (the residual), and splat3d (the prior).

HONEST LIMITS: RELATIVE depth, not metric. Per-frame parallax is normalized, so
the belief is a relative ordering, not millimetres. Independent motion (a hand
waving) reads as "near" — same illusion the visual system has. The learned-prior
anchor is currently weak on real scenes (sim-to-real); train it better and it
plugs straight in. Verified here only on the synthetic pan-then-freeze meter
(`--smoke`); the live path reuses the verified motion functions.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""

import os, time, argparse
import numpy as np
import cv2


# ----------------------------------------------------------------------
# motion organ (verified in the_splat_3d.py) — flow, global prior, residual
# ----------------------------------------------------------------------
def dense_flow(prev_gray, cur_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 25, 3, 7, 1.5, 0)


def fit_global_motion(flow, irls_iters=2):
    H, W, _ = flow.shape
    ys, xs = np.mgrid[0:H, 0:W]
    A = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], -1).astype(np.float64)
    fx, fy = flow[..., 0].ravel(), flow[..., 1].ravel()
    w = np.ones(H * W)
    for _ in range(max(1, irls_iters)):
        Aw = A * w[:, None]
        cx, *_ = np.linalg.lstsq(Aw, fx * w, rcond=None)
        cy, *_ = np.linalg.lstsq(Aw, fy * w, rcond=None)
        r = np.sqrt((fx - A @ cx) ** 2 + (fy - A @ cy) ** 2)
        s = np.median(r) + 1e-6
        w = 1.0 / (1.0 + (r / (2.0 * s)) ** 2)
    return np.stack([(A @ cx).reshape(H, W), (A @ cy).reshape(H, W)], -1)


def motion_depth(flow, pred_flow):
    """residual magnitude -> per-frame relative inverse-depth in [0,1]."""
    res = flow - pred_flow
    mag = cv2.GaussianBlur(np.linalg.norm(res, axis=-1), (0, 0), 1.5)
    hi = np.percentile(mag, 98) + 1e-6
    return np.clip(mag / hi, 0, 1)


def warp_map(img, flow):
    """warp img forward by flow (predict step). img can be HxW or HxWxC."""
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    mx = (xs - flow[..., 0]).astype(np.float32)
    my = (ys - flow[..., 1]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def texture_validity(gray):
    """flow is trustworthy only where there is structure. -> [0,1] map."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)
    hi = np.percentile(m, 90) + 1e-6
    return np.clip(m / hi, 0, 1)


def colorize(d01):
    return cv2.applyColorMap((np.clip(d01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


# ----------------------------------------------------------------------
# the belief: predict -> correct -> (optional) anchor
# ----------------------------------------------------------------------
class DepthBelief:
    def __init__(self, size, alpha=0.5, motion_thresh=0.8, prior_pull=0.0):
        self.D = np.full((size, size), 0.5, np.float32)   # start: flat, unknown
        self.alpha = alpha            # correction rate when fully confident
        self.motion_thresh = motion_thresh  # px of global motion for full confidence
        self.prior_pull = prior_pull  # weight toward learned prior when still
        self.inited = False

    def step(self, prev_gray, cur_gray, prior=None):
        flow = dense_flow(prev_gray, cur_gray)
        gflow = fit_global_motion(flow)

        # 1. PREDICT: carry the belief with the camera
        self.D = warp_map(self.D, gflow)

        # 2. CORRECT: motion parallax, weighted by how much we moved x texture
        M = motion_depth(flow, gflow)
        cam_motion = float(np.mean(np.linalg.norm(gflow, axis=-1)))
        c_global = np.clip(cam_motion / self.motion_thresh, 0, 1)
        # M is already a per-frame relative inverse-depth in [0,1]; blend directly.
        c = (self.alpha * c_global) * texture_validity(cur_gray)   # per-pixel gain
        if c_global > 0.05:
            self.D = (1 - c) * self.D + c * M
            self.inited = True

        # 3. ANCHOR: when chronically still, drift gently toward the learned prior
        if prior is not None and self.prior_pull > 0:
            self.D = self.D + self.prior_pull * (1 - c_global) * (np.clip(prior, 0, 1) - self.D)

        return self.D, M, c_global


# ----------------------------------------------------------------------
# the meter: pan, then FREEZE — does the belief survive the freeze?
# ----------------------------------------------------------------------
def _texture(size, seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 255, (size, size, 3)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), 2.2)
    for _ in range(12):
        cv2.circle(t, (int(rng.integers(0, size)), int(rng.integers(0, size))),
                   int(rng.integers(8, 22)),
                   tuple(int(v) for v in rng.integers(40, 215, 3)), -1)
    return np.clip(cv2.GaussianBlur(t, (0, 0), 1.0), 0, 255).astype(np.uint8)


def _shift(img, dx):
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def run_meter(size=256, pan_frames=6, hold_frames=6, far_step=1.5, near_step=4.5,
              save_dir=None):
    bg, fg = _texture(size, 1), _texture(size, 2)
    box = (size // 4, size // 4, 3 * size // 4, 3 * size // 4)
    x0, y0, x1, y1 = box

    def frame(bg_dx, fg_dx):
        out = _shift(bg, bg_dx).copy()
        out[y0:y1, x0:x1] = _shift(fg, fg_dx)[y0:y1, x0:x1]
        return out

    # build sequence: pan right, then freeze
    seq, bgx, fgx = [], 0.0, 0.0
    for _ in range(pan_frames):
        seq.append(frame(bgx, fgx)); bgx += far_step; fgx += near_step
    last = frame(bgx, fgx)
    for _ in range(hold_frames):
        seq.append(last.copy())     # identical frames = perfectly still

    m = 14
    core = np.zeros((size, size), bool); core[y0 + m:y1 - m, x0 + m:x1 - m] = True
    out = np.zeros((size, size), bool)
    out[:m] = out[-m:] = out[:, :m] = out[:, -m:] = True

    belief = DepthBelief(size, alpha=0.6, motion_thresh=1.2)
    prev = cv2.cvtColor(seq[0], cv2.COLOR_BGR2GRAY)
    rows, sep_belief, sep_motion = [], [], []
    for i in range(1, len(seq)):
        cur = cv2.cvtColor(seq[i], cv2.COLOR_BGR2GRAY)
        D, M, cg = belief.step(prev, cur)
        sb = float(np.median(D[core]) - np.median(D[out]))
        sm = float(np.median(M[core]) - np.median(M[out]))
        sep_belief.append(sb); sep_motion.append(sm)
        if save_dir and i in (pan_frames - 1, len(seq) - 1):
            rows.append(np.hstack([seq[i], colorize(M), colorize(D)]))
        prev = cur

    end_pan_b, end_pan_m = sep_belief[pan_frames - 2], sep_motion[pan_frames - 2]
    end_hold_b, end_hold_m = sep_belief[-1], sep_motion[-1]

    print("\n=== live_fusion meter: pan, then FREEZE (near>far separation) ===")
    print(f"  scene: near patch slides {near_step}px/frame, background {far_step}px/frame")
    print(f"  panning ({pan_frames} frames), then frozen ({hold_frames} identical frames)\n")
    print(f"  end of PAN :  belief sep = {end_pan_b:+.3f}   motion-only sep = {end_pan_m:+.3f}")
    print(f"  end of FREEZE: belief sep = {end_hold_b:+.3f}   motion-only sep = {end_hold_m:+.3f}")
    print(f"\n  motion-only collapses when frozen: {end_pan_m:+.3f} -> {end_hold_m:+.3f}")
    print(f"  belief HOLDS through the freeze:    {end_pan_b:+.3f} -> {end_hold_b:+.3f}")
    ok = (end_pan_b > 0.05) and (end_hold_b > 0.6 * end_pan_b) and (end_hold_m < end_pan_m)
    print(f"  VERDICT: {'PASS - the belief persists through stillness' if ok else 'FAIL'}")
    print("=================================================================\n")

    if save_dir and rows:
        os.makedirs(save_dir, exist_ok=True)
        panel = np.vstack(rows)
        for j, lab in enumerate(["scene", "motion-only (flickers)", "belief (holds)"]):
            cv2.putText(panel, lab, (j * size + 5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(panel, "top: end of pan", (5, size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, "bottom: end of freeze", (5, 2 * size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(save_dir, "fusion_meter.png"), panel)
        print(f"  wrote {save_dir}/fusion_meter.png")
    return ok


# ----------------------------------------------------------------------
# live: webcam -> real | motion-only | fused belief (+ optional prior)
# ----------------------------------------------------------------------
def run_live(cam=0, size=256, model_path="", prior_pull=0.15):
    prior_model = None
    if model_path:
        try:
            import torch
            from splat3d import Splat3D
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # NOTE: match these to your trained checkpoint
            prior_model = Splat3D(56, 128, 192).to(dev).eval()
            prior_model.load_state_dict(torch.load(model_path, map_location=dev))
            prior_model._dev = dev
            print(f"learned prior loaded from {model_path} (soft anchor, pull={prior_pull})")
        except Exception as e:
            print(f"prior not loaded ({e}); running motion+belief only")
            prior_model = None

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print(f"cannot open camera {cam} (OBS virtual cam is often --cam 1)"); return
    print("fusion online. MOVE sideways to write depth into the belief; "
          "hold still and watch it PERSIST. Press 'q'.")

    belief = DepthBelief(size, alpha=0.5, motion_thresh=0.8,
                         prior_pull=prior_pull if prior_model else 0.0)
    prev_gray, prev_bgr = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        s = min(h, w)
        frame = cv2.resize(frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2],
                           (size, size))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            P = None
            if prior_model is not None:
                import torch
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (56, 56))
                t = torch.from_numpy(small).float().div(255).permute(2, 0, 1)[None]
                t = t.to(prior_model._dev)
                with torch.no_grad():
                    _, dep, _ = prior_model(t)
                P = cv2.resize(dep[0].cpu().numpy(), (size, size))
            D, M, cg = belief.step(prev_gray, gray, prior=P)
            panel = np.hstack([frame, colorize(M), colorize(D)])
            for j, lab in enumerate(["1.retina", "2.motion-only", "3.belief (held+sharpened)"]):
                cv2.putText(panel, lab, (j * size + 5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(panel, f"cam motion {cg:.2f}", (2 * size + 5, size - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow("live_fusion - depth belief", panel)

        prev_gray, prev_bgr = gray, frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="pan-then-freeze meter, no camera")
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--model", type=str, default="", help="optional Splat3D model.pt as soft prior anchor")
    ap.add_argument("--prior_pull", type=float, default=0.15)
    args = ap.parse_args()
    if args.smoke:
        run_meter(min(args.image_size, 320), save_dir=args.save_dir or None)
    else:
        run_live(args.cam, args.image_size, args.model, args.prior_pull)


if __name__ == "__main__":
    main()
