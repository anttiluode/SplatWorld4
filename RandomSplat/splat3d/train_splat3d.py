"""
train_splat3d.py — teach the splats to see depth from a single frame
================================================================================
Trains Splat3D on rotating objects (obj_world). The depth target is the renderer's
z-buffer — free, exact supervision. Loss has three parts:

    rgb   : reconstruct the image          (MSE)          — the packets must still LOOK right
    depth : predict the inverse-depth map   (L1)          — the z's must be right
    edge  : match depth gradients           (L1 on grads) — sharp at occlusion boundaries

This is the experiment: can a sparse Gabor code, with a depth per packet, learn to
read 3D structure off a single still image — the thing the motion-residual organ
could only do while moving?

HONESTY (read before believing a trained model):
  - I verified shapes, forward/backward, and that the depth loss DROPS on a tiny
    CPU run (`--smoke`). I did NOT train it to convergence (no GPU here). Run the
    real thing on your 3060.
  - SIM-TO-REAL is the real risk, not convergence. A model trained only on rotating
    primitives will learn depth ON THOSE. It will NOT automatically see your living
    room — synthetic spheres are not faces and rooms. Domain randomization (textures,
    lighting, backgrounds) is on to help, and `--obj_dir` lets you train on real
    object meshes, but closing sim-to-real for your webcam likely needs real RGBD
    data or distillation from an existing depth model (see the README).

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""

import os, glob, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from splat3d import Splat3D
import obj_world


def make_batch(bs, size, rng, obj_paths):
    rgb = np.zeros((bs, 3, size, size), np.float32)
    dep = np.zeros((bs, size, size), np.float32)
    msk = np.zeros((bs, size, size), np.float32)
    for i in range(bs):
        r, d, m = obj_world.sample(size, rng, obj_paths)
        rgb[i] = np.transpose(r, (2, 0, 1))
        dep[i] = d
        msk[i] = m.astype(np.float32)
    return torch.from_numpy(rgb), torch.from_numpy(dep), torch.from_numpy(msk)


def grad_xy(t):
    """image gradients for the edge term. t: (B,H,W)."""
    gx = t[:, :, 1:] - t[:, :, :-1]
    gy = t[:, 1:, :] - t[:, :-1, :]
    return gx, gy


def depth_losses(pred, target, mask):
    # foreground-weighted L1 (object pixels matter most, bg still pulled to far)
    w = mask * 1.0 + (1 - mask) * 0.3
    l1 = (w * (pred - target).abs()).mean()
    # edge / gradient matching (sharp occlusion boundaries)
    pgx, pgy = grad_xy(pred); tgx, tgy = grad_xy(target)
    edge = (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()
    return l1, edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/splat3d")
    ap.add_argument("--obj_dir", default="", help="folder of .obj files; empty = procedural primitives")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--num_packets", type=int, default=512)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_rgb", type=float, default=1.0)
    ap.add_argument("--w_depth", type=float, default=2.0)
    ap.add_argument("--w_edge", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", default="")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run; proves depth loss drops")
    args = ap.parse_args()

    if args.smoke:
        args.image_size, args.num_packets, args.batch = 48, 128, 8
        args.steps, args.amp, args.chunk = 60, False, 64
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    obj_paths = sorted(glob.glob(os.path.join(args.obj_dir, "*.obj"))) if args.obj_dir else None
    print(f"device={dev} size={args.image_size} packets={args.num_packets} "
          f"batch={args.batch} objs={'procedural' if not obj_paths else len(obj_paths)}")

    model = Splat3D(args.image_size, args.latent, args.num_packets, args.chunk).to(dev)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev)); print("resumed", args.resume)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    rng = np.random.default_rng(0)

    fixed = make_batch(min(8, args.batch), args.image_size, np.random.default_rng(7), obj_paths)
    fixed = [t.to(dev) for t in fixed]

    first_depth = None
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rgb, dep, msk = make_batch(args.batch, args.image_size, rng, obj_paths)
        rgb, dep, msk = rgb.to(dev), dep.to(dev), msk.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_rgb, pred_dep, _ = model(rgb)
            l_rgb = F.mse_loss(pred_rgb, rgb)
            l_dep, l_edge = depth_losses(pred_dep, dep, msk)
            loss = args.w_rgb * l_rgb + args.w_depth * l_dep + args.w_edge * l_edge
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update()

        if first_depth is None:
            first_depth = l_dep.item()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:6d}  rgb {l_rgb.item():.4f}  depth {l_dep.item():.4f}  "
                  f"edge {l_edge.item():.4f}  {(time.time()-t0)/step*1000:.0f}ms/step")
        if not args.smoke and step % args.save_every == 0:
            save_grid(model, fixed, args.out, step, dev)
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))

    if args.smoke:
        print(f"\nsmoke: depth loss {first_depth:.4f} -> {l_dep.item():.4f}  "
              f"({'DROPPED — it is learning depth' if l_dep.item() < first_depth*0.9 else 'no clear drop'})")
    else:
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
        save_grid(model, fixed, args.out, args.steps, dev)
        print("done. model.pt + grids in", args.out)


@torch.no_grad()
def save_grid(model, fixed, out, step, dev):
    import cv2
    rgb, dep, msk = fixed
    model.eval()
    pred_rgb, pred_dep, _ = model(rgb)
    model.train()
    rows = []
    for i in range(rgb.shape[0]):
        r = (rgb[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        pr = (pred_rgb[i].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)[..., ::-1]
        dg = cv2.applyColorMap((dep[i].cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        dp = cv2.applyColorMap((pred_dep[i].cpu().numpy().clip(0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        rows.append(np.hstack([r, pr, dg, dp]))
    cv2.imwrite(os.path.join(out, f"grid_{step:06d}.png"), np.vstack(rows))


if __name__ == "__main__":
    main()