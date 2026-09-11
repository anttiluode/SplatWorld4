"""
splat3d.py — Gabor wave-packets that carry depth (one code, two readouts)
================================================================================
`the_splat` proved an image is a sparse sum of Gabor wave-packets. `live_cortex`
let reality supply the phase. `the_splat_3d` read depth off motion parallax but
only while moving. This is the learned version: the splat decoder now emits, for
each packet, a DEPTH z in addition to its 2D appearance params, and the same
packet set is read out two ways —

    APPEARANCE  (the Gabor splatter)   packets -> RGB        "what it looks like"
    DEPTH       (soft-occlusion splat) packets -> inv-depth  "how far each bit is"

The position+envelope of a packet are SHARED across both readouts: the packet that
explains a region of the image also declares that region's depth. One sparse code,
two views of the world. Trained on rotating objects (obj_world.py) where the depth
label is the free z-buffer.

WHY DETERMINISTIC, NOT A VAE: the whole blur lesson was that SAMPLING a prior with
nothing to look at forces regression-to-the-mean (blur). Here we are not generating
from nothing — we have a real image in and a real depth target out. So we drop the
VAE sampling and KL: this is a supervised perception network, and sampling would
only reintroduce the exact blur we are trying to escape. Reality (the z-buffer)
supplies the target the way the webcam supplied the phase.

DEPTH RENDER & OCCLUSION (honest): each packet i covers pixels with its Gaussian
envelope a_i(p) and owns a depth z_i in [0,1] (1 = nearest). Near packets must
occlude far ones, so we weight by depth:  w_i(p) = a_i(p) * exp(beta * z_i), and
    depth(p) = sum_i w_i z_i / (sum_i w_i + eps_bg)
with a small background term at z=0 (far) for uncovered pixels. The exp(beta*z)
makes nearer packets dominate — a SOFT z-buffer, differentiable, no sorting. It is
an approximation of occlusion, not a true depth test; stated plainly.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""

import math
import torch
import torch.nn as nn

# per-packet params: dpx,dpy,ls(sigma),th,lf(freq), (a,b)x3 chans, dz(depth) = 12
K = 12


class GaborDepthRenderer(nn.Module):
    """Renders both RGB (Gabor) and inverse-depth (soft-occlusion) from packets."""

    def __init__(self, image_size=64, num_packets=512, chunk=64, depth_beta=6.0):
        super().__init__()
        self.H = self.W = image_size
        self.N = num_packets
        self.chunk = chunk
        self.depth_beta = depth_beta
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None])
        self.register_buffer("GY", gy[None, None])
        side = int(math.ceil(math.sqrt(num_packets)))
        ax = torch.linspace(0.08, 0.92, side)
        anch = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:num_packets]
        anch = torch.clamp(anch, 1e-3, 1 - 1e-3)
        self.register_buffer("anchor_logit", torch.log(anch / (1 - anch)))

    def activate(self, raw):
        a_px = self.anchor_logit[:, 0][None]
        a_py = self.anchor_logit[:, 1][None]
        px = torch.sigmoid(a_px + raw[..., 0])
        py = torch.sigmoid(a_py + raw[..., 1])
        sigma = 0.012 + 0.14 * torch.sigmoid(raw[..., 2])
        theta = raw[..., 3]
        freq = 1.0 + 15.0 * torch.sigmoid(raw[..., 4])
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)
        z = torch.sigmoid(raw[..., 11])                 # depth in [0,1], 1 = near
        return px, py, sigma, theta, freq, coeff, z

    def forward(self, raw):
        px, py, sigma, theta, freq, coeff, z = self.activate(raw.float())
        B = raw.shape[0]
        rgb = torch.zeros(B, 3, self.H, self.W, device=raw.device)
        # depth accumulators
        wsum = torch.full((B, self.H, self.W), 1e-3, device=raw.device)   # bg weight (z=0)
        wzsum = torch.zeros(B, self.H, self.W, device=raw.device)
        for i in range(0, self.N, self.chunk):
            sl = slice(i, i + self.chunk)
            px_ = px[:, sl, None, None]; py_ = py[:, sl, None, None]
            s_ = sigma[:, sl, None, None]; th = theta[:, sl, None, None]
            f_ = freq[:, sl, None, None]; z_ = z[:, sl, None, None]
            dx = self.GX - px_; dy = self.GY - py_
            xr = dx * torch.cos(th) + dy * torch.sin(th)
            env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))   # (B,n,H,W)
            cos = torch.cos(2 * math.pi * f_ * xr)
            sin = torch.sin(2 * math.pi * f_ * xr)
            for c in range(3):
                a = coeff[:, sl, c, 0][..., None, None]
                b = coeff[:, sl, c, 1][..., None, None]
                rgb[:, c] = rgb[:, c] + (env * (a * cos - b * sin)).sum(1)
            # depth: soft-occlusion weight = env * exp(beta*z)
            w = env * torch.exp(self.depth_beta * z_)              # (B,n,H,W)
            wsum = wsum + w.sum(1)
            wzsum = wzsum + (w * z_).sum(1)
        rgb = torch.sigmoid(rgb)
        depth = wzsum / wsum                                       # (B,H,W) in [0,1]
        return rgb, depth


class Encoder(nn.Module):
    def __init__(self, image_size=64, latent=128, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], 3, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc = nn.Linear(self.flat, latent)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent=128, num_packets=512, hidden=512):
        super().__init__()
        self.N = num_packets
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, num_packets * K))
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].weight.data *= 0.1

    def forward(self, z):
        return self.net(z).view(-1, self.N, K)


class Splat3D(nn.Module):
    """image -> latent -> packets(with depth) -> (rgb, inv-depth)."""

    def __init__(self, image_size=64, latent=128, num_packets=512, chunk=64):
        super().__init__()
        self.enc = Encoder(image_size, latent)
        self.dec = Decoder(latent, num_packets)
        self.ren = GaborDepthRenderer(image_size, num_packets, chunk)
        self.latent = latent

    def forward(self, x):
        packets = self.dec(self.enc(x))
        rgb, depth = self.ren(packets)
        return rgb, depth, packets


if __name__ == "__main__":
    # shape + forward/backward smoke
    m = Splat3D(image_size=64, num_packets=256, chunk=64)
    x = torch.rand(2, 3, 64, 64)
    rgb, depth, pk = m(x)
    print("rgb", tuple(rgb.shape), "depth", tuple(depth.shape), "packets", tuple(pk.shape))
    loss = rgb.mean() + depth.mean()
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print("backward ok, grad-sum %.3f" % g)
    print("depth range [%.3f, %.3f]" % (depth.min().item(), depth.max().item()))