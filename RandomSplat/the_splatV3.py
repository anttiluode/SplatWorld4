"""
the_splatV3.py  —  FLOW-PROBES: the cortex reads where things GO, not what
                   color they are.

PerceptionLab / Antti Luode (Helsinki), with Claude (Fable 5). July 2026.

    Do not hype. Do not lie. Just show.

WHY V3 EXISTS
-------------
V2live held a multi-D latent from K pointwise COLOR probes. It worked, but two
findings — one from the sandbox instrumentation, one from Antti's live run —
point the same way:

  1. (measured) pointwise color on a Gabor field is a rough observable: the
     probe-loss landscape oscillates at the carrier frequency; gradient
     alignment toward truth was only ~0.4.
  2. (live)     the celeba manifold derives skin tone from webcam LUMINANCE —
     color probes couple the belief to the lighting, not the subject.

Motion4Motion (SIGGRAPH 2026) steers a million-dollar video manifold with the
same medicine at the other end of the budget: a sparse set of TRACKED POINT
TRAJECTORIES, injected training-free. Displacements, not appearances.

So in V3 the afferent is a flow: K points are TRACKED on the world frame
(Lucas-Kanade, no big model, no cv2 needed), and each tick the observable is
their displacement d_i — 2K numbers. The correction asks the manifold to MOVE
THE SAME WAY:

    d_i        = LK_flow(frame_{t-1}, frame_t, p_i)          # the sparse afferent
    L_flow(z)  = sum_i || R(p_i + d_i ; z) - R(p_i ; z_prev) ||^2
                 "what the belief rendered at p under the old state must appear
                  at p+d under the new one"
    L_anchor   = zero-mean patch match (contrast, luminance-free) * small weight
    z         <- z_pred - eta * precision * clamp(grad_z [L_flow + w*L_anchor])
    p_i       <- p_i + precision * d_i                        # probes ride the flow
    (reseed)     lost/low-confidence probes jump to high-gradient features
                 — the saccade, as active sensing

The belief never has to MATCH the world's colors; it has to MOVE like the
world. Brightness constancy is only assumed frame-to-frame (LK's assumption),
so slow lighting drift — the exact live failure of color probes — passes
through the flow untouched. That is claim [B], falsified or not by --selftest.

HONEST SCOPE
  [V2 carried over] strict .pt compatibility with your trained the_splat
      SplatVAE (classes embedded verbatim; loader infers hyperparams).
  [claim A] flow-probes track a moving latent at least as well as color-probes,
      both far better than open-loop coasting.
  [claim B] under luminance drift, color-probes bias/degrade; flow-probes hold.
  [claim C] tracking improves with K and saturates — the channel-width claim.
  [claim D] dynamic precision still coasts through slop (noise frames blow up
      the LK residual -> precision collapses -> probes freeze, z coasts).
  [K, open] flow fixes MOTION, not absolute pose: pure flow integrates drift.
      The small contrast anchor (and, live, the encoder GIST) re-anchors it.
      Acquisition is still the encoder's job; probes hold, they do not find.

RUN
    python the_splatV3.py --selftest                    # headless scorecard
    python the_splatV3.py                               # GUI, synthetic world
    python the_splatV3.py --model runs/splat/model.pt --webcam
GUI: START, MODE FLOW/COLOR (A/B it live), INJECT SLOP, precision DYN/FIXED,
     GIST (encoder re-anchor, needs --model), K slider. Green dots = tracked
     probes, green lines = the measured flow they rode in on.
"""

from __future__ import annotations
import argparse, math, os, sys, time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_PARAMS = 11


# ========================================================================== #
#  the_splat v1 — EMBEDDED VERBATIM (ArtificialCortex/the_splat), so your    #
#  trained model.pt loads strict=True. Only additions: .contiguous() on the  #
#  grid buffers (a modern-torch loading fix, values unchanged) and           #
#  render_probes() (sparse evaluation; touches no state_dict key).           #
# ========================================================================== #

class GaborRenderer(nn.Module):
    def __init__(self, image_size=128, num_packets=512, chunk=64, use_checkpoint=True):
        super().__init__()
        self.H = self.W = image_size
        self.N = num_packets
        self.chunk = chunk
        self.use_checkpoint = use_checkpoint
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None].contiguous())
        self.register_buffer("GY", gy[None, None].contiguous())
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
        return px, py, sigma, theta, freq, coeff

    def _render_chunk(self, px, py, sigma, theta, freq, coeff):
        px_ = px[..., None, None]; py_ = py[..., None, None]; s_ = sigma[..., None, None]
        th = theta[..., None, None]; f_ = freq[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None, None]
            b = coeff[:, :, c, 1][..., None, None]
            chans.append((env * (a * cos - b * sin)).sum(dim=1))
        return torch.stack(chans, dim=1)

    def forward(self, raw):
        with torch.amp.autocast("cuda", enabled=False):
            px, py, sigma, theta, freq, coeff = self.activate(raw.float())
            B = raw.shape[0]
            out = torch.zeros(B, 3, self.H, self.W, device=raw.device)
            for i in range(0, self.N, self.chunk):
                sl = slice(i, i + self.chunk)
                args = (px[:, sl], py[:, sl], sigma[:, sl], theta[:, sl], freq[:, sl], coeff[:, sl])
                if self.use_checkpoint and self.training:
                    out = out + checkpoint(self._render_chunk, *args, use_reentrant=False)
                else:
                    out = out + self._render_chunk(*args)
            return torch.sigmoid(out)

    def render_probes(self, raw, pxy):
        """Evaluate the field only at coords pxy (M,2) xy in [0,1] -> (B,M,3).
        Same math as _render_chunk with the pixel grid replaced by the points.
        Verified to match the full render at grid coords to float precision."""
        px, py, sigma, theta, freq, coeff = self.activate(raw.float())
        qx = pxy[:, 0][None, None, :]
        qy = pxy[:, 1][None, None, :]
        px_ = px[..., None]; py_ = py[..., None]; s_ = sigma[..., None]
        th = theta[..., None]; f_ = freq[..., None]
        dx = qx - px_; dy = qy - py_
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None]
            b = coeff[:, :, c, 1][..., None]
            chans.append((env * (a * cos - b * sin)).sum(dim=1))
        return torch.sigmoid(torch.stack(chans, dim=-1))


class Encoder(nn.Module):
    def __init__(self, image_size=64, latent=128, ch=32):
        super().__init__()
        layers, c_in, sz, c = [], 3, image_size, ch
        while sz > 4:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2, True)]
            c_in, sz, c = c, sz // 2, min(c * 2, 512)
        self.conv = nn.Sequential(*layers)
        self.flat = c_in * sz * sz
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent=128, num_packets=256, hidden=512):
        super().__init__()
        self.N = num_packets
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, num_packets * K_PARAMS))
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].weight.data *= 0.1

    def forward(self, z):
        return self.net(z).view(-1, self.N, K_PARAMS)


class SplatVAE(nn.Module):
    def __init__(self, image_size=64, latent=128, num_packets=256, chunk=64):
        super().__init__()
        self.enc = Encoder(image_size, latent)
        self.dec = Decoder(latent, num_packets)
        self.ren = GaborRenderer(image_size, num_packets, chunk)
        self.latent = latent

    def forward(self, x):
        mu, lv = self.enc(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.ren(self.dec(z)), mu, lv

    @torch.no_grad()
    def generate(self, z):
        return self.ren(self.dec(z))


def load_v1(path, device=DEVICE):
    """Load a real the_splat checkpoint; hyperparams inferred, strict=True."""
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict) or "ren.GX" not in sd:
        raise ValueError(f"{path} is not a the_splat v1 SplatVAE state_dict")
    image_size = sd["ren.GX"].shape[-1]
    num_packets = sd["ren.anchor_logit"].shape[0]
    latent = sd["enc.fc_mu.weight"].shape[0]
    hidden = sd["dec.net.0.weight"].shape[0]
    assert sd["dec.net.4.weight"].shape[0] // K_PARAMS == num_packets
    model = SplatVAE(image_size, latent, num_packets).to(device)
    model.dec.net[0] = nn.Linear(latent, hidden)
    model.dec.net[2] = nn.Linear(hidden, hidden)
    model.dec.net[4] = nn.Linear(hidden, num_packets * K_PARAMS)
    model.dec.to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[load_v1] {path}: image_size={image_size} packets={num_packets} "
          f"latent={latent} hidden={hidden}  (strict load OK)")
    return model


# ========================================================================== #
#  LUCAS-KANADE FLOW AT K POINTS  —  torch, no cv2, two pyramid levels       #
# ========================================================================== #

class LKFlow:
    """Sparse LK: for each point, solve the 2x2 structure-tensor system over a
    small window, iterated, on a 2-level pyramid. Returns per-point flow (in
    [0,1] coords), a confidence (min eigenvalue of the structure tensor) and
    the post-fit residual (used for precision)."""

    def __init__(self, win=9, iters=3, device=DEVICE):
        self.win = win
        self.iters = iters
        r = win // 2
        oy, ox = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1),
                                indexing="ij")
        self.offs = torch.stack([ox, oy], -1).reshape(-1, 2).float().to(device)  # (W2,2) px

    @staticmethod
    def gray(img):                       # (3,H,W) -> (H,W)
        return img.mean(0)

    @staticmethod
    def grads(g):                        # central differences, per pixel step
        gx = torch.zeros_like(g); gy = torch.zeros_like(g)
        gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
        gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
        return gx, gy

    @staticmethod
    def _sample(field, coords):          # field (H,W), coords (K,W2,2) xy [0,1]
        H = field.shape[-1]
        grid = coords[None] * 2 - 1                              # (1,K,W2,2)
        v = F.grid_sample(field[None, None], grid, align_corners=True,
                          padding_mode="border")
        return v[0, 0]                                           # (K,W2)

    def _level(self, g0, g1, pts, d0):
        """One pyramid level. pts (K,2) [0,1]; d0 initial flow [0,1]. Returns
        d, conf, resid."""
        H = g0.shape[-1]
        px = 1.0 / (H - 1)
        coords0 = pts[:, None, :] + self.offs[None] * px          # (K,W2,2)
        gx, gy = self.grads(g0)
        Ix = self._sample(gx, coords0); Iy = self._sample(gy, coords0)
        I0 = self._sample(g0, coords0)
        Sxx = (Ix * Ix).sum(1); Sxy = (Ix * Iy).sum(1); Syy = (Iy * Iy).sum(1)
        det = Sxx * Syy - Sxy * Sxy
        tr = Sxx + Syy
        conf = 0.5 * (tr - torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0)))  # min eig
        d = d0.clone()
        for _ in range(self.iters):
            I1 = self._sample(g1, coords0 + d[:, None, :])
            It = I1 - I0                                          # (K,W2)
            bx = -(Ix * It).sum(1); by = -(Iy * It).sum(1)
            safe = det.abs() > 1e-9
            dx = torch.where(safe, ( Syy * bx - Sxy * by) / det, torch.zeros_like(det))
            dy = torch.where(safe, (-Sxy * bx + Sxx * by) / det, torch.zeros_like(det))
            d = d + torch.stack([dx, dy], -1) * px                # px -> [0,1]
        I1 = self._sample(g1, coords0 + d[:, None, :])
        resid = (I1 - I0).abs().mean(1)
        return d, conf, resid

    def __call__(self, frame0, frame1, pts):
        g0f, g1f = self.gray(frame0), self.gray(frame1)
        g0c = F.avg_pool2d(g0f[None, None], 2)[0, 0]
        g1c = F.avg_pool2d(g1f[None, None], 2)[0, 0]
        d, _, _ = self._level(g0c, g1c, pts, torch.zeros_like(pts))
        d, conf, resid = self._level(g0f, g1f, pts, d)
        return d, conf, resid


def good_features(frame, n_cand=256, k=1, rng=None, avoid=None, min_d=0.06):
    """Pick k high-gradient locations from n_cand random candidates (a cheap
    Shi-Tomasi). avoid: (M,2) existing points to keep distance from."""
    g = LKFlow.gray(frame)
    gx, gy = LKFlow.grads(g)
    energy = (gx * gx + gy * gy)
    rng = rng or np.random.default_rng()
    cand = torch.tensor(rng.uniform(0.08, 0.92, (n_cand, 2)), dtype=torch.float32,
                        device=frame.device)
    e = LKFlow._sample(energy, cand[:, None, :])[:, 0]
    if avoid is not None and len(avoid):
        dist = torch.cdist(cand, avoid).min(dim=1).values
        e = torch.where(dist > min_d, e, torch.zeros_like(e))
    idx = torch.topk(e, k).indices
    return cand[idx]


# ========================================================================== #
#  THE FLOW CORTEX  —  held z, corrected by where the probes went            #
# ========================================================================== #

class FlowCortex:
    def __init__(self, vae, n_probes=16, m_hist=8, sig_ref=0.02,
                 eta=20.0, beta_mom=0.5, leak=0.02, dz_clamp=0.25,
                 w_anchor=0.25, mode="flow", seed=0):
        self.vae = vae
        self.rng = np.random.default_rng(seed)
        self.z = torch.zeros(1, vae.latent, device=DEVICE)
        self.z_prev = self.z.clone()
        self.z_prior = self.z.clone()
        self.n_probes = n_probes
        self.probes = None                       # seeded on first frame
        self.prev_frame = None
        self.lk = LKFlow()
        self.m_hist = m_hist
        self.res_hist = deque(maxlen=m_hist)
        self.sig_ref = sig_ref
        self.eta = eta
        self.beta_mom = beta_mom
        self.leak = leak
        self.dz_clamp = dz_clamp
        self.w_anchor = w_anchor
        self.mode = mode                         # "flow" | "color"
        self.dynamic_precision = True
        # a 3x3 stencil (unit coords) gives each probe spatial support
        s = 1.5
        oy, ox = torch.meshgrid(torch.tensor([-s, 0., s]), torch.tensor([-s, 0., s]),
                                indexing="ij")
        self.stencil_px = torch.stack([ox, oy], -1).reshape(-1, 2).to(DEVICE)  # (9,2) px
        # telemetry
        self.precision = 1.0; self.rough = 0.0; self.resid = 0.0
        self.dz = 0.0; self.flow = None; self.flow_mag = 0.0

    # ---------- helpers ----------
    def _stencil(self, pts):
        px = 1.0 / (self.vae.ren.H - 1)
        return (pts[:, None, :] + self.stencil_px[None] * px).reshape(-1, 2).clamp(0, 1)

    def seed(self, frame):
        self.probes = good_features(frame, k=self.n_probes, rng=self.rng)
        self.prev_frame = frame.detach()
        self.res_hist.clear()

    def set_probes(self, k):
        self.n_probes = k
        self.probes = None                       # reseed on next frame

    def bootstrap(self, frame):
        """Encoder GIST: acquisition is the encoder's job; probes only hold."""
        with torch.no_grad():
            mu, _ = self.vae.enc(frame[None])
        self.z = mu.detach(); self.z_prev = self.z.clone(); self.z_prior = self.z.clone()

    @staticmethod
    def sample_frame(frame, pts):
        g = pts[None, None] * 2 - 1
        v = F.grid_sample(frame[None], g, align_corners=True)
        return v[0, :, 0, :].T                                   # (M,3)

    def _precision_from(self, r):
        self.res_hist.append(r)
        if len(self.res_hist) < 4:
            self.rough = 0.0
            return 1.0
        d2 = np.diff(np.array(self.res_hist), n=2)
        self.rough = float(np.sqrt(np.mean(d2 * d2)))
        return float(self.sig_ref**2 / (self.sig_ref**2 + self.rough**2))

    # ---------- the tick ----------
    def step(self, frame):
        frame = frame.detach()
        if self.probes is None or self.prev_frame is None:
            self.seed(frame)
            return

        # 1) the sparse afferent: where did the tracked points go
        d, conf, lk_resid = self.lk(self.prev_frame, frame, self.probes)
        self.flow = d.detach()
        self.flow_mag = float(d.norm(dim=1).mean())
        r_t = float(lk_resid.mean())
        self.resid = r_t

        # 2) precision from the LK residual's delay-history roughness
        prec = self._precision_from(r_t) if self.dynamic_precision else 1.0
        self.precision = prec

        # 3) prior flow on z: velocity from history + weak leak (null-space anchor)
        vel = self.z - self.z_prev
        z_pred = self.z + self.beta_mom * vel
        z_pred = z_pred - self.leak * (z_pred - self.z_prior)

        # 4) correction through the renderer
        z_var = z_pred.detach().clone().requires_grad_(True)
        raw = self.vae.dec(z_var)
        if self.mode == "flow":
            moved = self._stencil((self.probes + d).clamp(0, 1))
            here = self._stencil(self.probes)
            with torch.no_grad():
                ref = self.vae.ren.render_probes(self.vae.dec(self.z), here)[0]
            pred = self.vae.ren.render_probes(raw, moved)[0]
            l_flow = F.mse_loss(pred, ref)
            # luminance-free anchor: zero-mean patches (contrast/structure only)
            wp = self.sample_frame(frame, moved).reshape(-1, 9, 3)
            bp = pred.reshape(-1, 9, 3)
            l_anch = F.mse_loss(bp - bp.mean(1, keepdim=True),
                                (wp - wp.mean(1, keepdim=True)).detach())
            loss = l_flow + self.w_anchor * l_anch
        else:  # "color": the V2live baseline — raw values at the probe stencil
            here = self._stencil(self.probes)
            pred = self.vae.ren.render_probes(raw, here)[0]
            loss = F.mse_loss(pred, self.sample_frame(frame, here).detach())
        loss.backward()

        with torch.no_grad():
            step = self.eta * prec * z_var.grad
            n = step.norm()
            if n > self.dz_clamp:
                step = step * (self.dz_clamp / n)
            self.dz = float(step.norm())
            z_new = z_pred - step
        self.z_prev = self.z
        self.z = z_new.detach()
        self.z_prior = 0.995 * self.z_prior + 0.005 * self.z

        # 5) probes ride the flow — gated, so garbage flow freezes them (coast)
        with torch.no_grad():
            self.probes = (self.probes + prec * d).clamp(0.02, 0.98)
            # reseed lost / low-texture probes: the saccade
            bad = (conf < 1e-4) | (self.probes.min(1).values < 0.04) \
                  | (self.probes.max(1).values > 0.96)
            if bad.any() and prec > 0.5:
                nb = int(bad.sum())
                self.probes[bad] = good_features(frame, k=nb, rng=self.rng,
                                                 avoid=self.probes[~bad])
        self.prev_frame = frame

    @torch.no_grad()
    def belief_render(self):
        return self.vae.generate(self.z)[0]


# ========================================================================== #
#  SYNTHETIC WORLD  —  hidden z on a low-D motion; luminance drift optional  #
# ========================================================================== #

class SplatWorld:
    def __init__(self, vae, d_motion=2, radius=1.2, omega=0.05, seed=0,
                 lum_drift=False, lum_period=90):
        self.vae = vae
        g = torch.Generator(device="cpu").manual_seed(seed)
        basis = torch.randn(d_motion, vae.latent, generator=g)
        basis, _ = torch.linalg.qr(basis.T)
        self.basis = basis[:, :d_motion].to(DEVICE)
        self.d = d_motion
        self.radius = radius; self.omega = omega
        self.t = 0
        self.rng = np.random.default_rng(seed + 1)
        self.ou = np.zeros(max(0, d_motion - 2))
        self.slop_left = 0
        self.lum_drift = lum_drift; self.lum_period = lum_period
        self.z0 = torch.randn(1, vae.latent, generator=g).to(DEVICE) * 0.3

    def z_true(self):
        th = self.omega * self.t
        c = [self.radius * math.cos(th), self.radius * math.sin(th)]
        if len(self.ou):
            self.ou += -0.02 * self.ou + 0.05 * self.rng.standard_normal(len(self.ou))
            c += list(self.radius * 0.5 * self.ou)
        coeff = torch.tensor(c, dtype=torch.float32, device=DEVICE)
        return self.z0 + (self.basis @ coeff)[None]

    def inject_slop(self, n=60):
        self.slop_left = n

    @torch.no_grad()
    def frame(self):
        self.t += 1
        z = self.z_true()
        img = self.vae.generate(z)[0]
        if self.lum_drift:
            gain = 0.65 + 0.35 * math.sin(2 * math.pi * self.t / self.lum_period)
            img = (img * gain).clamp(0, 1)
        if self.slop_left > 0:
            self.slop_left -= 1
            img = torch.rand_like(img)
        return img, z


# ========================================================================== #
#  SELFTEST                                                                  #
# ========================================================================== #

def make_test_vae(seed=0, image_size=64, packets=144, latent=64,
                  wscale=8.0, bscale=2.0):
    """Amplitude from the BIAS (does not inflate the Jacobian), moderate weight
    scale — the smooth stand-in manifold. (The x60-weight version from the V2
    sandbox was an adversarial manifold of our own making; do not repeat it.)"""
    torch.manual_seed(seed)
    vae = SplatVAE(image_size, latent, packets).to(DEVICE)
    with torch.no_grad():
        vae.dec.net[-1].weight *= wscale
        vae.dec.net[-1].bias.normal_(0, bscale)
    vae.eval()
    return vae


def run_tracking(vae, T=180, n_probes=16, mode="flow", dynamic=True,
                 closed=True, lum_drift=False, slop_at=None, d_motion=2, seed=0):
    world = SplatWorld(vae, d_motion=d_motion, seed=seed, lum_drift=lum_drift)
    ctx = FlowCortex(vae, n_probes=n_probes, mode=mode, seed=seed)
    ctx.dynamic_precision = dynamic
    f0, z0 = world.frame()
    ctx.z = (z0 + 0.25 * torch.randn_like(z0)).detach()   # warm: holding, not finding
    ctx.z_prev = ctx.z.clone(); ctx.z_prior = ctx.z.clone()
    ctx.seed(f0)
    errs, errs_after = [], []
    for t in range(T):
        if slop_at and t == slop_at[0]:
            world.inject_slop(slop_at[1] - slop_at[0])
        frame, z_true = world.frame()
        if closed:
            ctx.step(frame)
        else:
            ctx.z_prev, ctx.z = ctx.z, ctx.z + ctx.beta_mom * (ctx.z - ctx.z_prev)
        if t % 2 == 0:
            with torch.no_grad():
                e = F.mse_loss(vae.generate(ctx.z), vae.generate(z_true)).item()
            errs.append(e)
            if slop_at and t >= slop_at[1] + 20:
                errs_after.append(e)
    tail = errs[len(errs) // 3:]
    return float(np.mean(tail)), (float(np.mean(errs_after)) if errs_after else None)


def selftest(seed=0):
    print(f"\n=== the_splatV3 selftest (seed {seed}, device {DEVICE}) ===")
    vae = make_test_vae(seed)
    with torch.no_grad():
        std = vae.generate(torch.randn(1, vae.latent, device=DEVICE))[0].std().item()
    print(f"stand-in field pixel std {std:.3f}")

    a_open, _ = run_tracking(vae, closed=False, seed=seed)
    a_col, _ = run_tracking(vae, mode="color", seed=seed)
    a_flow, _ = run_tracking(vae, mode="flow", seed=seed)
    print(f"[A] clean world:  flow {a_flow:.4f}   color {a_col:.4f}   open {a_open:.4f}")

    b_col, _ = run_tracking(vae, mode="color", lum_drift=True, seed=seed)
    b_flow, _ = run_tracking(vae, mode="flow", lum_drift=True, seed=seed)
    print(f"[B] luminance drift:  flow {b_flow:.4f}   color {b_col:.4f}   "
          f"(clean flow was {a_flow:.4f})")

    print("[C] flow error vs K probes:")
    row = []
    for k in (4, 8, 16, 32):
        e, _ = run_tracking(vae, n_probes=k, mode="flow", T=140, seed=seed)
        row.append(f"K={k}:{e:.4f}")
    print("    " + "  ".join(row) + f"   (open {a_open:.4f})")

    d_dyn, d_dyn_a = run_tracking(vae, mode="flow", slop_at=(80, 130), dynamic=True, seed=seed)
    d_fix, d_fix_a = run_tracking(vae, mode="flow", slop_at=(80, 130), dynamic=False, seed=seed)
    print(f"[D] slop: dyn {d_dyn:.4f} (after {d_dyn_a:.4f})   "
          f"fixed {d_fix:.4f} (after {d_fix_a:.4f})")

    print("\nledger: [A] flow<=color<<open, [B] color degrades under lum-drift &")
    print("        flow does not, [C] error falls with K then saturates,")
    print("        [D] dyn<=fixed through slop. Any reversal kills that claim.")


# ========================================================================== #
#  GUI                                                                       #
# ========================================================================== #

def run_gui(model_path=None, webcam=False, cam_index=0):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw

    vae = load_v1(model_path) if model_path else make_test_vae(0)
    has_enc = model_path is not None
    world = None if webcam else SplatWorld(vae, d_motion=3, seed=0)
    cap = None
    if webcam:
        import cv2
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("webcam not available — synthetic world instead")
            world, cap = SplatWorld(vae, d_motion=3, seed=0), None

    ctx = FlowCortex(vae, n_probes=16)
    H = vae.ren.H
    running = {"on": False}; slop = {"left": 0}; booted = {"done": False}

    root = tk.Tk()
    root.title("the_splatV3 — flow-probes: the belief moves like the world")
    root.configure(bg="#101014")
    VIEW = 280

    top = tk.Frame(root, bg="#101014"); top.pack(padx=8, pady=8)
    tk.Label(top, text="AFFERENT (tracked probes + flow)", fg="#8fd",
             bg="#101014").grid(row=0, column=0)
    tk.Label(top, text="BELIEF  render(dec(z))", fg="#fd8",
             bg="#101014").grid(row=0, column=1)
    lab_world = tk.Label(top, bg="#101014"); lab_world.grid(row=1, column=0, padx=4)
    lab_belief = tk.Label(top, bg="#101014"); lab_belief.grid(row=1, column=1, padx=4)

    tele = tk.Label(root, fg="#ddd", bg="#101014", font=("Courier", 10), justify="left")
    tele.pack()
    ctrl = tk.Frame(root, bg="#101014"); ctrl.pack(pady=6)

    def toggle():
        running["on"] = not running["on"]
        b_start.config(text="STOP" if running["on"] else "START")

    def do_slop():
        if world: world.inject_slop(60)
        else: slop["left"] = 60

    def toggle_prec():
        ctx.dynamic_precision = not ctx.dynamic_precision
        b_prec.config(text=f"precision {'DYN' if ctx.dynamic_precision else 'FIXED'}")

    def toggle_mode():
        ctx.mode = "color" if ctx.mode == "flow" else "flow"
        b_mode.config(text=f"MODE {ctx.mode.upper()}")

    def do_gist():
        if ctx.prev_frame is not None and has_enc:
            ctx.bootstrap(ctx.prev_frame)

    def set_k(v):
        ctx.set_probes(int(float(v)))

    b_start = tk.Button(ctrl, text="START", command=toggle, width=7)
    b_mode = tk.Button(ctrl, text="MODE FLOW", command=toggle_mode)
    b_slop = tk.Button(ctrl, text="INJECT SLOP", command=do_slop)
    b_prec = tk.Button(ctrl, text="precision DYN", command=toggle_prec)
    b_gist = tk.Button(ctrl, text="GIST (enc)", command=do_gist,
                       state="normal" if has_enc else "disabled")
    for i, b in enumerate((b_start, b_mode, b_slop, b_prec, b_gist)):
        b.grid(row=0, column=i, padx=3)
    tk.Label(ctrl, text="K", fg="#ddd", bg="#101014").grid(row=0, column=5, padx=(12, 0))
    s_k = tk.Scale(ctrl, from_=4, to=64, orient="horizontal", bg="#101014",
                   fg="#ddd", command=set_k, length=140)
    s_k.set(16); s_k.grid(row=0, column=6)

    def to_photo(img_t, probes=None, flow=None):
        arr = (img_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((VIEW, VIEW), Image.NEAREST)
        if probes is not None:
            dr = ImageDraw.Draw(im)
            pts = probes.cpu().numpy()
            fl = flow.cpu().numpy() if flow is not None else None
            for i, (x, y) in enumerate(pts):
                dr.ellipse([x*VIEW-3, y*VIEW-3, x*VIEW+3, y*VIEW+3],
                           outline="#00ff88", width=2)
                if fl is not None:
                    dr.line([x*VIEW, y*VIEW, (x+fl[i,0]*8)*VIEW, (y+fl[i,1]*8)*VIEW],
                            fill="#00ff88", width=2)
        return ImageTk.PhotoImage(im)

    def grab_frame():
        if cap is not None:
            import cv2
            ok, fr = cap.read()
            if not ok: return None
            h, w, _ = fr.shape
            s = min(h, w)
            fr = fr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            fr = cv2.resize(fr, (H, H))
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(fr).float().permute(2, 0, 1).to(DEVICE) / 255.0
            if slop["left"] > 0:
                slop["left"] -= 1
                t = torch.rand_like(t)
            return t
        img, _ = world.frame()
        return img

    n = {"t": 0}

    def tick():
        if running["on"]:
            frame = grab_frame()
            if frame is not None:
                if not booted["done"] and has_enc:
                    ctx.prev_frame = frame
                    ctx.bootstrap(frame)
                    booted["done"] = True
                ctx.step(frame)
                n["t"] += 1
                ph_w = to_photo(frame, ctx.probes, ctx.flow)
                lab_world.configure(image=ph_w); lab_world.image = ph_w
                if DEVICE == "cuda" or n["t"] % 2 == 0:
                    ph_b = to_photo(ctx.belief_render())
                    lab_belief.configure(image=ph_b); lab_belief.image = ph_b
                tele.config(text=(
                    f"mode {ctx.mode:5s}  precision {ctx.precision:5.2f}  "
                    f"rough {ctx.rough:6.4f}  lk-resid {ctx.resid:6.4f}  "
                    f"|flow| {ctx.flow_mag:6.4f}  |dz| {ctx.dz:6.3f}  "
                    f"K {ctx.n_probes}  t {n['t']}"))
        root.after(40, tick)

    tick()
    root.mainloop()
    if cap is not None:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.seed)
    else:
        run_gui(args.model, args.webcam, args.cam)


if __name__ == "__main__":
    main()
