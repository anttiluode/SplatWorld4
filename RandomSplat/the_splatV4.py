"""
the_splatV4.py  —  TWO-BAND flow-probes: coarse flow steers the low-frequency
                   packets (body/pose) hard; fine flow steers the high-frequency
                   packets (face) softly. The prior/world division, in code.

PerceptionLab / Antti Luode (Helsinki), with Claude (Fable 5). July 2026.

    Do not hype. Do not lie. Just show.

WHY V4 EXISTS  (Antti's frequency insight, made mechanical)
-----------------------------------------------------------
V3 proved flow-probes beat color-probes and hold under luminance drift. But it
threw one band of identical probes at the whole frame. Live, the useful thing
was the SHOULDERS: a big, smooth, slow structure that the loop could actually be
steered by ("an amalgamation of shoulders from thousands of celebs became a
channel I connect to by moving my shoulders"). The face, by contrast, fights
back — the prior is so strong that a table still renders a face.

The reason is frequency. A face is HIGH spatial frequency (fine detail); the
body and surround are LOW spatial frequency (a smooth envelope). In the splat
field this is explicit: each Gabor packet carries a `freq`. So:

  - LOW-freq packets render the envelope: body mass, pose, the slow surround.
    Their motion is low spatial AND low temporal frequency — exactly what a
    LARGE-window LK probe on a downsampled frame measures robustly.
  - HIGH-freq packets render the face detail: the part the manifold is most
    confident about (a strong, over-trusted prior) and the part small LK probes
    catch then lose.

V4 splits the afferent into two bands and routes each to its own packets:

  d_coarse = LK(pyramid, big window)   -> corrects LOW-freq packets,  HIGH precision
  d_fine   = LK(full-res, small window)-> corrects HIGH-freq packets, LOW  precision

The coarse band is world-driven: it's reliable motion the manifold should honor,
so it corrects hard — the belief tracks your lean, turn, shoulder-rise, your
POSITION. The fine band is prior-dominated: it corrects weakly, so the face
stays a soft frontal amalgam, honest about being a projection. This is not a bug
fix; it is the correct division of labor between world-driven and prior-driven
perception, written into the loss weighting.

THE FALSIFIABLE CLAIM (--selftest [E]): coarse-band pose tracking HOLDS under
conditions — luminance drift, a high-frequency distractor in the surround —
where full-band V3-style tracking DRIFTS, because the coarse band never listens
to the high-frequency channel that those conditions corrupt.

WHAT'S CARRIED OVER VERBATIM (verified in V3, unchanged here):
  - the_splat v1 SplatVAE (your model.pt loads strict; renders bit-identical)
  - render_probes (sparse eval matches full render to float precision)
  - LKFlow single-band core (recovers a known pixel shift exactly)
  - precision-from-residual-roughness, the prior-flow + leak, the saccade
V4 ADDS: a band split on the packets (by freq percentile), a two-band LK, and a
two-band correction with per-band precision. The numpy prototype confirmed the
bands separate cleanly: a frame carrying a +3px low-freq slide and a -2px
high-freq nod is recovered as coarse [+2.98, 0] and fine [+0.29, -1.74].

HONEST SCOPE
  [carried] flow>color, flow holds under lum-drift, K-scaling, dyn coasts slop.
  [E, the new bet] two-band pose tracking is more robust than one-band under
      surround/lighting corruption — falsified if coarse drifts with full-band.
  [K, open] the band split is by the manifold's OWN freq parameter, so it is
      only as meaningful as the trained field's freq organization. On a real
      celeba .pt this is an empirical question — the GUI shows both bands so you
      can see whether low-freq packets really are the body. Acquisition is still
      the encoder's job (GIST); probes hold, they do not find.

RUN
    python the_splatV4.py --selftest
    python the_splatV4.py                                  # synthetic world
    python the_splatV4.py --model runs/splat/model.pt --webcam
GUI: START, BANDS BOTH/COARSE/FINE (isolate a band live), INJECT SLOP,
     precision DYN/FIXED, GIST, K slider. Cyan probes+lines = coarse band,
     magenta = fine band; you can watch which one your shoulders drive.
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


def packet_bands(vae, low_frac=0.5):
    """Split packets into (coarse_idx, fine_idx) by their trained spatial
    frequency. Low-freq packets = the envelope (body/pose); high-freq = detail
    (face). On the random stand-in field this is by construction; on a real
    celeba .pt it is an empirical read of the field's own organization."""
    with torch.no_grad():
        # freq for each packet at z=0 baseline (freq depends only on raw[...,4],
        # which the decoder sets; evaluate at the current prior mean is fine, but
        # z=0 gives a stable, state-independent split we can cache).
        raw = vae.dec(torch.zeros(1, vae.latent, device=DEVICE))
        freq = (1.0 + 15.0 * torch.sigmoid(raw[0, :, 4]))          # (N,)
    thresh = torch.quantile(freq, low_frac)
    coarse = (freq <= thresh).nonzero(as_tuple=True)[0]
    fine = (freq > thresh).nonzero(as_tuple=True)[0]
    return coarse, fine, freq


def render_probes_subset(ren, raw, pxy, idx):
    """render_probes but summing ONLY the packets in idx — so a band's flow
    corrects a band's packets. raw (1,N,K).

    NOTE: ren.activate() adds the full (N,2) anchor_logit buffer, so it can only
    be called on the full packet set. Here we inline the identical activation
    math with the anchor SLICED to idx — same numbers, subset-safe. (The verbatim
    v1 class is left untouched.)"""
    sub = raw[:, idx, :].float()
    a_px = ren.anchor_logit[idx, 0][None]
    a_py = ren.anchor_logit[idx, 1][None]
    px = torch.sigmoid(a_px + sub[..., 0])
    py = torch.sigmoid(a_py + sub[..., 1])
    sigma = 0.012 + 0.14 * torch.sigmoid(sub[..., 2])
    theta = sub[..., 3]
    freq = 1.0 + 15.0 * torch.sigmoid(sub[..., 4])
    coeff = torch.tanh(sub[..., 5:11]).reshape(*sub.shape[:2], 3, 2)
    qx = pxy[:, 0][None, None, :]; qy = pxy[:, 1][None, None, :]
    px_ = px[..., None]; py_ = py[..., None]; s_ = sigma[..., None]
    th = theta[..., None]; f_ = freq[..., None]
    dx = qx - px_; dy = qy - py_
    xr = dx * torch.cos(th) + dy * torch.sin(th)
    env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
    cos = torch.cos(2 * math.pi * f_ * xr); sin = torch.sin(2 * math.pi * f_ * xr)
    chans = []
    for c in range(3):
        a = coeff[:, :, c, 0][..., None]; b = coeff[:, :, c, 1][..., None]
        chans.append((env * (a * cos - b * sin)).sum(dim=1))
    # NOTE: no sigmoid here — a band is a partial field; we compare partial
    # renders old-vs-new consistently, so the monotone squash is unnecessary and
    # would compress the very gradients we steer by.
    return torch.stack(chans, dim=-1)


@torch.no_grad()
def render_full_subset(ren, raw, idx):
    """Full HxW render from ONLY the packets in idx. Same math as the verbatim
    forward(), anchor sliced to the subset. Used by the band diagnostic to SHOW
    what each half of the field draws. Returns (3,H,W) in [0,1]."""
    sub = raw[:, idx, :].float()
    a_px = ren.anchor_logit[idx, 0][None]
    a_py = ren.anchor_logit[idx, 1][None]
    px = torch.sigmoid(a_px + sub[..., 0])
    py = torch.sigmoid(a_py + sub[..., 1])
    sigma = 0.012 + 0.14 * torch.sigmoid(sub[..., 2])
    theta = sub[..., 3]
    freq = 1.0 + 15.0 * torch.sigmoid(sub[..., 4])
    coeff = torch.tanh(sub[..., 5:11]).reshape(*sub.shape[:2], 3, 2)
    px_ = px[..., None, None]; py_ = py[..., None, None]; s_ = sigma[..., None, None]
    th = theta[..., None, None]; f_ = freq[..., None, None]
    dx = ren.GX - px_; dy = ren.GY - py_
    xr = dx * torch.cos(th) + dy * torch.sin(th)
    env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
    cos = torch.cos(2 * math.pi * f_ * xr); sin = torch.sin(2 * math.pi * f_ * xr)
    chans = []
    for c in range(3):
        a = coeff[:, :, c, 0][..., None, None]; b = coeff[:, :, c, 1][..., None, None]
        chans.append((env * (a * cos - b * sin)).sum(dim=1))
    return torch.sigmoid(torch.stack(chans, dim=1))[0]


def band_diagnostic(model_path=None, seed=0, n_samples=5, out="band_diagnostic.png"):
    """BEFORE the loop: show what each half of the field draws. For several
    random z (or the encoder's z on the stand-in), render (full | coarse-only |
    fine-only) side by side. If the split is meaningful, the coarse column is the
    body/surround envelope and the fine column is the face detail. This is the
    empirical test of whether a 2-epoch celeba field organized freq -> content.
    Saves a PNG; no torch install needed beyond what runs the model."""
    from PIL import Image
    vae = load_v1(model_path) if model_path else make_test_vae(seed)
    coarse_idx, fine_idx, freq = packet_bands(vae)
    print(f"[band_diagnostic] {len(coarse_idx)} coarse packets (freq<={freq[coarse_idx].max():.2f}), "
          f"{len(fine_idx)} fine (freq>={freq[fine_idx].min():.2f})")
    H = vae.ren.H
    rng = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for s in range(n_samples):
        z = (torch.randn(1, vae.latent, generator=rng) * 1.1).to(DEVICE)
        with torch.no_grad():
            raw = vae.dec(z)
            full = vae.generate(z)                      # (1,3,H,W)
            coa = render_full_subset(vae.ren, raw, coarse_idx)
            fin = render_full_subset(vae.ren, raw, fine_idx)
        def np_img(t):
            return (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
        trip = np.concatenate([np_img(full[0]), np_img(coa), np_img(fin)], axis=1)
        rows.append(trip)
    grid = np.concatenate(rows, axis=0)
    # column headers strip
    hdr = Image.new("RGB", (grid.shape[1], 22), (16, 16, 20))
    from PIL import ImageDraw
    d = ImageDraw.Draw(hdr)
    for i, name in enumerate(("FULL", "COARSE (low-freq)", "FINE (high-freq)")):
        d.text((i * H + 6, 5), name, fill=(140, 230, 200))
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + 22), (16, 16, 20))
    canvas.paste(hdr, (0, 0))
    canvas.paste(Image.fromarray(grid), (0, 22))
    canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.NEAREST)
    canvas.save(out)
    print(f"[band_diagnostic] wrote {out}  ({n_samples} samples x [full|coarse|fine])")
    return out


class TwoBandLK:
    """Coarse band: large window on the half-res pyramid (envelope motion).
    Fine band: small window at full res (texture motion). Same verified LK math
    as V3's LKFlow, run at two configurations."""
    def __init__(self, device=DEVICE):
        self.coarse = LKFlow(win=11, iters=4, device=device)
        self.fine = LKFlow(win=5, iters=3, device=device)

    def coarse_flow(self, f0, f1, pts):
        g0 = F.avg_pool2d(LKFlow.gray(f0)[None, None], 2)[0, 0]
        g1 = F.avg_pool2d(LKFlow.gray(f1)[None, None], 2)[0, 0]
        return self.coarse._level(g0, g1, pts, torch.zeros_like(pts))

    def fine_flow(self, f0, f1, pts):
        g0 = LKFlow.gray(f0); g1 = LKFlow.gray(f1)
        return self.fine._level(g0, g1, pts, torch.zeros_like(pts))


# ========================================================================== #
#  THE TWO-BAND FLOW CORTEX                                                  #
# ========================================================================== #

class BandedFlowCortex:
    def __init__(self, vae, n_coarse=10, n_fine=14, m_hist=8, sig_ref=0.02,
                 eta=20.0, beta_mom=0.5, leak=0.02, dz_clamp=0.25,
                 w_coarse=1.0, w_fine=0.25, prec_fine_cap=0.5,
                 bands="both", seed=0):
        self.vae = vae
        self.rng = np.random.default_rng(seed)
        self.z = torch.zeros(1, vae.latent, device=DEVICE)
        self.z_prev = self.z.clone(); self.z_prior = self.z.clone()
        self.n_coarse = n_coarse; self.n_fine = n_fine
        self.pc = None; self.pf = None                # probe sets, seeded on frame 1
        self.prev_frame = None
        self.lk = TwoBandLK()
        self.coarse_idx, self.fine_idx, self.freq = packet_bands(vae)
        self.m_hist = m_hist
        self.rc = deque(maxlen=m_hist); self.rf = deque(maxlen=m_hist)
        self.sig_ref = sig_ref; self.eta = eta; self.beta_mom = beta_mom
        self.leak = leak; self.dz_clamp = dz_clamp
        self.w_coarse = w_coarse; self.w_fine = w_fine
        self.prec_fine_cap = prec_fine_cap            # fine band never fully trusted
        self.bands = bands                            # "both" | "coarse" | "fine"
        self.dynamic_precision = True
        # 3x3 stencil in pixel units for spatial support
        s = 1.5
        oy, ox = torch.meshgrid(torch.tensor([-s, 0., s]), torch.tensor([-s, 0., s]),
                                indexing="ij")
        self.stencil_px = torch.stack([ox, oy], -1).reshape(-1, 2).to(DEVICE)
        # telemetry
        self.prec_c = 1.0; self.prec_f = 1.0; self.rough_c = 0.0; self.rough_f = 0.0
        self.dz = 0.0; self.flow_c = None; self.flow_f = None
        self.mag_c = 0.0; self.mag_f = 0.0

    def _stencil(self, pts):
        px = 1.0 / (self.vae.ren.H - 1)
        return (pts[:, None, :] + self.stencil_px[None] * px).reshape(-1, 2).clamp(0, 1)

    def seed(self, frame):
        self.pc = good_features(frame, k=self.n_coarse, rng=self.rng, n_cand=200)
        self.pf = good_features(frame, k=self.n_fine, rng=self.rng, n_cand=300,
                                avoid=self.pc, min_d=0.05)
        self.prev_frame = frame.detach(); self.rc.clear(); self.rf.clear()

    def set_probes(self, kc, kf):
        self.n_coarse, self.n_fine = kc, kf
        self.pc = self.pf = None
        self.flow_c = self.flow_f = None   # drop stale flow until reseed+step

    def bootstrap(self, frame):
        with torch.no_grad():
            mu, _ = self.vae.enc(frame[None])
        self.z = mu.detach(); self.z_prev = self.z.clone(); self.z_prior = self.z.clone()

    def _prec(self, hist, r):
        hist.append(r)
        if len(hist) < 4:
            return 1.0, 0.0
        d2 = np.diff(np.array(hist), n=2)
        rough = float(np.sqrt(np.mean(d2 * d2)))
        return float(self.sig_ref**2 / (self.sig_ref**2 + rough**2)), rough

    def step(self, frame):
        frame = frame.detach()
        if self.pc is None or self.prev_frame is None:
            self.seed(frame); return

        # --- two-band afferent ---
        dc, confc, resc = self.lk.coarse_flow(self.prev_frame, frame, self.pc)
        df, conff, resf = self.lk.fine_flow(self.prev_frame, frame, self.pf)
        self.flow_c, self.flow_f = dc.detach(), df.detach()
        self.mag_c = float(dc.norm(dim=1).mean()); self.mag_f = float(df.norm(dim=1).mean())

        # --- per-band precision from residual roughness ---
        if self.dynamic_precision:
            pc_, self.rough_c = self._prec(self.rc, float(resc.mean()))
            pf_, self.rough_f = self._prec(self.rf, float(resf.mean()))
        else:
            pc_ = pf_ = 1.0
        pf_ = min(pf_, self.prec_fine_cap)            # fine band structurally capped
        self.prec_c, self.prec_f = pc_, pf_

        # --- prior flow on z ---
        vel = self.z - self.z_prev
        z_pred = self.z + self.beta_mom * vel
        z_pred = z_pred - self.leak * (z_pred - self.z_prior)

        # --- banded correction: each band's flow moves its band's packets ---
        z_var = z_pred.detach().clone().requires_grad_(True)
        raw = self.vae.dec(z_var)
        with torch.no_grad():
            raw_old = self.vae.dec(self.z)
        loss = 0.0 * z_var.sum()
        if self.bands in ("both", "coarse"):
            moved = self._stencil((self.pc + dc).clamp(0, 1))
            here = self._stencil(self.pc)
            with torch.no_grad():
                ref = render_probes_subset(self.vae.ren, raw_old, here, self.coarse_idx)[0]
            pred = render_probes_subset(self.vae.ren, raw, moved, self.coarse_idx)[0]
            loss = loss + self.w_coarse * self.prec_c * F.mse_loss(pred, ref)
        if self.bands in ("both", "fine"):
            moved = self._stencil((self.pf + df).clamp(0, 1))
            here = self._stencil(self.pf)
            with torch.no_grad():
                ref = render_probes_subset(self.vae.ren, raw_old, here, self.fine_idx)[0]
            pred = render_probes_subset(self.vae.ren, raw, moved, self.fine_idx)[0]
            loss = loss + self.w_fine * self.prec_f * F.mse_loss(pred, ref)
        loss.backward()

        with torch.no_grad():
            step = self.eta * z_var.grad
            n = step.norm()
            if n > self.dz_clamp:
                step = step * (self.dz_clamp / n)
            self.dz = float(step.norm())
            z_new = z_pred - step
        self.z_prev = self.z; self.z = z_new.detach()
        self.z_prior = 0.995 * self.z_prior + 0.005 * self.z

        # --- probes ride their band's flow, gated by that band's precision ---
        with torch.no_grad():
            self.pc = (self.pc + pc_ * dc).clamp(0.02, 0.98)
            self.pf = (self.pf + pf_ * df).clamp(0.02, 0.98)
            for pts, conf, kfn, pr in ((self.pc, confc, self.n_coarse, pc_),
                                       (self.pf, conff, self.n_fine, pf_)):
                bad = (conf < 1e-4) | (pts.min(1).values < 0.04) | (pts.max(1).values > 0.96)
                if bad.any() and pr > 0.5:
                    pts[bad] = good_features(frame, k=int(bad.sum()), rng=self.rng,
                                             avoid=pts[~bad])
        self.prev_frame = frame

    @torch.no_grad()
    def belief_render(self):
        return self.vae.generate(self.z)[0]


# ========================================================================== #
#  SYNTHETIC WORLD  —  two independent motions + a corrupting HF distractor  #
# ========================================================================== #

class BandedWorld:
    """Hidden z on a low-D motion. The world's z-motion is split so that the
    COARSE latent axes move slowly (pose) and the FINE axes move faster
    (detail). Optional lum_drift and a high-frequency surround distractor that
    corrupts the FINE band without touching the coarse — the [E] stressor."""
    def __init__(self, vae, seed=0, omega_c=0.04, omega_f=0.13, radius=1.1,
                 lum_drift=False, hf_distractor=False):
        self.vae = vae
        g = torch.Generator(device="cpu").manual_seed(seed)
        B = torch.randn(4, vae.latent, generator=g)
        B, _ = torch.linalg.qr(B.T)
        self.basis = B[:, :4].to(DEVICE)               # cols 0,1 coarse; 2,3 fine
        self.omega_c, self.omega_f, self.radius = omega_c, omega_f, radius
        self.t = 0
        self.z0 = torch.randn(1, vae.latent, generator=g).to(DEVICE) * 0.3
        self.lum_drift = lum_drift; self.hf_distractor = hf_distractor
        self.slop_left = 0
        yy, xx = torch.meshgrid(torch.linspace(0, 1, vae.ren.H),
                                torch.linspace(0, 1, vae.ren.H), indexing="ij")
        self.hf = (torch.sin(xx * 60) * torch.sin(yy * 60)).to(DEVICE)   # fine texture

    def z_true(self):
        tc, tf = self.omega_c * self.t, self.omega_f * self.t
        c = torch.tensor([self.radius * math.cos(tc), self.radius * math.sin(tc),
                          0.5 * self.radius * math.cos(tf), 0.5 * self.radius * math.sin(tf)],
                         dtype=torch.float32, device=DEVICE)
        return self.z0 + (self.basis @ c)[None]

    def pose_z(self):
        """Ground-truth COARSE component only (pose), for scoring [E]."""
        tc = self.omega_c * self.t
        c = torch.tensor([self.radius * math.cos(tc), self.radius * math.sin(tc), 0, 0],
                         dtype=torch.float32, device=DEVICE)
        return self.z0 + (self.basis @ c)[None]

    def inject_slop(self, n=60): self.slop_left = n

    @torch.no_grad()
    def frame(self):
        self.t += 1
        img = self.vae.generate(self.z_true())[0]
        if self.hf_distractor:
            # a moving high-frequency overlay in the surround: wrecks fine LK,
            # leaves the coarse (downsampled) band almost untouched.
            phase = 0.5 * math.sin(self.t * 0.3)
            img = (img + 0.25 * torch.roll(self.hf, int(6 * phase), 0)[None]).clamp(0, 1)
        if self.lum_drift:
            img = (img * (0.65 + 0.35 * math.sin(2 * math.pi * self.t / 90))).clamp(0, 1)
        if self.slop_left > 0:
            self.slop_left -= 1; img = torch.rand_like(img)
        return img, self.z_true()


# ========================================================================== #
#  SELFTEST                                                                  #
# ========================================================================== #

def make_test_vae(seed=0, image_size=64, packets=144, latent=64,
                  wscale=8.0, bscale=2.0):
    torch.manual_seed(seed)
    vae = SplatVAE(image_size, latent, packets).to(DEVICE)
    with torch.no_grad():
        vae.dec.net[-1].weight *= wscale
        vae.dec.net[-1].bias.normal_(0, bscale)
    vae.eval(); return vae


def _run(vae, T=170, bands="both", dynamic=True, lum_drift=False,
         hf_distractor=False, slop_at=None, kc=10, kf=14, seed=0,
         score="pose"):
    world = BandedWorld(vae, seed=seed, lum_drift=lum_drift, hf_distractor=hf_distractor)
    ctx = BandedFlowCortex(vae, n_coarse=kc, n_fine=kf, bands=bands, seed=seed)
    ctx.dynamic_precision = dynamic
    f0, z0 = world.frame()
    ctx.z = (z0 + 0.2 * torch.randn_like(z0)).detach()
    ctx.z_prev = ctx.z.clone(); ctx.z_prior = ctx.z.clone()
    ctx.seed(f0)
    errs, after = [], []
    for t in range(T):
        if slop_at and t == slop_at[0]:
            world.inject_slop(slop_at[1] - slop_at[0])
        frame, z_true = world.frame()
        ctx.step(frame)
        if t % 2 == 0:
            with torch.no_grad():
                tgt = world.pose_z() if score == "pose" else z_true
                e = F.mse_loss(vae.generate(ctx.z), vae.generate(tgt)).item()
            errs.append(e)
            if slop_at and t >= slop_at[1] + 20:
                after.append(e)
    tail = errs[len(errs) // 3:]
    return float(np.mean(tail)), (float(np.mean(after)) if after else None)


def _open(vae, seed=0, score="pose", lum_drift=False, hf_distractor=False, T=170):
    world = BandedWorld(vae, seed=seed, lum_drift=lum_drift, hf_distractor=hf_distractor)
    f0, z0 = world.frame()
    z = (z0 + 0.2 * torch.randn_like(z0)); zp = z.clone()
    errs = []
    for t in range(T):
        frame, z_true = world.frame()
        z, zp = z + 0.5 * (z - zp), z
        if t % 2 == 0:
            with torch.no_grad():
                tgt = world.pose_z() if score == "pose" else z_true
                errs.append(F.mse_loss(vae.generate(z), vae.generate(tgt)).item())
    return float(np.mean(errs[len(errs)//3:]))


def selftest(seed=0):
    print(f"\n=== the_splatV4 selftest (seed {seed}, device {DEVICE}) ===")
    vae = make_test_vae(seed)
    ci, fi, freq = packet_bands(vae)
    print(f"packets: {len(ci)} coarse (freq<={freq[ci].max():.1f})  "
          f"{len(fi)} fine (freq>={freq[fi].min():.1f})")

    both, _ = _run(vae, bands="both", score="pose", seed=seed)
    coa, _ = _run(vae, bands="coarse", score="pose", seed=seed)
    opn = _open(vae, score="pose", seed=seed)
    print(f"[A] pose tracking (clean):  both {both:.4f}   coarse-only {coa:.4f}   open {opn:.4f}")

    # [E] the bet: under an HF surround distractor, does coarse-only hold pose
    # while the full (both-band) loop drifts, because 'both' listens to the
    # corrupted fine band?
    both_d, _ = _run(vae, bands="both", score="pose", hf_distractor=True, seed=seed)
    coa_d, _ = _run(vae, bands="coarse", score="pose", hf_distractor=True, seed=seed)
    opn_d = _open(vae, score="pose", hf_distractor=True, seed=seed)
    print(f"[E] pose under HF distractor: coarse-only {coa_d:.4f}   both {both_d:.4f}   "
          f"open {opn_d:.4f}")
    print(f"    (clean coarse was {coa:.4f} — did the distractor move it?)")

    # [B] luminance drift, pose score
    both_l, _ = _run(vae, bands="both", score="pose", lum_drift=True, seed=seed)
    coa_l, _ = _run(vae, bands="coarse", score="pose", lum_drift=True, seed=seed)
    print(f"[B] pose under lum-drift:  coarse-only {coa_l:.4f}   both {both_l:.4f}")

    # [C] coarse-band pose vs K
    print("[C] coarse-only pose error vs K_coarse:")
    row = []
    for k in (4, 8, 16, 24):
        e, _ = _run(vae, bands="coarse", kc=k, T=130, score="pose", seed=seed)
        row.append(f"K={k}:{e:.4f}")
    print("    " + "  ".join(row) + f"   (open {opn:.4f})")

    # [D] slop coasting (both bands, full z score)
    dyn, dyn_a = _run(vae, bands="both", slop_at=(75, 120), dynamic=True, score="full", seed=seed)
    fix, fix_a = _run(vae, bands="both", slop_at=(75, 120), dynamic=False, score="full", seed=seed)
    print(f"[D] slop: dyn {dyn:.4f} (after {dyn_a:.4f})   fixed {fix:.4f} (after {fix_a:.4f})")

    print("\nledger: [A] both/coarse both beat open on POSE.")
    print("        [E] THE BET: coarse-only pose error ~ its clean value under the")
    print("            HF distractor, and <= 'both' — the coarse band ignores the")
    print("            channel the distractor corrupts. If both<=coarse here, or")
    print("            coarse jumps vs its clean value, the two-band claim is dead.")
    print("        [B] coarse holds pose under lum-drift. [C] falls with K.")
    print("        [D] dyn<=fixed through slop.")


# ========================================================================== #
#  GUI                                                                       #
# ========================================================================== #

def run_gui(model_path=None, webcam=False, cam_index=0):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw

    vae = load_v1(model_path) if model_path else make_test_vae(0)
    has_enc = model_path is not None
    world = None if webcam else BandedWorld(vae, seed=0)
    cap = None
    if webcam:
        import cv2
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("webcam not available — synthetic world instead")
            world, cap = BandedWorld(vae, seed=0), None

    ctx = BandedFlowCortex(vae, n_coarse=10, n_fine=14)
    H = vae.ren.H
    running = {"on": False}; slop = {"left": 0}; booted = {"done": False}

    root = tk.Tk()
    root.title("the_splatV4 — two bands: cyan=coarse(body) magenta=fine(face)")
    root.configure(bg="#101014")
    VIEW = 280

    top = tk.Frame(root, bg="#101014"); top.pack(padx=8, pady=8)
    tk.Label(top, text="AFFERENT (cyan=coarse  magenta=fine)", fg="#8fd",
             bg="#101014").grid(row=0, column=0)
    tk.Label(top, text="BELIEF  render(dec(z))  [VIEW toggles full/coarse/fine]",
             fg="#fd8", bg="#101014").grid(row=0, column=1)
    lab_world = tk.Label(top, bg="#101014"); lab_world.grid(row=1, column=0, padx=4)
    lab_belief = tk.Label(top, bg="#101014"); lab_belief.grid(row=1, column=1, padx=4)

    tele = tk.Label(root, fg="#ddd", bg="#101014", font=("Courier", 10), justify="left")
    tele.pack()
    ctrl = tk.Frame(root, bg="#101014"); ctrl.pack(pady=6)

    def toggle():
        running["on"] = not running["on"]
        b_start.config(text="STOP" if running["on"] else "START")

    def cycle_bands():
        ctx.bands = {"both": "coarse", "coarse": "fine", "fine": "both"}[ctx.bands]
        b_band.config(text=f"BANDS {ctx.bands.upper()}")

    def do_slop():
        if world: world.inject_slop(60)
        else: slop["left"] = 60

    def toggle_prec():
        ctx.dynamic_precision = not ctx.dynamic_precision
        b_prec.config(text=f"precision {'DYN' if ctx.dynamic_precision else 'FIXED'}")

    def do_gist():
        if ctx.prev_frame is not None and has_enc:
            ctx.bootstrap(ctx.prev_frame)

    view = {"mode": "full"}   # belief pane: full | coarse | fine

    def cycle_view():
        view["mode"] = {"full": "coarse", "coarse": "fine", "fine": "full"}[view["mode"]]
        b_view.config(text=f"VIEW {view['mode'].upper()}")

    def set_k(v):
        k = int(float(v)); ctx.set_probes(max(4, k // 2), k)

    b_start = tk.Button(ctrl, text="START", command=toggle, width=7)
    b_band = tk.Button(ctrl, text="BANDS BOTH", command=cycle_bands)
    b_view = tk.Button(ctrl, text="VIEW FULL", command=cycle_view)
    b_slop = tk.Button(ctrl, text="INJECT SLOP", command=do_slop)
    b_prec = tk.Button(ctrl, text="precision DYN", command=toggle_prec)
    b_gist = tk.Button(ctrl, text="GIST (enc)", command=do_gist,
                       state="normal" if has_enc else "disabled")
    for i, b in enumerate((b_start, b_band, b_view, b_slop, b_prec, b_gist)):
        b.grid(row=0, column=i, padx=3)
    tk.Label(ctrl, text="K", fg="#ddd", bg="#101014").grid(row=0, column=5, padx=(12, 0))
    s_k = tk.Scale(ctrl, from_=8, to=64, orient="horizontal", bg="#101014",
                   fg="#ddd", command=set_k, length=130)
    s_k.set(24); s_k.grid(row=0, column=6)

    def to_photo(img_t, bands=None):
        arr = (img_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((VIEW, VIEW), Image.NEAREST)
        if bands:
            dr = ImageDraw.Draw(im)
            for pts, fl, col in bands:
                if pts is None: continue
                P = pts.cpu().numpy()
                Fl = fl.cpu().numpy() if fl is not None else None
                # after a K-slider change, probes reseed a frame before flow
                # catches up — draw flow lines only when the counts agree.
                draw_flow = Fl is not None and len(Fl) == len(P)
                for i, (x, y) in enumerate(P):
                    dr.ellipse([x*VIEW-3, y*VIEW-3, x*VIEW+3, y*VIEW+3], outline=col, width=2)
                    if draw_flow:
                        dr.line([x*VIEW, y*VIEW, (x+Fl[i,0]*8)*VIEW, (y+Fl[i,1]*8)*VIEW],
                                fill=col, width=2)
        return ImageTk.PhotoImage(im)

    def grab():
        if cap is not None:
            import cv2
            ok, fr = cap.read()
            if not ok: return None
            h, w, _ = fr.shape; s = min(h, w)
            fr = fr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
            fr = cv2.cvtColor(cv2.resize(fr, (H, H)), cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(fr).float().permute(2, 0, 1).to(DEVICE) / 255.0
            if slop["left"] > 0:
                slop["left"] -= 1; t = torch.rand_like(t)
            return t
        img, _ = world.frame(); return img

    n = {"t": 0}

    def tick():
        if running["on"]:
            frame = grab()
            if frame is not None:
                if not booted["done"] and has_enc:
                    ctx.prev_frame = frame; ctx.bootstrap(frame); booted["done"] = True
                ctx.step(frame); n["t"] += 1
                ph_w = to_photo(frame, [(ctx.pc, ctx.flow_c, "#00e5ff"),
                                        (ctx.pf, ctx.flow_f, "#ff45d0")])
                lab_world.configure(image=ph_w); lab_world.image = ph_w
                if DEVICE == "cuda" or n["t"] % 2 == 0:
                    if view["mode"] == "full":
                        bimg = ctx.belief_render()
                    else:
                        with torch.no_grad():
                            raw = ctx.vae.dec(ctx.z)
                            idx = ctx.coarse_idx if view["mode"] == "coarse" else ctx.fine_idx
                            bimg = render_full_subset(ctx.vae.ren, raw, idx)
                    ph_b = to_photo(bimg)
                    lab_belief.configure(image=ph_b); lab_belief.image = ph_b
                tele.config(text=(
                    f"bands {ctx.bands:6s}  prec_c {ctx.prec_c:4.2f} prec_f {ctx.prec_f:4.2f}  "
                    f"|flow_c| {ctx.mag_c:5.3f} |flow_f| {ctx.mag_f:5.3f}  "
                    f"|dz| {ctx.dz:5.3f}  Kc {ctx.n_coarse} Kf {ctx.n_fine}  t {n['t']}"))
        root.after(40, tick)

    tick(); root.mainloop()
    if cap is not None:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--diagnostic", action="store_true",
                    help="render full|coarse|fine of the field to a PNG and exit")
    args = ap.parse_args()
    if args.diagnostic:
        band_diagnostic(args.model, args.seed)
    elif args.selftest:
        selftest(args.seed)
    else:
        run_gui(args.model, args.webcam, args.cam)


if __name__ == "__main__":
    main()
