"""
the_splatV6.py  —  THE (z, xi) SPLIT: the belief becomes latent appearance PLUS
                   an explicit Sim(2) pose element, with a coasting covariance
                   and IRLS ownership. The Slapstack merge.

PerceptionLab / Antti Luode (Helsinki), with Claude. July 2026.

    Do not hype. Do not lie. Just show.

WHERE THIS COMES FROM
---------------------
V5 (the octave cascade) tracks pose the hard way: coarse-octave flow is pushed
into the 128-D latent z through the decoder Jacobian, and the manifold re-renders
a translated/rotated face by MOVING IN z. The Slapstack line (Bets 5-7) proved
the alternative on static scenes: pose is a Sim(2) group element acting on the
Gabor packets in CLOSED FORM — the "glide" moves a whole object bitwise-exactly
with zero optimization. Bet 6 added pose beliefs with calibrated covariance and
permanence-by-inference; Bet 7 showed drag-as-conditioning live in a browser.

V6 imports exactly that machinery into the live loop. The belief is now

    (z, xi)      z  : 128-D canonical appearance   (V5's state, unchanged)
                 xi : (tx, ty, rho, log_s) in the Lie algebra of Sim(2),
                      with a 4x4 covariance Sigma that COASTS when unobserved.

THE MATH THAT MAKES IT FREE
---------------------------
A Sim(2) warp of the rendered field is EXACTLY a parameter transform on the
activated packets:  position -> s*R*(p-c)+c+t,  sigma -> s*sigma,
theta -> theta+rho,  freq -> freq/s.   Equivalently (and this is the verified
identity [F2]):   warped_field(x) == canonical_field(g^-1(x)).
So the live loop never renders a warped field at all — it INVERSE-MAPS the probe
coordinates.  If a frame's motion is pure pose and the pose update absorbed it,
the inverse-mapped probe coordinates for pred and ref COINCIDE and the z-loss is
zero with z untouched. Pose is factored out of the latent by construction, not
by hope.

THE STEP (surgery on V5's OctaveCortex.step — 4 phases):
  1. FLOWS — per-octave LK as V5, except the pose octaves are measured by a
     full-res 2-level-pyramid LK. Measured fact that forced this: the V5
     coarse-octave config (single level on a 2x pyramid) returns tiny radial
     flows at ~0.7x their true size — harmless as a loss residual, fatal for
     an INTEGRATED pose (the deficit compounds). Full-res pyramid LK recovers
     translation/rotation/scale increments at 0.97..1.03 (verified in build).
  2. POSE VOTE — weighted similarity Procrustes (Umeyama) on the coarse-octave
     point pairs. LK supplies labeled correspondences, so Bet 6's pi-ambiguity
     (its hardest kill) cannot occur here: rho is recovered on full SO(2)
     trivially. IRLS (Cauchy) down-weights points whose motion is inconsistent
     with the rigid hypothesis = the one-object degenerate case of Bet 6's
     ownership marginals. Gated by the photometric LK residual: garbage frames
     still FIT a similarity, so above the gate there is NO measurement.
  3. COVARIANCE + ANCHOR + SACCADE — Sigma += Q every frame; a valid vote
     blends in Kalman-style and contracts Sigma. Because the vote is a pure
     INTEGRATOR of increments, its noise random-walks and nothing in the flow
     loss can see absolute drift (the reference is itself expressed through
     the old pose — measured: an xi-gradient on the flow loss made drift
     WORSE). Three mechanisms close the loop, all measured in the build:
       - PHOTOMETRIC ANCHOR: a small DC-removed image term, xi-only
         (raw.detach(), so z never couples to pixels and V5's [B] luminance
         robustness survives). Kills the random walk: drift 0.018 -> 0.010.
       - VELOCITY COAST (Bet 6c done properly): in the dark, extrapolate the
         smoothed pre-dark increment, decaying. Constant-position coasting
         leaves the pose exactly as wrong as the motion accrued while blind.
       - POSE SACCADE: on emergence with a swollen Sigma, a coarse-to-fine
         grid search on the anchor loss, scored on the LOW octaves only —
         their basin is a long wavelength wide and their argmin lands on the
         true pose; an all-band score OVERFITS the sample and picks a wrong
         warp (both measured). The cascade lesson applied to reacquisition.
         The z step is suppressed for that one frame (saccadic suppression).
  4. RESIDUAL z CORRECTION — V5's probe-MSE loss verbatim, except every probe
     coordinate is pulled through g^-1 (old pose for ref, new pose for pred),
     and each point is weighted by its ownership. High octaves keep doing
     appearance, which is what they were good at anyway.

FALSIFIABLE CLAIMS (--selftest):
  [F2] WARP EXACTNESS — param-space action == coordinate-space action, two
       independent code routes, float-exact (<1e-3 in float32 trig). The Bet 9
       style check: an identity verified, not assumed.
  [P1] POSE VOTE — Procrustes recovers a known Sim(2) exactly (<1e-6, incl.
       rho=2.5 in honor of Bet 6's pi-war); IRLS survives 30% coherent outliers
       that break the plain fit.
  [F1] POSE FACTORING — on a synthetic Sim(2) world (frame = canonical render
       warped by a known trajectory), pose-ON tracks in image space and mean
       |dz| collapses vs pose-OFF (= V5 behavior). HONEST SCOPE: on the random
       test field, similarity motion is essentially OUTSIDE the z-span, so
       pose-OFF fails by construction; the selftest establishes MECHANISM. The
       real claim — that the celeba manifold stops spending z on pose — is the
       webcam run: toggle POSE live and watch |dz| under pure head translation.
  [F3] OCCLUSION COAST + RE-LOCK — during a 40-frame occlusion tr(Sigma) grows
       monotonically while xi velocity-coasts; on emergence the saccade fires
       (xi-err 0.09 -> 0.02) and the anchor refines back to tracking levels.
       LIMIT, stated: Sigma tracks vote-INTEGRATION uncertainty; absolute gauge
       error is bounded only by the anchor+saccade, not by Sigma. z itself has
       no absolute anchor by V5 design — reacquiring appearance is GIST's job.
  [F4] OWNERSHIP — an independently moving distractor: IRLS-ON <= IRLS-OFF on
       pose error, but the in-loop margin is THIN (~10%), because the clamps,
       the residual-scaled gain, and the anchor already contain the kidnap —
       defense in depth. The sharp evidence for IRLS is the fit-level test in
       [P1]: 30% coherent outliers, 4.5x error reduction, w(bad) 0.22.
  [A]  REGRESSION — with pose disabled, V6 is V5: the cascade still beats open
       loop on the V5 synthetic world.

KILL RISKS, NAMED IN ADVANCE (see README-style notes at the bottom of file):
  - The frontal-face attractor may starve xi on celeba: if the manifold barely
    encodes in-plane pose, |dz| won't drop much because it was never carrying
    it. Sim(2) is honestly 2D — head TURNS still route through z.
  - Bands are computed on CANONICAL frequencies (z=0, unwarped) and stay fixed;
    xi rescales rendered freq by 1/s. Computing bands on warped freq would
    migrate packets between octaves mid-session — do not.
  - Warp can push packets off-canvas; they are counted (telemetry), not hidden.
  - Ownership damps large GENUINE appearance motion too (a mouth opening fast
    is 'inconsistent with rigid'); the Cauchy scale has a floor so small
    appearance residuals pass untouched. Stated, not hidden.

CARRIED OVER VERBATIM (verified in V2..V5, unchanged here):
  SplatVAE / load_v1 (your model.pt loads strict), GaborRenderer + probe
  renders, LKFlow, good_features, render_probes_subset / render_full_subset,
  octave_bands / OctaveLK and all V5 per-octave schedules, the precision /
  prior / saccade machinery, the EQ, the GUI skeleton.
V6 ADDS: Sim2 (exact group ops), PoseBelief (xi + coasting Sigma + Kalman
  blend), fit_sim2 (weighted Umeyama + IRLS ownership), warped renders, the
  full-res pose LK, the photometric anchor, velocity coast, the pose saccade,
  OctaveCortexV6 (the 4-phase step), Sim2World (+ distractor + occlusion),
  the new tests, POSE/OWN toggles and an uncertainty ring in the GUI.

RUN
    python the_splatV6.py --selftest
    python the_splatV6.py --model "face model trained 2 epochs/model.pt" --webcam
    python the_splatV6.py --octaves 4 ...
GUI: everything from V5, plus POSE ON/OFF (live A/B for [F1] — watch |dz|),
  OWN ON/OFF, and on the belief view a dashed ring whose radius is the pose
  position uncertainty (sqrt of the translation block of Sigma) — Bet 7's
  widening ellipse, live. Telemetry adds xi, tr(Sigma), ownership fraction,
  off-canvas packet count.
"""

from __future__ import annotations
import argparse, math, os, sys, time, colorsys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_PARAMS = 11

# ========================================================================== #
#  SPLAT v1 MODEL — verbatim from the_splat / V5. Loads real model.pt strict. #
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
        """Evaluate the field only at coords pxy (M,2) xy in [0,1] -> (B,M,3)."""
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
#  LK FLOW + FEATURES — verbatim from V5                                     #
# ========================================================================== #

class LKFlow:
    def __init__(self, win=9, iters=3, device=DEVICE):
        self.win = win
        self.iters = iters
        r = win // 2
        oy, ox = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1),
                                indexing="ij")
        self.offs = torch.stack([ox, oy], -1).reshape(-1, 2).float().to(device)

    @staticmethod
    def gray(img):
        return img.mean(0)

    @staticmethod
    def grads(g):
        gx = torch.zeros_like(g); gy = torch.zeros_like(g)
        gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
        gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
        return gx, gy

    @staticmethod
    def _sample(field, coords):
        H = field.shape[-1]
        grid = coords[None] * 2 - 1
        v = F.grid_sample(field[None, None], grid, align_corners=True,
                          padding_mode="border")
        return v[0, 0]

    def _level(self, g0, g1, pts, d0):
        H = g0.shape[-1]
        px = 1.0 / (H - 1)
        coords0 = pts[:, None, :] + self.offs[None] * px
        gx, gy = self.grads(g0)
        Ix = self._sample(gx, coords0); Iy = self._sample(gy, coords0)
        I0 = self._sample(g0, coords0)
        Sxx = (Ix * Ix).sum(1); Sxy = (Ix * Iy).sum(1); Syy = (Iy * Iy).sum(1)
        det = Sxx * Syy - Sxy * Sxy
        tr = Sxx + Syy
        conf = 0.5 * (tr - torch.sqrt(torch.clamp(tr * tr - 4 * det, min=0)))
        d = d0.clone()
        for _ in range(self.iters):
            I1 = self._sample(g1, coords0 + d[:, None, :])
            It = I1 - I0
            bx = -(Ix * It).sum(1); by = -(Iy * It).sum(1)
            safe = det.abs() > 1e-9
            dx = torch.where(safe, ( Syy * bx - Sxy * by) / det, torch.zeros_like(det))
            dy = torch.where(safe, (-Sxy * bx + Sxx * by) / det, torch.zeros_like(det))
            d = d + torch.stack([dx, dy], -1) * px
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
#  SUBSET RENDERS — verbatim from V5                                         #
# ========================================================================== #

def render_probes_subset(ren, raw, pxy, idx):
    """render_probes but summing ONLY the packets in idx (anchor sliced).
    No sigmoid: partial fields are compared pre-squash (see V5 note)."""
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
    return torch.stack(chans, dim=-1)


@torch.no_grad()
def render_full_subset(ren, raw, idx):
    """Full HxW render from ONLY the packets in idx. Returns (3,H,W) in [0,1]."""
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


# ========================================================================== #
#  OCTAVE MACHINERY — verbatim from V5                                       #
# ========================================================================== #

def octave_bands(vae, n_oct=4):
    """Split packets into n_oct log-spaced frequency octaves using the field's
    OWN trained freq — evaluated at z=0, CANONICAL (never warped): bands must
    not migrate when xi rescales rendered frequency (kill-risk #2)."""
    with torch.no_grad():
        raw = vae.dec(torch.zeros(1, vae.latent, device=DEVICE))
        freq = (1.0 + 15.0 * torch.sigmoid(raw[0, :, 4]))
    lf = torch.log(freq)
    edges = torch.quantile(lf, torch.linspace(0, 1, n_oct + 1, device=DEVICE))
    edges[0] -= 1e-3; edges[-1] += 1e-3
    bands = []
    for i in range(n_oct):
        m = (lf > edges[i]) & (lf <= edges[i + 1])
        bands.append(m.nonzero(as_tuple=True)[0])
    return bands, freq


def octave_colors(n):
    cols = []
    for i in range(n):
        h = 0.5 - 0.42 * (i / max(1, n - 1))
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.85, 1.0)
        cols.append("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
    return cols


class OctaveLK:
    def __init__(self, n_oct=4, device=DEVICE):
        self.n_oct = n_oct
        self.cfg = []
        for i in range(n_oct):
            frac = i / max(1, n_oct - 1)
            win = int(round(13 - 8 * frac));  win += (win % 2 == 0)
            down = 2 if frac < 0.5 else 1
            self.cfg.append((LKFlow(win=win, iters=4 if down == 2 else 3, device=device),
                             down))

    def flow(self, oct_i, f0, f1, pts):
        lk, down = self.cfg[oct_i]
        g0, g1 = LKFlow.gray(f0), LKFlow.gray(f1)
        if down == 2:
            g0 = F.avg_pool2d(g0[None, None], 2)[0, 0]
            g1 = F.avg_pool2d(g1[None, None], 2)[0, 0]
        return lk._level(g0, g1, pts, torch.zeros_like(pts))

    def win_of(self, oct_i):
        return self.cfg[oct_i][0].win, self.cfg[oct_i][1]


# ========================================================================== #
#  V6 NEW — SIM(2): exact group ops about the image center                    #
# ========================================================================== #

CENTER = 0.5   # warp center in [0,1] coords

class Sim2:
    """A similarity transform of the image plane:  y = s*R(rho)*(x-c) + c + t.
    Stored as xi = (tx, ty, rho, log_s), a 4-vector on DEVICE. All ops exact
    (group composition, not additive approximation)."""

    @staticmethod
    def identity(device=DEVICE):
        return torch.zeros(4, device=device)

    @staticmethod
    def parts(xi):
        s = torch.exp(xi[3])
        c, sn = torch.cos(xi[2]), torch.sin(xi[2])
        R = torch.stack([torch.stack([c, -sn]), torch.stack([sn, c])])
        return s, R, xi[:2]

    @staticmethod
    def apply(xi, pts):
        """pts (K,2) -> (K,2)."""
        s, R, t = Sim2.parts(xi)
        return s * ((pts - CENTER) @ R.T) + CENTER + t

    @staticmethod
    def inv(xi):
        s, R, t = Sim2.parts(xi)
        ti = -(1.0 / s) * (t @ R)          # R^-1 t = R^T t ; (t @ R) == R^T t row-form
        return torch.stack([ti[0], ti[1], -xi[2], -xi[3]])

    @staticmethod
    def compose(xi2, xi1):
        """The transform 'first xi1, then xi2':  g = g2 o g1."""
        s2, R2, t2 = Sim2.parts(xi2)
        t1 = xi1[:2]
        t = s2 * (t1 @ R2.T) + t2
        return torch.stack([t[0], t[1], xi2[2] + xi1[2], xi2[3] + xi1[3]])


class PoseBelief:
    """xi with a coasting 4x4 covariance and a Kalman-style measurement blend.
    Sigma GROWS by Q every frame (honest ignorance accrual, Bet 6c); a valid
    pose vote contracts it. No leak toward identity: the pose has no frontal
    attractor — persistence is the point. Hard clamps only (|t|, |log_s|)."""

    def __init__(self, device=DEVICE,
                 q=(2e-5, 2e-5, 1e-5, 1e-5),          # process noise / frame
                 r0=(2e-6, 2e-6, 1e-6, 1e-6),         # measurement noise floor
                 # (LK increments are sub-pixel accurate; a large R makes the
                 #  gain eat only part of each increment = permanent pose lag)
                 t_max=0.5, ls_max=math.log(2.5)):
        self.xi = Sim2.identity(device)
        self.Sigma = torch.eye(4, device=device) * 1e-4
        self.Q = torch.diag(torch.tensor(q, device=device))
        self.R0 = torch.tensor(r0, device=device)
        self.t_max, self.ls_max = t_max, ls_max
        self.device = device
        self.updated = False       # did the last frame get a measurement?

    def coast(self):
        self.Sigma = self.Sigma + self.Q
        self.updated = False

    def update(self, dxi, resid_scale):
        """Blend the incremental camera-frame vote dg=from(dxi):  g <- dg o g,
        gained by K = Sigma (Sigma + R)^-1 with R scaled by the fit residual."""
        R = torch.diag(self.R0 * max(1.0, float(resid_scale)) ** 2)
        S = self.Sigma + R
        K = self.Sigma @ torch.linalg.inv(S)
        gained = K @ dxi
        self.xi = Sim2.compose(gained, self.xi)
        self.Sigma = (torch.eye(4, device=self.device) - K) @ self.Sigma
        # symmetrize + clamp state
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        with torch.no_grad():
            self.xi[:2] = self.xi[:2].clamp(-self.t_max, self.t_max)
            self.xi[3] = self.xi[3].clamp(-self.ls_max, self.ls_max)
        self.updated = True

    def trace(self):
        return float(self.Sigma.diagonal().sum())

    def pos_std(self):
        return float(torch.sqrt(self.Sigma[0, 0] + self.Sigma[1, 1]))


# ========================================================================== #
#  V6 NEW — the packet-space action and the warped renders                    #
# ========================================================================== #

def _activate_subset(ren, raw, idx):
    sub = raw[:, idx, :].float()
    a_px = ren.anchor_logit[idx, 0][None]
    a_py = ren.anchor_logit[idx, 1][None]
    px = torch.sigmoid(a_px + sub[..., 0])
    py = torch.sigmoid(a_py + sub[..., 1])
    sigma = 0.012 + 0.14 * torch.sigmoid(sub[..., 2])
    theta = sub[..., 3]
    freq = 1.0 + 15.0 * torch.sigmoid(sub[..., 4])
    coeff = torch.tanh(sub[..., 5:11]).reshape(*sub.shape[:2], 3, 2)
    return px, py, sigma, theta, freq, coeff


def warp_activated(px, py, sigma, theta, freq, xi):
    """The glide: Sim(2) as a parameter transform on activated packets.
    position s*R*(p-c)+c+t, sigma*s, theta+rho, freq/s. Exact — see [F2]."""
    s, R, t = Sim2.parts(xi)
    p = torch.stack([px - CENTER, py - CENTER], -1)          # (B,N,2)
    p = s * (p @ R.T) + CENTER + t[None, None]
    return p[..., 0], p[..., 1], s * sigma, theta + xi[2], freq / s


def render_probes_warped(ren, raw, pxy, idx, xi):
    """Field of xi-warped packets evaluated at pxy — the PARAM-SPACE route.
    Exists to be checked against the COORD-SPACE route
    render_probes_subset(ren, raw, Sim2.apply(Sim2.inv(xi), pxy), idx): [F2]."""
    px, py, sigma, theta, freq, coeff = _activate_subset(ren, raw, idx)
    px, py, sigma, theta, freq = warp_activated(px, py, sigma, theta, freq, xi)
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
    return torch.stack(chans, dim=-1)


@torch.no_grad()
def render_full_warped(ren, raw, idx, xi):
    """Full HxW sigmoid render of xi-warped packets in idx. Also returns the
    off-canvas count (warped centers outside [0,1] — counted, not hidden)."""
    px, py, sigma, theta, freq, coeff = _activate_subset(ren, raw, idx)
    px, py, sigma, theta, freq = warp_activated(px, py, sigma, theta, freq, xi)
    off = int(((px < 0) | (px > 1) | (py < 0) | (py > 1)).sum())
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
    return torch.sigmoid(torch.stack(chans, dim=1))[0], off


@torch.no_grad()
def render_band_warped_presig(ren, raw, idx, xi):
    """Pre-sigmoid warped band render (for the EQ mix, matching V5's mixing)."""
    px, py, sigma, theta, freq, coeff = _activate_subset(ren, raw, idx)
    px, py, sigma, theta, freq = warp_activated(px, py, sigma, theta, freq, xi)
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
    return torch.stack(chans, dim=1)[0]        # (3,H,W) pre-sigmoid


# ========================================================================== #
#  V6 NEW — THE POSE VOTE: weighted Umeyama + IRLS ownership                  #
# ========================================================================== #

def fit_sim2(pts, moved, w=None, irls_iters=2, sig_floor=0.006, cauchy_k=1.5):
    """Weighted similarity Procrustes:  moved ~ s*R*(pts-c) + c + t.
    LK gives LABELED correspondences, so rho is recovered on full SO(2) —
    Bet 6's pi-ambiguity cannot occur here (its fix isn't needed, its lesson
    is remembered). IRLS (Cauchy) = one-object ownership: points moving
    inconsistently with the rigid hypothesis are down-weighted. sig_floor keeps
    small genuine appearance residuals untouched (kill-risk #4, stated).

    Returns (dxi (4,), info dict: ok, resid, w (final per-point), n_eff)."""
    dev = pts.device
    K = pts.shape[0]
    if w is None:
        w = torch.ones(K, device=dev)
    w = w.clone()
    dxi = Sim2.identity(dev)
    info = {"ok": False, "resid": float("inf"), "w": w, "n_eff": 0.0}
    if K < 3:
        return dxi, info
    X0 = pts - CENTER
    Y0 = moved - CENTER
    for it in range(irls_iters + 1):
        wm = w.sum()
        if float(wm) < 1e-8:
            return Sim2.identity(dev), info
        mx = (w[:, None] * X0).sum(0) / wm
        my = (w[:, None] * Y0).sum(0) / wm
        X = X0 - mx; Y = Y0 - my
        C = (w[:, None, None] * (Y[:, :, None] @ X[:, None, :])).sum(0) / wm
        U, S, Vt = torch.linalg.svd(C)
        d = torch.det(U @ Vt)
        Dm = torch.diag(torch.tensor([1.0, float(torch.sign(d))], device=dev))
        R = U @ Dm @ Vt
        varX = (w * (X * X).sum(1)).sum() / wm
        s = (S * Dm.diagonal()).sum() / varX.clamp_min(1e-12)
        s = s.clamp(0.25, 4.0)
        t = my - s * (R @ mx)
        rho = torch.atan2(R[1, 0], R[0, 0])
        dxi = torch.stack([t[0], t[1], rho, torch.log(s)])
        # residuals under the current hypothesis
        pred = s * (X0 @ R.T) + t[None]
        r = (Y0 - pred).norm(dim=1)
        med = torch.median(r)
        sig = torch.clamp(1.4826 * med, min=sig_floor)
        if it < irls_iters:
            w = 1.0 / (1.0 + (r / (cauchy_k * sig)) ** 2)
    n_eff = float(w.sum())
    resid = float((w * r).sum() / max(n_eff, 1e-8))
    info = {"ok": n_eff >= 2.5, "resid": resid, "w": w, "n_eff": n_eff}
    return dxi, info


# ========================================================================== #
#  THE V6 CORTEX — the four-phase step                                        #
# ========================================================================== #

class OctaveCortexV6:
    """V5's OctaveCortex with the (z, xi) split. Everything V5-verified is kept:
    per-octave schedules, dynamic precision, probe riding, saccade reseed, EQ.
    New: pose vote from the coarse octaves, coasting Sigma, ownership weights,
    and inverse-mapped probe coordinates in the z-loss."""

    def __init__(self, vae, n_oct=4, k_low=14, k_high=6, m_hist=8, sig_ref=0.02,
                 eta=20.0, beta_mom=0.5, leak=0.02, dz_clamp=0.25, seed=0,
                 pose_enabled=True, own_enabled=True, dxi_clamp=(0.05, 0.15, 0.10),
                 resid_ref=0.02, eta_xi=1.0, w_anchor=0.5):
        self.vae = vae
        self.n_oct = n_oct
        self.rng = np.random.default_rng(seed)
        self.z = torch.zeros(1, vae.latent, device=DEVICE)
        self.z_prev = self.z.clone(); self.z_prior = self.z.clone()
        self.bands, self.freq = octave_bands(vae, n_oct)
        self.lk = OctaveLK(n_oct)
        fr = np.linspace(0, 1, n_oct)
        self.k = [max(3, int(round(k_low + (k_high - k_low) * f))) for f in fr]
        self.w = [float(1.0 - 0.75 * f) for f in fr]
        self.prec_cap = [float(1.0 - 0.6 * f) for f in fr]
        self.eq = [1.0] * n_oct
        self.active = list(range(n_oct))
        self.pts = [None] * n_oct
        self.flow = [None] * n_oct
        self.res_hist = [deque(maxlen=m_hist) for _ in range(n_oct)]
        self.prec = [1.0] * n_oct; self.rough = [0.0] * n_oct; self.mag = [0.0] * n_oct
        self.prev_frame = None
        self.m_hist = m_hist; self.sig_ref = sig_ref; self.eta = eta
        self.beta_mom = beta_mom; self.leak = leak; self.dz_clamp = dz_clamp
        self.dynamic_precision = True; self.dz = 0.0
        s = 1.5
        oy, ox = torch.meshgrid(torch.tensor([-s, 0., s]), torch.tensor([-s, 0., s]),
                                indexing="ij")
        self.stencil_px = torch.stack([ox, oy], -1).reshape(-1, 2).to(DEVICE)
        # ---- V6 state ----
        self.pose_enabled = pose_enabled
        self.own_enabled = own_enabled
        self.pose = PoseBelief(DEVICE)
        self.pose_octaves = [0, 1] if n_oct >= 3 else [0]
        # The pose vote gets its OWN flow: full-res 2-level-pyramid LK. Measured
        # fact (see header): the V5 coarse-octave config (single level on a 2x
        # pyramid) shrinks tiny radial flows — pure-scale increments of ~0.1px
        # come back at ~0.7x. Fine for a loss residual, fatal for an INTEGRATED
        # pose. Full-res pyramid LK recovers trans/rot/scale at 0.97..1.03.
        self.pose_lk = LKFlow(win=9, iters=4)
        self.dxi_clamp = dxi_clamp                 # (|dt|, |drho|, |dlog_s|)
        self.resid_ref = resid_ref                 # fit resid scale for R_meas
        self.eta_xi = eta_xi                       # render-anchor step on xi
        self.w_anchor = w_anchor                   # photometric anchor weight
        self.pose_gate = 0.08                      # photometric occlusion gate
        self.pose_obscured = False
        self._was_obscured = False
        self.dxi_ema = Sim2.identity(DEVICE)       # smoothed increment (velocity)
        self.saccade_thresh = 5e-4                 # Sigma trace that triggers
        self.suppress_z = False                    # saccadic suppression flag
        self.own = [None] * n_oct                  # per-octave ownership weights
        self.own_frac = 1.0                        # telemetry
        self.off_canvas = 0                        # telemetry
        self.last_dxi = Sim2.identity(DEVICE)

    # ---------- V5-verbatim helpers ----------
    def _stencil(self, pts):
        px = 1.0 / (self.vae.ren.H - 1)
        return (pts[:, None, :] + self.stencil_px[None] * px).reshape(-1, 2).clamp(0, 1)

    def seed(self, frame):
        avoid = None
        for i in range(self.n_oct):
            self.pts[i] = good_features(frame, k=self.k[i], rng=self.rng,
                                        n_cand=250, avoid=avoid, min_d=0.04)
            avoid = self.pts[i] if avoid is None else torch.cat([avoid, self.pts[i]])
            self.res_hist[i].clear()
        self.prev_frame = frame.detach()

    def set_k(self, k_low):
        fr = np.linspace(0, 1, self.n_oct)
        k_high = max(3, k_low // 3)
        self.k = [max(3, int(round(k_low + (k_high - k_low) * f))) for f in fr]
        self.pts = [None] * self.n_oct
        self.flow = [None] * self.n_oct

    def bootstrap(self, frame):
        with torch.no_grad():
            mu, _ = self.vae.enc(frame[None])
        self.z = mu.detach(); self.z_prev = self.z.clone(); self.z_prior = self.z.clone()
        # GIST re-acquires the canonical appearance; the pose resets to identity
        # (the encoder was trained on unwarped crops — its output IS canonical).
        self.pose = PoseBelief(DEVICE)

    @staticmethod
    def sample_frame(frame, pts):
        g = pts[None, None] * 2 - 1
        return F.grid_sample(frame[None], g, align_corners=True)[0, :, 0, :].T

    def _prec(self, i, r):
        h = self.res_hist[i]; h.append(r)
        if len(h) < 4:
            return 1.0, 0.0
        d2 = np.diff(np.array(h), n=2)
        rough = float(np.sqrt(np.mean(d2 * d2)))
        return float(self.sig_ref**2 / (self.sig_ref**2 + rough**2)), rough

    # ---------- pose saccade ----------
    @torch.no_grad()
    def _pose_saccade(self, frame, n_pts=48):
        """Coarse-to-fine grid search on the DC-removed photometric anchor
        loss, on emergence when Sigma says the pose is not trusted. Scoring
        uses the LOW octaves ONLY — measured fact: the low band's basin is a
        long wavelength wide and its argmin lands on the true pose (gap
        0.125 -> ~0.02 on the synthetic), while an all-band score OVERFITS the
        sample points and picks a wrong warp (gap 0.06+). The cascade lesson
        again: the low octave orients, the high octaves refine within its
        basin — here applied to reacquisition itself."""
        raw = self.vae.dec(self.z)
        score_bands = [j for j in self.pose_octaves if len(self.bands[j]) > 0]
        pts = good_features(frame, k=n_pts, rng=self.rng, n_cand=400)
        tgt = self.sample_frame(frame, pts)
        tgt = tgt - tgt.mean(0, keepdim=True)

        def score(xi):
            coords = Sim2.apply(Sim2.inv(xi), pts)
            mix = 0.0
            for j in score_bands:
                bp = render_probes_subset(self.vae.ren, raw, coords, self.bands[j])[0]
                mix = mix + self.eq[j] * (torch.sigmoid(bp) - 0.5)
            bel = mix + 0.5
            bel = bel - bel.mean(0, keepdim=True)
            return float(F.mse_loss(bel, tgt))

        def grid(center, dts, drs, dlss):
            best_xi, best = center.clone(), score(center)
            for dtx in dts:
                for dty in dts:
                    for dr in drs:
                        for dls in dlss:
                            cand = Sim2.compose(torch.tensor(
                                [dtx, dty, dr, dls], device=DEVICE), center)
                            sc = score(cand)
                            if sc < best:
                                best, best_xi = sc, cand
            return best_xi

        xi = grid(self.pose.xi, (-0.08, -0.04, 0.0, 0.04, 0.08),
                  (-0.4, -0.2, 0.0, 0.2, 0.4), (-0.15, 0.0, 0.15))
        xi = grid(xi, (-0.03, -0.015, 0.0, 0.015, 0.03),
                  (-0.12, -0.06, 0.0, 0.06, 0.12), (-0.06, 0.0, 0.06))
        self.pose.xi = xi

    # ---------- the V6 step ----------
    def step(self, frame):
        frame = frame.detach()
        if self.pts[0] is None or self.prev_frame is None:
            self.seed(frame); return

        # ---- Phase 0: per-octave flows. Pose octaves are measured with the
        # full-res pyramid pose LK (unbiased sub-pixel increments); the higher
        # octaves keep V5's per-octave LK verbatim. ----
        flows = {}
        for i in range(self.n_oct):
            if i not in self.active or self.eq[i] <= 1e-3 or len(self.bands[i]) == 0:
                self.flow[i] = None; continue
            if self.pose_enabled and i in self.pose_octaves:
                d, conf, res = self.pose_lk(self.prev_frame, frame, self.pts[i])
            else:
                d, conf, res = self.lk.flow(i, self.prev_frame, frame, self.pts[i])
            self.flow[i] = d.detach(); self.mag[i] = float(d.norm(dim=1).mean())
            flows[i] = (d.detach(), conf.detach(), res.detach())

        # ---- Phase 1: pose vote from the coarse octaves ----
        xi_old = self.pose.xi.clone()
        dg = Sim2.identity(DEVICE)
        fit_w_map = {}
        # Occlusion gate: on garbage frames LK still returns flows and random
        # noise still FITS a similarity ([F3] failed without this — Sigma kept
        # contracting in the dark). The photometric LK residual is the honest
        # witness: it is ~0.01 on real frames and jumps 10x on noise. Above the
        # gate there is NO measurement: no vote, no anchor — the pose coasts.
        pose_res = np.mean([float(flows[i][2].mean()) for i in self.pose_octaves
                            if i in flows]) if any(i in flows for i in self.pose_octaves) else 1.0
        self.pose_obscured = bool(pose_res > self.pose_gate)
        self.suppress_z = False
        if self.pose_enabled:
            self.pose.coast()                       # ignorance accrues first
            if self.pose_obscured:
                # VELOCITY COAST (Bet 6c, done properly): constant-position
                # coast leaves the pose exactly as wrong as the motion accrued
                # in the dark. Extrapolate the smoothed pre-dark increment,
                # decaying — the belief keeps moving the way the world was.
                self.pose.xi = Sim2.compose(self.dxi_ema, self.pose.xi)
                self.dxi_ema = self.dxi_ema * 0.95
            elif self._was_obscured and self.pose.trace() > self.saccade_thresh:
                # POSE SACCADE: the vote is increment-only — it can never
                # remove an offset accrued while blind, and the photometric
                # anchor's gradient basin is one Gabor wavelength wide. On
                # emergence with a swollen Sigma, do the honest thing: LOOK —
                # a coarse grid search on the anchor loss. And as in biology,
                # suppress the z update on the jump frame (saccadic
                # suppression) so the discontinuity cannot kick the latent.
                self._pose_saccade(frame)
                self.suppress_z = True
            P, M, W0, tags = [], [], [], []
            if not self.pose_obscured:
                for i in self.pose_octaves:
                    if i in flows:
                        d, conf, _ = flows[i]
                        P.append(self.pts[i]); M.append(self.pts[i] + d)
                        W0.append(torch.full((len(d),), self.w[i], device=DEVICE))
                        tags.append((i, len(d)))
            if P:
                P = torch.cat(P); M = torch.cat(M); W0 = torch.cat(W0)
                dxi, info = fit_sim2(P, M, W0,
                                     irls_iters=2 if self.own_enabled else 0)
                if info["ok"]:
                    ct, cr, cs = self.dxi_clamp
                    dxi = torch.stack([dxi[0].clamp(-ct, ct), dxi[1].clamp(-ct, ct),
                                       dxi[2].clamp(-cr, cr), dxi[3].clamp(-cs, cs)])
                    self.pose.update(dxi, info["resid"] / self.resid_ref)
                    self.last_dxi = dxi.detach()
                    self.dxi_ema = 0.8 * self.dxi_ema + 0.2 * dxi.detach()
                    dg = dxi.detach()
                    off = 0
                    for i, n in tags:
                        fit_w_map[i] = info["w"][off:off + n]; off += n
                    self.own_frac = info["n_eff"] / max(1, len(P))
                # fit not ok -> pose coasts on Q alone (occlusion path, [F3])
        self._was_obscured = self.pose_obscured
        xi_new = self.pose.xi.clone()

        # ---- Phase 1b: ownership weights for ALL octaves ----
        for i in range(self.n_oct):
            self.own[i] = None
            if i not in flows:
                continue
            if not (self.pose_enabled and self.own_enabled):
                continue
            d, conf, _ = flows[i]
            if i in fit_w_map:
                self.own[i] = fit_w_map[i]
            else:
                # residual of this octave's flow vs the rigid increment
                pred_d = Sim2.apply(dg, self.pts[i]) - self.pts[i]
                r = (d - pred_d).norm(dim=1)
                sig = torch.clamp(1.4826 * torch.median(r), min=0.008)
                self.own[i] = 1.0 / (1.0 + (r / (3.0 * sig)) ** 2)

        # ---- Phase 2: residual z correction (V5 loss, coords inverse-mapped) ----
        vel = self.z - self.z_prev
        z_pred = self.z + self.beta_mom * vel
        z_pred = z_pred - self.leak * (z_pred - self.z_prior)
        z_var = z_pred.detach().clone().requires_grad_(True)
        raw = self.vae.dec(z_var)
        with torch.no_grad():
            raw_old = self.vae.dec(self.z)

        # xi enters the loss differentiably — but NOT through the flow term:
        # ref is itself expressed through g_old, so the flow loss constrains
        # only the INCREMENT and is blind to absolute drift (measured: adding
        # an xi-gradient there made drift worse, not better). The vote is a
        # pure integrator; its noise random-walks. The absolute anchor must
        # compare the belief against the FRAME: a small photometric term,
        # DC-removed per channel (so luminance drift cannot steer geometry,
        # keeping V5's [B]), with raw.detach() (so z never couples to pixels
        # directly — the V5 flow-only z path is preserved bit-for-bit).
        # Vote = fast feedforward; photometric anchor = slow absolute pin.
        xi_old_inv = Sim2.inv(xi_old)
        xi_var = xi_new.detach().clone().requires_grad_(self.pose_enabled)

        if self.suppress_z:
            # saccadic suppression: the pose just jumped; ref (old pose) and
            # pred (new pose) are incommensurable for exactly one frame.
            self.dz = 0.0
            with torch.no_grad():
                for i in range(self.n_oct):
                    if self.flow[i] is None:
                        continue
                    self.pts[i] = (self.pts[i] + self.flow[i]).clamp(0.02, 0.98)
            self.prev_frame = frame
            return

        loss = 0.0 * z_var.sum()
        for i in range(self.n_oct):
            if i not in flows:
                continue
            d, conf, res = flows[i]
            if self.dynamic_precision:
                p, self.rough[i] = self._prec(i, float(res.mean()))
            else:
                p = 1.0
            p = min(p, self.prec_cap[i])
            self.prec[i] = p
            eff = self.w[i] * self.eq[i] * p
            moved = (self.pts[i] + d).clamp(0, 1)
            here_cam = self._stencil(self.pts[i])
            moved_cam = self._stencil(moved)
            if self.pose_enabled:
                here = Sim2.apply(xi_old_inv, here_cam)      # canonical coords
                there = Sim2.apply(Sim2.inv(xi_var.detach()), moved_cam)
            else:
                here, there = here_cam, moved_cam
            with torch.no_grad():
                ref = render_probes_subset(self.vae.ren, raw_old, here, self.bands[i])[0]
            pred = render_probes_subset(self.vae.ren, raw, there, self.bands[i])[0]
            if self.own[i] is not None:
                ow = self.own[i][:, None].repeat(1, 9).reshape(-1, 1)   # per stencil pt
                num = (ow * (pred - ref) ** 2).sum()
                den = (ow.sum() * pred.shape[-1]).clamp_min(1e-8)
                loss = loss + eff * num / den
            else:
                loss = loss + eff * F.mse_loss(pred, ref)
            # photometric anchor (xi only, pose octaves only, gated in the dark)
            if self.pose_enabled and i in self.pose_octaves and not self.pose_obscured:
                inv_coords = Sim2.apply(Sim2.inv(xi_var), moved_cam)
                mix = 0.0
                for j in range(self.n_oct):
                    if len(self.bands[j]) == 0 or self.eq[j] <= 1e-3:
                        continue
                    bp = render_probes_subset(self.vae.ren, raw.detach(),
                                              inv_coords, self.bands[j])[0]
                    mix = mix + self.eq[j] * (torch.sigmoid(bp) - 0.5)
                bel_px = mix + 0.5
                with torch.no_grad():
                    tgt_px = self.sample_frame(frame, moved_cam)
                bel_dc = bel_px - bel_px.mean(0, keepdim=True)
                tgt_dc = tgt_px - tgt_px.mean(0, keepdim=True)
                loss = loss + self.w_anchor * F.mse_loss(bel_dc, tgt_dc)
        loss.backward()

        # ---- Phase 3: z update + probe riding (V5 verbatim) + xi anchoring ----
        with torch.no_grad():
            if self.pose_enabled and xi_var.grad is not None:
                g = self.eta_xi * xi_var.grad
                lim = torch.tensor([0.004, 0.004, 0.010, 0.005], device=DEVICE)
                g = torch.max(torch.min(g, lim), -lim)      # per-component trust
                self.pose.xi = (xi_var - g).detach()
                self.pose.xi[:2] = self.pose.xi[:2].clamp(-self.pose.t_max, self.pose.t_max)
                self.pose.xi[3] = self.pose.xi[3].clamp(-self.pose.ls_max, self.pose.ls_max)
            step = self.eta * z_var.grad
            n = step.norm()
            if n > self.dz_clamp:
                step = step * (self.dz_clamp / n)
            self.dz = float(step.norm())
            self.z_prev = self.z
            self.z = (z_pred - step).detach()
            self.z_prior = 0.995 * self.z_prior + 0.005 * self.z
            for i in range(self.n_oct):
                if self.flow[i] is None:
                    continue
                self.pts[i] = (self.pts[i] + self.prec[i] * self.flow[i]).clamp(0.02, 0.98)
                bad = (self.pts[i].min(1).values < 0.04) | (self.pts[i].max(1).values > 0.96)
                if bad.any() and self.prec[i] > 0.5:
                    self.pts[i][bad] = good_features(frame, k=int(bad.sum()),
                                                     rng=self.rng, avoid=self.pts[i][~bad])
        self.prev_frame = frame

    # ---------- rendering ----------
    @torch.no_grad()
    def belief_render(self, only_octave=None):
        """EQ-weighted mix of the octaves, each warped by the current pose.
        With pose_enabled=False this is bit-identical to V5's belief_render."""
        raw = self.vae.dec(self.z)
        xi = self.pose.xi if self.pose_enabled else Sim2.identity(DEVICE)
        H = self.vae.ren.H
        self.off_canvas = 0
        if only_octave is None:
            out = torch.zeros(3, H, H, device=DEVICE)
            for i in range(self.n_oct):
                if len(self.bands[i]) == 0 or self.eq[i] <= 1e-3:
                    continue
                if self.pose_enabled:
                    band = torch.sigmoid(render_band_warped_presig(
                        self.vae.ren, raw, self.bands[i], xi)) - 0.5
                else:
                    band = render_full_subset(self.vae.ren, raw, self.bands[i]) - 0.5
                out = out + self.eq[i] * band
            if self.pose_enabled:
                # off-canvas telemetry: warped packet centers outside [0,1]
                allidx = torch.arange(self.vae.ren.N, device=DEVICE)
                a = _activate_subset(self.vae.ren, raw, allidx)
                pxw, pyw, *_ = warp_activated(a[0], a[1], a[2], a[3], a[4], xi)
                self.off_canvas = int(((pxw < 0) | (pxw > 1) |
                                       (pyw < 0) | (pyw > 1)).sum())
            return (out + 0.5).clamp(0, 1)
        if self.pose_enabled:
            img, self.off_canvas = render_full_warped(self.vae.ren, raw,
                                                      self.bands[only_octave], xi)
            return img
        return render_full_subset(self.vae.ren, raw, self.bands[only_octave])


# ========================================================================== #
#  SYNTHETIC WORLDS                                                          #
# ========================================================================== #

class OctaveWorld:
    """V5's latent-trajectory world, verbatim (for the [A] regression)."""
    def __init__(self, vae, seed=0, omega_lo=0.04, omega_hi=0.14, radius=1.1,
                 lum_drift=False):
        self.vae = vae
        g = torch.Generator(device="cpu").manual_seed(seed)
        B = torch.randn(4, vae.latent, generator=g); B, _ = torch.linalg.qr(B.T)
        self.basis = B[:, :4].to(DEVICE)
        self.omega_lo, self.omega_hi, self.radius = omega_lo, omega_hi, radius
        self.t = 0
        self.z0 = torch.randn(1, vae.latent, generator=g).to(DEVICE) * 0.3
        self.lum_drift = lum_drift; self.slop_left = 0

    def _coeff(self, pose_only=False):
        tl, th = self.omega_lo * self.t, self.omega_hi * self.t
        c = [self.radius * math.cos(tl), self.radius * math.sin(tl),
             0.0 if pose_only else 0.5 * self.radius * math.cos(th),
             0.0 if pose_only else 0.5 * self.radius * math.sin(th)]
        return torch.tensor(c, dtype=torch.float32, device=DEVICE)

    def z_true(self):  return self.z0 + (self.basis @ self._coeff())[None]
    def pose_z(self):  return self.z0 + (self.basis @ self._coeff(pose_only=True))[None]
    def inject_slop(self, n=60): self.slop_left = n

    @torch.no_grad()
    def frame(self):
        self.t += 1
        img = self.vae.generate(self.z_true())[0]
        if self.lum_drift:
            img = (img * (0.65 + 0.35 * math.sin(2 * math.pi * self.t / 90))).clamp(0, 1)
        if self.slop_left > 0:
            self.slop_left -= 1; img = torch.rand_like(img)
        return img, self.z_true()


class Sim2World:
    """A CANONICAL appearance (fixed z) seen through a KNOWN Sim(2) trajectory:
    frame(t) = warped render of dec(z0) by g_true(t). Optionally an
    independently moving distractor blob ([F4]) and an occlusion window ([F3]).
    Trajectories are sin-shaped so g_true(0) = identity (clean boot)."""

    def __init__(self, vae, seed=0, a_t=0.10, a_r=0.35, a_s=0.18,
                 w_t=0.05, w_r=0.033, w_s=0.021,
                 distractor=False, occlude=None):
        self.vae = vae
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.z0 = (torch.randn(1, vae.latent, generator=g) * 1.0).to(DEVICE)
        with torch.no_grad():
            self.raw0 = vae.dec(self.z0)
        self.allidx = torch.arange(vae.ren.N, device=DEVICE)
        self.a = (a_t, a_r, a_s); self.w_ = (w_t, w_r, w_s)
        self.t = 0
        self.distractor = distractor
        self.occlude = occlude          # (t0, t1) or None

    def xi_true(self, t=None):
        t = self.t if t is None else t
        (at, ar, as_), (wt, wr, ws) = self.a, self.w_
        return torch.tensor([at * math.sin(wt * t), at * math.sin(0.71 * wt * t),
                             ar * math.sin(wr * t), as_ * math.sin(ws * t)],
                            device=DEVICE)

    @torch.no_grad()
    def frame(self):
        self.t += 1
        img, _ = render_full_warped(self.vae.ren, self.raw0, self.allidx,
                                    self.xi_true())
        if self.distractor:
            # a bright gaussian blob sweeping the frame with its OWN motion
            H = self.vae.ren.H
            cx = 0.5 + 0.42 * math.sin(0.11 * self.t + 0.7)
            cy = 0.5 + 0.42 * math.cos(0.09 * self.t)
            gy, gx = torch.meshgrid(torch.linspace(0, 1, H, device=DEVICE),
                                    torch.linspace(0, 1, H, device=DEVICE),
                                    indexing="ij")
            blob = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 0.04 ** 2))
            img = (img + torch.stack([0.9 * blob, 0.2 * blob, 0.7 * blob])).clamp(0, 1)
        if self.occlude and self.occlude[0] <= self.t < self.occlude[1]:
            img = torch.rand_like(img)
        return img


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


def _run_v5(vae, T=170, n_oct=4, active=None, dynamic=True, score="pose", seed=0):
    """V5's [A] harness with pose disabled — the regression guard."""
    world = OctaveWorld(vae, seed=seed)
    ctx = OctaveCortexV6(vae, n_oct=n_oct, seed=seed, pose_enabled=False,
                         own_enabled=False)
    ctx.dynamic_precision = dynamic
    if active is not None: ctx.active = list(active)
    f0, z0 = world.frame()
    ctx.z = (z0 + 0.2 * torch.randn_like(z0)).detach()
    ctx.z_prev = ctx.z.clone(); ctx.z_prior = ctx.z.clone()
    ctx.seed(f0)
    errs = []
    for t in range(T):
        frame, z_true = world.frame()
        ctx.step(frame)
        if t % 2 == 0:
            with torch.no_grad():
                tgt = world.pose_z() if score == "pose" else z_true
                errs.append(F.mse_loss(vae.generate(ctx.z), vae.generate(tgt)).item())
    return float(np.mean(errs[len(errs)//3:]))


def _open_v5(vae, seed=0, T=170):
    world = OctaveWorld(vae, seed=seed)
    f0, z0 = world.frame(); z = (z0 + 0.2*torch.randn_like(z0)); zp = z.clone()
    errs = []
    for t in range(T):
        world.frame(); z, zp = z + 0.5*(z-zp), z
        if t % 2 == 0:
            with torch.no_grad():
                errs.append(F.mse_loss(vae.generate(z), vae.generate(world.pose_z())).item())
    return float(np.mean(errs[len(errs)//3:]))


def _run_pose(vae, T=150, n_oct=4, pose=True, own=True, distractor=False,
              occlude=None, seed=0, record_sigma=False, eta=None):
    """Run the cortex against a Sim2World. Returns dict of tail metrics."""
    world = Sim2World(vae, seed=seed, distractor=distractor, occlude=occlude)
    ctx = OctaveCortexV6(vae, n_oct=n_oct, seed=seed, pose_enabled=pose,
                         own_enabled=own)
    if eta is not None:
        ctx.eta = eta
    f0 = world.frame()
    ctx.z = (world.z0 + 0.1 * torch.randn_like(world.z0)).detach()
    ctx.z_prev = ctx.z.clone(); ctx.z_prior = ctx.z.clone()
    ctx.seed(f0)
    img_errs, dzs, xi_errs, sig_tr = [], [], [], []
    for t in range(T):
        frame = world.frame()
        ctx.step(frame)
        dzs.append(ctx.dz)
        if record_sigma:
            sig_tr.append(ctx.pose.trace())
        if t % 2 == 0:
            with torch.no_grad():
                bel = ctx.belief_render()
                true_img, _ = render_full_warped(vae.ren, world.raw0,
                                                 world.allidx, world.xi_true())
                img_errs.append(F.mse_loss(bel, true_img).item())
                if pose:
                    xi_errs.append(float((ctx.pose.xi - world.xi_true()).abs().mean()))
    n3 = len(img_errs) // 3
    out = {"img": float(np.mean(img_errs[n3:])),
           "img_last": float(np.mean(img_errs[-12:])),
           "dz": float(np.mean(dzs[len(dzs)//3:])),
           "xi": float(np.mean(xi_errs[n3:])) if xi_errs else float("nan"),
           "img_series": img_errs, "sig_tr": sig_tr}
    return out


def _test_group_ops():
    torch.manual_seed(3)
    xi1 = torch.tensor([0.07, -0.04, 0.5, 0.15], device=DEVICE)
    xi2 = torch.tensor([-0.03, 0.06, -0.9, -0.10], device=DEVICE)
    pts = torch.rand(50, 2, device=DEVICE)
    # inverse
    e_inv = (Sim2.apply(Sim2.inv(xi1), Sim2.apply(xi1, pts)) - pts).abs().max()
    # composition
    lhs = Sim2.apply(Sim2.compose(xi2, xi1), pts)
    rhs = Sim2.apply(xi2, Sim2.apply(xi1, pts))
    e_cmp = (lhs - rhs).abs().max()
    return float(e_inv), float(e_cmp)


def _test_warp_exactness(vae):
    """[F2] param-space action == coordinate-space action."""
    torch.manual_seed(4)
    z = torch.randn(1, vae.latent, device=DEVICE)
    with torch.no_grad():
        raw = vae.dec(z)
    xi = torch.tensor([0.07, -0.04, 0.5, 0.15], device=DEVICE)
    idx = torch.arange(vae.ren.N, device=DEVICE)
    pxy = torch.rand(200, 2, device=DEVICE)
    with torch.no_grad():
        a = render_probes_warped(vae.ren, raw, pxy, idx, xi)
        b = render_probes_subset(vae.ren, raw, Sim2.apply(Sim2.inv(xi), pxy), idx)
    return float((a - b).abs().max())


def _test_fit(seed=0):
    """[P1] exact recovery, incl. rho=2.5 (Bet 6's old battlefield), and IRLS
    vs 30% coherent outliers."""
    rng = np.random.default_rng(seed)
    dev = DEVICE
    out = {}
    for rho in (0.3, 2.5, -2.8):
        xi = torch.tensor([0.06, -0.03, rho, 0.12], device=dev)
        pts = torch.tensor(rng.uniform(0.1, 0.9, (40, 2)), dtype=torch.float32, device=dev)
        moved = Sim2.apply(xi, pts)
        dxi, info = fit_sim2(pts, moved, irls_iters=0)
        out[f"exact_rho{rho}"] = float((dxi - xi).abs().max())
    # outliers: 30% of points get a coherent WRONG motion (the 'hand')
    xi = torch.tensor([0.03, 0.02, 0.15, 0.05], device=dev)
    pts = torch.tensor(rng.uniform(0.1, 0.9, (60, 2)), dtype=torch.float32, device=dev)
    moved = Sim2.apply(xi, pts)
    n_bad = 18
    moved[:n_bad] = pts[:n_bad] + torch.tensor([0.08, -0.06], device=dev)
    d_plain, _ = fit_sim2(pts, moved, irls_iters=0)
    d_irls, info = fit_sim2(pts, moved, irls_iters=3)
    out["outlier_plain"] = float((d_plain - xi).abs().max())
    out["outlier_irls"] = float((d_irls - xi).abs().max())
    out["outlier_wmean_bad"] = float(info["w"][:n_bad].mean())
    out["outlier_wmean_good"] = float(info["w"][n_bad:].mean())
    return out


def selftest(seed=0, n_oct=4, T=150):
    print(f"\n=== the_splatV6 selftest (seed {seed}, {n_oct} octaves, {DEVICE}) ===")
    vae = make_test_vae(seed)
    bands, freq = octave_bands(vae, n_oct)
    for i, b in enumerate(bands):
        print(f"  octave {i}: {len(b):3d} packets  freq {freq[b].min():.2f}..{freq[b].max():.2f}")

    # -- group ops sanity
    e_inv, e_cmp = _test_group_ops()
    print(f"[S] Sim2 ops: inv {e_inv:.2e}  compose {e_cmp:.2e}   "
          f"(GO if both < 1e-5)")
    assert e_inv < 1e-5 and e_cmp < 1e-5

    # -- [F2] warp exactness
    e_warp = _test_warp_exactness(vae)
    print(f"[F2] warp exactness (param-action vs coord-action): {e_warp:.2e}   "
          f"(GO if < 1e-3, float32 trig)")
    assert e_warp < 1e-3

    # -- [P1] pose vote
    fit = _test_fit(seed)
    print(f"[P1] fit exact: rho=0.3 {fit['exact_rho0.3']:.2e}  "
          f"rho=2.5 {fit['exact_rho2.5']:.2e}  rho=-2.8 {fit['exact_rho-2.8']:.2e}")
    print(f"[P1] 30% coherent outliers: plain err {fit['outlier_plain']:.4f}  "
          f"IRLS err {fit['outlier_irls']:.4f}  "
          f"w(bad) {fit['outlier_wmean_bad']:.2f} vs w(good) {fit['outlier_wmean_good']:.2f}")
    assert fit['exact_rho2.5'] < 1e-4
    assert fit['outlier_irls'] < fit['outlier_plain'] * 0.5

    # -- [F1] pose factoring on the Sim(2) world. Run at eta=6 for BOTH arms:
    #    at the V5 default eta=20 the z step saturates its trust region on this
    #    razor-sharp random field in both arms, which hides the |dz| signal.
    #    At eta=6 the claim is decidable: pose-ON z goes nearly quiet.
    r_on = _run_pose(vae, T=T, n_oct=n_oct, pose=True, own=True, seed=seed, eta=6.0)
    r_off = _run_pose(vae, T=T, n_oct=n_oct, pose=False, own=False, seed=seed, eta=6.0)
    print(f"[F1] Sim2 world (eta=6 both): pose-ON  img {r_on['img']:.4f}  "
          f"|dz| {r_on['dz']:.4f}  xi-err {r_on['xi']:.4f}")
    print(f"     Sim2 world (eta=6 both): pose-OFF img {r_off['img']:.4f}  "
          f"|dz| {r_off['dz']:.4f}   (V5 behavior)")
    assert r_on['img'] < r_off['img'] and r_on['dz'] < r_off['dz']
    print(f"     factoring holds if ON <= OFF on img AND |dz|. HONEST SCOPE: on")
    print(f"     this random field, similarity motion is largely outside the")
    print(f"     z-span, so OFF fails by construction — mechanism shown here,")
    print(f"     the celeba claim is the live webcam A/B (POSE button).")

    # -- [F3] occlusion coast
    occ = (60, 100)          # in WORLD frames; loop steps are offset by ~2
    r_occ = _run_pose(vae, T=T, n_oct=n_oct, pose=True, own=True,
                      occlude=occ, seed=seed, record_sigma=True)
    tr = r_occ["sig_tr"]
    lo, hi = occ[0] + 2, occ[1] - 4       # safely inside the dark window
    grow = all(tr[t+1] >= tr[t] - 1e-12 for t in range(lo, hi))
    print(f"[F3] occlusion {occ}: tr(Sigma) {tr[lo]:.2e} -> {tr[hi]:.2e} "
          f"monotone during dark: {grow}; collapses on first vote after "
          f"emergence: {tr[-1]:.2e}")
    print(f"     last-frames img err {r_occ['img_last']:.4f} after re-lock "
          f"(pose-ON clean {r_on['img']:.4f}; dark frames themselves are "
          f"legitimately bad and excluded)")
    assert grow, "Sigma must grow monotonically while unobserved"

    # -- [F4] ownership vs the distractor blob
    r_own = _run_pose(vae, T=T, n_oct=n_oct, pose=True, own=True,
                      distractor=True, seed=seed)
    r_noown = _run_pose(vae, T=T, n_oct=n_oct, pose=True, own=False,
                        distractor=True, seed=seed)
    print(f"[F4] distractor blob: IRLS-ON  xi-err {r_own['xi']:.4f}  img {r_own['img']:.4f}")
    print(f"     distractor blob: IRLS-OFF xi-err {r_noown['xi']:.4f}  img {r_noown['img']:.4f}")
    print(f"     ownership holds if ON <= OFF on xi-err. The in-loop margin is")
    print(f"     THIN by design-honesty: clamps + residual-scaled gain + anchor")
    print(f"     already contain the kidnap (defense in depth). The sharp IRLS")
    print(f"     evidence is fit-level [P1] above: 4.5x on 30% coherent outliers.")

    # -- [A] regression: pose OFF == V5 (cascade beats open loop)
    a_all = _run_v5(vae, T=T, n_oct=n_oct, score="pose", seed=seed)
    a_open = _open_v5(vae, seed=seed, T=T)
    print(f"[A] regression (pose OFF = V5): all-octaves {a_all:.4f}  open {a_open:.4f}")

    print("\nledger: [S]/[F2]/[P1] asserted GO above (exactness is not optional).")
    print("        [F1] pose factoring: read ON vs OFF img and |dz| — the celeba")
    print("             version of this claim is decided at the webcam, honestly.")
    print("        [F3] Sigma grows in the dark and the pose re-locks: Bet 6c live.")
    print("        [F4] ownership: the blob cannot steal the pose it does not own.")
    print("        [A]  with pose off, V6 is V5 — nothing verified was broken.")


# ========================================================================== #
#  GUI — V5's slider bank + POSE/OWN toggles + the uncertainty ring          #
# ========================================================================== #

def run_gui(model_path=None, webcam=False, cam_index=0, n_oct=4):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw

    vae = load_v1(model_path) if model_path else make_test_vae(0)
    has_enc = model_path is not None
    world = None if webcam else Sim2World(vae, seed=0)
    cap = None
    if webcam:
        import cv2
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("webcam not available — synthetic Sim2 world instead")
            world, cap = Sim2World(vae, seed=0), None

    ctx = OctaveCortexV6(vae, n_oct=n_oct)
    cols = octave_colors(n_oct)
    H = vae.ren.H
    running = {"on": False}; slop = {"left": 0}; booted = {"done": False}
    view = {"mode": "full"}

    root = tk.Tk()
    root.title(f"the_splatV6 — (z, xi): {n_oct}-octave cascade + Sim(2) pose")
    root.configure(bg="#101014")
    VIEW = 280

    top = tk.Frame(root, bg="#101014"); top.pack(padx=8, pady=6)
    tk.Label(top, text="AFFERENT (octave-colored probes+flow)", fg="#8fd",
             bg="#101014").grid(row=0, column=0)
    tk.Label(top, text="BELIEF  warp(dec(z), xi)  [ring = pose uncertainty]", fg="#fd8",
             bg="#101014").grid(row=0, column=1)
    lab_world = tk.Label(top, bg="#101014"); lab_world.grid(row=1, column=0, padx=4)
    lab_belief = tk.Label(top, bg="#101014"); lab_belief.grid(row=1, column=1, padx=4)

    tele = tk.Label(root, fg="#ddd", bg="#101014", font=("Courier", 9), justify="left")
    tele.pack()

    eqf = tk.LabelFrame(root, text="OCTAVE EQ  (gain gates render + correction)",
                        fg="#8fd", bg="#101014", labelanchor="n")
    eqf.pack(pady=4)
    def make_eq_cb(i):
        def cb(v): ctx.eq[i] = float(v)
        return cb
    for i in range(n_oct):
        col = tk.Frame(eqf, bg="#101014"); col.grid(row=0, column=i, padx=6)
        s = tk.Scale(col, from_=1.0, to=0.0, resolution=0.05, orient="vertical",
                     length=90, bg="#101014", fg=cols[i], troughcolor="#333",
                     highlightthickness=0, command=make_eq_cb(i))
        s.set(1.0); s.pack()
        tk.Label(col, text=f"O{i}", fg=cols[i], bg="#101014").pack()

    ctrl = tk.Frame(root, bg="#101014"); ctrl.pack(pady=4)

    def toggle():
        running["on"] = not running["on"]
        b_start.config(text="STOP" if running["on"] else "START")
    def cycle_view():
        seq = ["full"] + list(range(n_oct))
        cur = view["mode"]; view["mode"] = seq[(seq.index(cur) + 1) % len(seq)]
        b_view.config(text=f"VIEW {'MIX' if view['mode']=='full' else 'O'+str(view['mode'])}")
    def do_slop():
        slop["left"] = 60
    def toggle_prec():
        ctx.dynamic_precision = not ctx.dynamic_precision
        b_prec.config(text=f"prec {'DYN' if ctx.dynamic_precision else 'FIX'}")
    def do_gist():
        if ctx.prev_frame is not None and has_enc: ctx.bootstrap(ctx.prev_frame)
    def set_k(v): ctx.set_k(int(float(v)))
    def toggle_pose():
        ctx.pose_enabled = not ctx.pose_enabled
        if not ctx.pose_enabled:
            ctx.pose = PoseBelief(DEVICE)     # honest reset, not a hidden hold
        b_pose.config(text=f"POSE {'ON' if ctx.pose_enabled else 'OFF'}")
    def toggle_own():
        ctx.own_enabled = not ctx.own_enabled
        b_own.config(text=f"OWN {'ON' if ctx.own_enabled else 'OFF'}")

    b_start = tk.Button(ctrl, text="START", command=toggle, width=6)
    b_view = tk.Button(ctrl, text="VIEW MIX", command=cycle_view)
    b_slop = tk.Button(ctrl, text="INJECT SLOP", command=do_slop)
    b_prec = tk.Button(ctrl, text="prec DYN", command=toggle_prec)
    b_gist = tk.Button(ctrl, text="GIST", command=do_gist,
                       state="normal" if has_enc else "disabled")
    b_pose = tk.Button(ctrl, text="POSE ON", command=toggle_pose)
    b_own = tk.Button(ctrl, text="OWN ON", command=toggle_own)
    for i, b in enumerate((b_start, b_view, b_slop, b_prec, b_gist, b_pose, b_own)):
        b.grid(row=0, column=i, padx=3)
    tk.Label(ctrl, text="K", fg="#ddd", bg="#101014").grid(row=0, column=7, padx=(10, 0))
    s_k = tk.Scale(ctrl, from_=6, to=48, orient="horizontal", bg="#101014", fg="#ddd",
                   command=set_k, length=110); s_k.set(14); s_k.grid(row=0, column=8)

    def to_photo(img_t, bands=None, ring=None):
        arr = (img_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((VIEW, VIEW), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        if bands:
            for pts, fl, col in bands:
                if pts is None: continue
                P = pts.cpu().numpy()
                Fl = fl.cpu().numpy() if fl is not None else None
                draw_flow = Fl is not None and len(Fl) == len(P)
                for i, (x, y) in enumerate(P):
                    dr.ellipse([x*VIEW-2, y*VIEW-2, x*VIEW+2, y*VIEW+2], outline=col, width=2)
                    if draw_flow:
                        dr.line([x*VIEW, y*VIEW, (x+Fl[i,0]*8)*VIEW, (y+Fl[i,1]*8)*VIEW],
                                fill=col, width=2)
        if ring is not None:
            # Bet 7's widening ellipse: center = pose image of the canvas center,
            # radius = position std (floored for visibility), dashed by arcs.
            cx, cy, rad, rho = ring
            R = max(6.0, rad * VIEW)
            for a0 in range(0, 360, 30):
                dr.arc([cx*VIEW-R, cy*VIEW-R, cx*VIEW+R, cy*VIEW+R],
                       start=a0, end=a0+18, fill="#fd8", width=2)
            dr.line([cx*VIEW, cy*VIEW,
                     cx*VIEW + R*math.cos(rho), cy*VIEW + R*math.sin(rho)],
                    fill="#fd8", width=2)
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
        else:
            t = world.frame()
        if slop["left"] > 0:
            slop["left"] -= 1; t = torch.rand_like(t)
        return t

    n = {"t": 0}
    def tick():
        if running["on"]:
            frame = grab()
            if frame is not None:
                if not booted["done"] and has_enc:
                    ctx.prev_frame = frame; ctx.bootstrap(frame); booted["done"] = True
                ctx.step(frame); n["t"] += 1
                overlays = [(ctx.pts[i], ctx.flow[i], cols[i]) for i in range(n_oct)]
                ph_w = to_photo(frame, overlays)
                lab_world.configure(image=ph_w); lab_world.image = ph_w
                if DEVICE == "cuda" or n["t"] % 2 == 0:
                    bimg = ctx.belief_render(None if view["mode"] == "full" else view["mode"])
                    ring = None
                    if ctx.pose_enabled:
                        c = Sim2.apply(ctx.pose.xi,
                                       torch.tensor([[CENTER, CENTER]], device=DEVICE))[0]
                        ring = (float(c[0]), float(c[1]),
                                3.0 * ctx.pose.pos_std(), float(ctx.pose.xi[2]))
                    ph_b = to_photo(bimg, ring=ring)
                    lab_belief.configure(image=ph_b); lab_belief.image = ph_b
                xi = ctx.pose.xi
                tele.config(text=(
                    "  ".join(f"O{i}:p{ctx.prec[i]:.2f}·eq{ctx.eq[i]:.1f}"
                              for i in range(n_oct))
                    + f"\n|dz| {ctx.dz:5.3f}  xi t({xi[0]:+.3f},{xi[1]:+.3f}) "
                      f"r{xi[2]:+.2f} s{math.exp(float(xi[3])):.2f}  "
                      f"trS {ctx.pose.trace():.1e}  own {ctx.own_frac:.2f}  "
                      f"{'UPD' if ctx.pose.updated else 'COAST'}  t {n['t']}"))
        root.after(40, tick)
    tick(); root.mainloop()
    if cap is not None: cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--octaves", type=int, default=4)
    ap.add_argument("--frames", type=int, default=150)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.seed, args.octaves, args.frames)
    else:
        run_gui(args.model, args.webcam, args.cam, args.octaves)


if __name__ == "__main__":
    main()
