"""
the_splatV2_live.py  —  the LIVE cortex on the real the_splat manifold.

PerceptionLab / Antti Luode (Helsinki), with Claude (Fable 5). July 2026.

    Do not hype. Do not lie. Just show.

WHAT THIS IS
------------
the_splatV2 proved the loop closes through real pixels — but its state was 1-D
(a phase) and its field a stand-in. This file is the marked next attach point,
built: the held state is the FULL 128-D latent z of your trained the_splat
SplatVAE (celeba faces), and the afferent is SPARSE FOR REAL — K probe points
sampled from the frame, not the frame. The renderer is your exact v1 Gabor
splatter (classes embedded verbatim; your runs/splat/model.pt loads strict).

THE WIRE (multi-D now):

    z_pred  = z + beta*(z - z_prev)                # prior flow: velocity from history
    probes  = frame[p_1..p_K]                      # sparse afferent: K numbers, not HxW
    belief  = splat.render_probes(dec(z_pred), p)  # render ONLY at the K probes (cheap)
    r_t     = probes - belief                      # sparse residual
    prec    = sig_ref^2/(sig_ref^2+rough(r)^2)     # reliability from residual delay-history
    z       = z_pred - eta*prec*grad_z |r|^2       # precision-gated correction through
                                                   #   the decoder (gradient inversion)
    (saccade)  p <- argmax |r| neighborhoods       # efferent = active sensing: action
                                                   #   moves the MEASUREMENT, not the world

The cortex never sees the frame. It sees K numbers and their history, and from
that holds a 128-D latent whose render tracks the world. The Takens claim, in
its honest multi-D form: K generic sparse measurements suffice once K exceeds
the INTRINSIC dimension of the motion — and that knee is measured, not asserted
(--selftest claim [C]).

WHERE THE MATH STANDS (be precise about which theorem does what):
  - Takens/SYC gives OBSERVABILITY: generic sparse observables + delays
    determine the state of a low-D motion. V2's mirror-lock finding (symmetric
    observables are non-generic) carries over: probes are random, not symmetric.
  - The CORRECTION here is gradient inversion through the smooth generative map
    at the probe coordinates. It is sound when the measurement operator
    restricted to the manifold tangent space is injective — generically true
    for K random probes when K > intrinsic motion dim. That is claim [C].
  - The delay history is used for (a) precision (residual roughness) and
    (b) the prior velocity — the minimal honest uses. Full delay-coordinate
    state reconstruction is NOT claimed here; the decoder does that work.

WHAT WAN-STREAMER (2026) CONFIRMS AND WHERE THIS DIFFERS:
  their streaming contract — commit your own generated latents back into the
  causal history as context for the next unit — IS this loop's efferent copy;
  their idle "does not collapse into a frozen portrait" IS coasting on the
  prior when the afferent is silent. They hold that state in a transformer
  KV-cache grown every 160 ms; the bet here is that a fixed rendering manifold
  + a low-D held latent + K probes is the minimum that does the same job.

RUN
    python the_splatV2_live.py --selftest                 # headless scorecard
    python the_splatV2_live.py                            # GUI, synthetic world
    python the_splatV2_live.py --model runs/splat/model.pt          # your celeba field
    python the_splatV2_live.py --model runs/splat/model.pt --webcam # the real test
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
K_PARAMS = 11  # per-packet params, exactly as in v1 splat_generator.py


# ========================================================================== #
#  the_splat v1  —  EMBEDDED VERBATIM (splat_generator.py, ArtificialCortex) #
#  so that your trained runs/splat/model.pt loads with strict=True.          #
#  Do not "improve" these classes; compatibility is the point.               #
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

    # ---- V2-live addition (does NOT touch the state_dict): sparse render ----
    def render_probes(self, raw, pxy):
        """Evaluate the packet field ONLY at probe coords pxy (K,2) in [0,1]
        (x, y). Returns (B, K, 3). Identical math to _render_chunk with the
        pixel grid replaced by the K probe points — the sparse afferent's
        predicted values, at cost O(K*N) instead of O(H*W*N)."""
        px, py, sigma, theta, freq, coeff = self.activate(raw.float())
        qx = pxy[:, 0][None, None, :]                 # (1,1,K)
        qy = pxy[:, 1][None, None, :]
        px_ = px[..., None]; py_ = py[..., None]; s_ = sigma[..., None]
        th = theta[..., None]; f_ = freq[..., None]
        dx = qx - px_; dy = qy - py_                  # (B,N,K)
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None]
            b = coeff[:, :, c, 1][..., None]
            chans.append((env * (a * cos - b * sin)).sum(dim=1))   # (B,K)
        return torch.sigmoid(torch.stack(chans, dim=-1))           # (B,K,3)


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
    """Load a real the_splat checkpoint. Hyperparameters are INFERRED from the
    state_dict itself (no guessing flags), then loaded strict=True:
        image_size  <- ren.GX width
        num_packets <- ren.anchor_logit rows  (cross-checked vs decoder head)
        latent      <- enc.fc_mu rows
        hidden      <- dec.net.0 rows
    """
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict) or "ren.GX" not in sd:
        raise ValueError(f"{path} is not a the_splat v1 SplatVAE state_dict "
                         f"(keys: {list(sd)[:5] if isinstance(sd, dict) else type(sd)})")
    image_size = sd["ren.GX"].shape[-1]
    num_packets = sd["ren.anchor_logit"].shape[0]
    latent = sd["enc.fc_mu.weight"].shape[0]
    hidden = sd["dec.net.0.weight"].shape[0]
    head_n = sd["dec.net.4.weight"].shape[0] // K_PARAMS
    assert head_n == num_packets, f"decoder head {head_n} != renderer packets {num_packets}"
    model = SplatVAE(image_size, latent, num_packets).to(device)
    model.dec.net[0] = nn.Linear(latent, hidden)   # honor non-default hidden
    model.dec.net[2] = nn.Linear(hidden, hidden)
    model.dec.net[4] = nn.Linear(hidden, num_packets * K_PARAMS)
    model.dec.to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[load_v1] {path}: image_size={image_size} packets={num_packets} "
          f"latent={latent} hidden={hidden}  (strict load OK)")
    return model


# ========================================================================== #
#  THE CORTEX  —  a held multi-D latent, corrected sparsely                  #
# ========================================================================== #

class LatentCortex:
    """Holds z. Prior flow = velocity extrapolation from its own history
    (the minimal delay-coordinate use) + a weak leak to a slow prior mean.
    Correction = precision-gated gradient inversion of the K-probe residual
    through the decoder. Precision = residual roughness vs sig_ref, exactly
    the machinery verified in alavirta/the_splatV2."""

    def __init__(self, vae, n_probes=24, m_hist=8, sig_ref=0.06,
                 eta=6.0, beta_mom=0.55, leak=0.002, seed=0):
        self.vae = vae
        self.rng = np.random.default_rng(seed)
        self.z = torch.zeros(1, vae.latent, device=DEVICE)
        self.z_prev = self.z.clone()
        self.z_prior = self.z.clone()          # slow EMA the leak pulls toward
        self.n_probes = n_probes
        self.probes = self._random_probes(n_probes)
        self.m_hist = m_hist
        self.res_hist = deque(maxlen=m_hist)   # scalar residual magnitude history
        self.sig_ref = sig_ref
        self.eta = eta
        self.beta_mom = beta_mom
        self.leak = leak
        self.dynamic_precision = True
        self.saccades = False
        # telemetry
        self.precision = 1.0
        self.rough = 0.0
        self.resid = 0.0
        self.dz = 0.0

    def _random_probes(self, k):
        p = self.rng.uniform(0.12, 0.88, size=(k, 2))
        return torch.tensor(p, dtype=torch.float32, device=DEVICE)

    def set_probes(self, k):
        self.n_probes = k
        self.probes = self._random_probes(k)
        self.res_hist.clear()

    @staticmethod
    def sample_frame(frame, probes):
        """frame (3,H,W) in [0,1]; probes (K,2) xy in [0,1] -> (K,3).
        Bilinear, so probe values are smooth in probe position."""
        g = probes[None, None, :, :] * 2 - 1                     # (1,1,K,2) in [-1,1]
        v = F.grid_sample(frame[None], g, align_corners=True)    # (1,3,1,K)
        return v[0, :, 0, :].T                                   # (K,3)

    def _precision_from(self, r_mag):
        self.res_hist.append(r_mag)
        if len(self.res_hist) < 4:
            self.rough = 0.0
            return 1.0
        h = np.array(self.res_hist)
        d2 = np.diff(h, n=2)
        self.rough = float(np.sqrt(np.mean(d2 * d2)))
        return float(self.sig_ref**2 / (self.sig_ref**2 + self.rough**2))

    def step(self, frame):
        """One tick. frame: (3,H,W) tensor in [0,1] on DEVICE. Returns nothing;
        read .z / telemetry."""
        # ---- prior flow: velocity from own history + weak leak ----
        vel = self.z - self.z_prev
        z_pred = self.z + self.beta_mom * vel
        z_pred = z_pred - self.leak * (z_pred - self.z_prior)

        # ---- sparse afferent + residual ----
        target = self.sample_frame(frame, self.probes)            # (K,3)
        z_var = z_pred.detach().clone().requires_grad_(True)
        pred = self.vae.ren.render_probes(self.vae.dec(z_var), self.probes)[0]  # (K,3)
        loss = F.mse_loss(pred, target)
        loss.backward()
        r_mag = float(torch.sqrt(loss.detach()))
        self.resid = r_mag

        # ---- precision from residual roughness (delay history) ----
        prec = self._precision_from(r_mag) if self.dynamic_precision else 1.0
        self.precision = prec

        # ---- precision-gated correction (gradient inversion) ----
        with torch.no_grad():
            step = self.eta * prec * z_var.grad
            z_new = z_pred - step
            self.dz = float(step.norm())
        self.z_prev = self.z
        self.z = z_new.detach()
        self.z_prior = 0.995 * self.z_prior + 0.005 * self.z

        # ---- efferent: saccade probes toward high residual (active sensing) ----
        if self.saccades and prec > 0.5:
            with torch.no_grad():
                per = (pred.detach() - target).pow(2).mean(-1)    # (K,)
                worst = torch.topk(per, max(1, self.n_probes // 6)).indices
                jump = torch.tensor(self.rng.uniform(0.12, 0.88, (len(worst), 2)),
                                    dtype=torch.float32, device=DEVICE)
                self.probes[worst] = jump

    @torch.no_grad()
    def belief_render(self):
        return self.vae.generate(self.z)[0]                       # (3,H,W)


# ========================================================================== #
#  A SYNTHETIC WORLD  —  a hidden z on a low-D motion, rendered by the SAME  #
#  field. Ground truth exists, so the loop is falsifiable without a webcam.  #
# ========================================================================== #

class SplatWorld:
    """Hidden z_world moves on a d_motion-dimensional trajectory: a circle in
    a random 2-plane + slow OU on (d_motion-2) extra axes. The cortex only
    ever sees frames (or probes of them)."""

    def __init__(self, vae, d_motion=2, radius=1.6, omega=0.11, seed=0):
        self.vae = vae
        g = torch.Generator(device="cpu").manual_seed(seed)
        basis = torch.randn(d_motion, vae.latent, generator=g)
        basis, _ = torch.linalg.qr(basis.T)                       # orthonormal columns
        self.basis = basis[:, :d_motion].to(DEVICE)               # (latent, d)
        self.d = d_motion
        self.radius = radius
        self.omega = omega
        self.t = 0
        self.rng = np.random.default_rng(seed + 1)
        self.ou = np.zeros(max(0, d_motion - 2))
        self.slop_left = 0
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
        img = self.vae.generate(z)[0]                             # (3,H,W)
        if self.slop_left > 0:
            self.slop_left -= 1
            img = torch.rand_like(img)                            # sensor gone: pure noise
        return img, z


# ========================================================================== #
#  SELFTEST  —  the scorecard. Chance/anchor = open-loop drift error.        #
# ========================================================================== #

def make_test_vae(seed=0, image_size=64, packets=144, latent=64):
    torch.manual_seed(seed)
    vae = SplatVAE(image_size, latent, packets).to(DEVICE)
    # a random decoder inits near-zero (flat gray render, zero gradients);
    # boost it so the stand-in manifold is actually expressive. This does NOT
    # apply to a trained checkpoint — only the synthetic test field.
    with torch.no_grad():
        vae.dec.net[-1].weight *= 60.0
        vae.dec.net[-1].bias.normal_(0, 0.8)
    vae.eval()
    return vae


def run_tracking(vae, T=260, n_probes=24, closed=True, dynamic=True,
                 slop_at=None, d_motion=2, seed=0):
    world = SplatWorld(vae, d_motion=d_motion, seed=seed)
    ctx = LatentCortex(vae, n_probes=n_probes, seed=seed)
    ctx.dynamic_precision = dynamic
    errs, errs_clean_after = [], []
    for t in range(T):
        if slop_at and t == slop_at[0]:
            world.inject_slop(slop_at[1] - slop_at[0])
        frame, z_true = world.frame()
        if closed:
            ctx.step(frame)
        else:  # open loop: prior flow only, never corrected
            ctx.z_prev, ctx.z = ctx.z, ctx.z + ctx.beta_mom * (ctx.z - ctx.z_prev)
        with torch.no_grad():
            e = F.mse_loss(vae.generate(ctx.z), vae.generate(z_true)).item()
        errs.append(e)
        if slop_at and t >= slop_at[1] + 25:
            errs_clean_after.append(e)
    tail = errs[T // 3:]
    return float(np.mean(tail)), (float(np.mean(errs_clean_after)) if errs_clean_after else None)


def selftest(seed=0):
    print(f"\n=== the_splatV2_live selftest (seed {seed}, device {DEVICE}) ===")
    vae = make_test_vae(seed)
    with torch.no_grad():
        std = vae.generate(torch.randn(1, vae.latent, device=DEVICE))[0].std().item()
    print(f"stand-in field pixel std {std:.3f} (needs to be >0.05 to be a real manifold)")

    # [A] K sparse probes hold a 128->64-D latent against open-loop drift
    a_closed, _ = run_tracking(vae, closed=True, seed=seed)
    a_open, _ = run_tracking(vae, closed=False, seed=seed)
    print(f"[A] sparse-probe closed loop tracks   closed {a_closed:.4f}   open {a_open:.4f}")

    # [B] dynamic precision coasts through slop; fixed gain chases the noise
    b_dyn, b_dyn_after = run_tracking(vae, dynamic=True, slop_at=(120, 170), seed=seed)
    b_fix, b_fix_after = run_tracking(vae, dynamic=False, slop_at=(120, 170), seed=seed)
    print(f"[B] slop: dyn {b_dyn:.4f} (after {b_dyn_after:.4f})   "
          f"fixed {b_fix:.4f} (after {b_fix_after:.4f})")

    # [C] the knee: error vs probe count, motion intrinsic dim = 2 and 4
    print("[C] error vs K probes (the Takens-dimension claim, measured):")
    for d in (2, 4):
        row = []
        for k in (2, 4, 8, 16, 32):
            e, _ = run_tracking(vae, n_probes=k, d_motion=d, T=200, seed=seed)
            row.append(f"K={k}:{e:.4f}")
        print(f"    d_motion={d}:  " + "  ".join(row) + f"   (open {a_open:.4f})")

    print("\nledger: [A] closed << open = the sparse afferent is doing the holding.")
    print("        [B] dyn <= fixed through slop = precision buys the bad frames.")
    print("        [C] error should fall to the closed-loop floor once K exceeds")
    print("            the motion's intrinsic dim, and NOT before. If K=2 already")
    print("            tracks d=4 motion, or K=32 fails d=2, the claim is dead.")


# ========================================================================== #
#  THE LIVE GUI                                                              #
# ========================================================================== #

def run_gui(model_path=None, webcam=False, cam_index=0):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw

    vae = load_v1(model_path) if model_path else make_test_vae(0)
    world = None if webcam else SplatWorld(vae, d_motion=3, seed=0)
    cap = None
    if webcam:
        import cv2
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("webcam not available — falling back to synthetic world")
            world, cap = SplatWorld(vae, d_motion=3, seed=0), None

    ctx = LatentCortex(vae, n_probes=24)
    H = vae.ren.H
    running = {"on": False}
    slop = {"left": 0}

    root = tk.Tk()
    root.title("the_splatV2_live — a held latent, corrected by K probes")
    root.configure(bg="#101014")
    VIEW = 280

    top = tk.Frame(root, bg="#101014"); top.pack(padx=8, pady=8)
    lab_world = tk.Label(top, bg="#101014"); lab_world.grid(row=1, column=0, padx=4)
    lab_belief = tk.Label(top, bg="#101014"); lab_belief.grid(row=1, column=1, padx=4)
    tk.Label(top, text="AFFERENT (world + probes)", fg="#8fd", bg="#101014").grid(row=0, column=0)
    tk.Label(top, text="BELIEF  render(dec(z))", fg="#fd8", bg="#101014").grid(row=0, column=1)

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

    def toggle_sacc():
        ctx.saccades = not ctx.saccades
        b_sacc.config(text=f"saccades {'ON' if ctx.saccades else 'off'}")

    def set_k(v):
        ctx.set_probes(int(float(v)))

    b_start = tk.Button(ctrl, text="START", command=toggle, width=7)
    b_slop = tk.Button(ctrl, text="INJECT SLOP", command=do_slop)
    b_prec = tk.Button(ctrl, text="precision DYN", command=toggle_prec)
    b_sacc = tk.Button(ctrl, text="saccades off", command=toggle_sacc)
    for i, b in enumerate((b_start, b_slop, b_prec, b_sacc)):
        b.grid(row=0, column=i, padx=3)
    tk.Label(ctrl, text="K probes", fg="#ddd", bg="#101014").grid(row=0, column=4, padx=(12, 0))
    s_k = tk.Scale(ctrl, from_=2, to=96, orient="horizontal", bg="#101014", fg="#ddd",
                   command=set_k, length=160)
    s_k.set(24); s_k.grid(row=0, column=5)

    def to_photo(img_t, probes=None):
        arr = (img_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(arr).resize((VIEW, VIEW), Image.NEAREST)
        if probes is not None:
            d = ImageDraw.Draw(im)
            for x, y in probes.cpu().numpy():
                d.ellipse([x * VIEW - 3, y * VIEW - 3, x * VIEW + 3, y * VIEW + 3],
                          outline="#00ff88", width=2)
        return ImageTk.PhotoImage(im)

    def grab_frame():
        if cap is not None:
            import cv2
            ok, fr = cap.read()
            if not ok: return None
            h, w, _ = fr.shape
            s = min(h, w)
            fr = fr[(h - s)//2:(h + s)//2, (w - s)//2:(w + s)//2]
            fr = cv2.resize(fr, (H, H))
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(fr).float().permute(2, 0, 1).to(DEVICE) / 255.0
            if slop["left"] > 0:
                slop["left"] -= 1
                t = torch.rand_like(t)
            return t
        img, _ = world.frame()
        return img

    frame_i = {"n": 0}

    def tick():
        if running["on"]:
            frame = grab_frame()
            if frame is not None:
                ctx.step(frame)
                frame_i["n"] += 1
                ph_w = to_photo(frame, ctx.probes)
                lab_world.configure(image=ph_w); lab_world.image = ph_w
                # thinker-performer: the cheap probe loop runs every tick; the
                # expensive full belief render refreshes every 2nd tick on CPU
                if DEVICE == "cuda" or frame_i["n"] % 2 == 0:
                    ph_b = to_photo(ctx.belief_render())
                    lab_belief.configure(image=ph_b); lab_belief.image = ph_b
                tele.config(text=(
                    f"precision {ctx.precision:5.2f}   rough {ctx.rough:6.4f}   "
                    f"probe-resid {ctx.resid:6.4f}   |dz| {ctx.dz:6.3f}   "
                    f"K {ctx.n_probes}   t {frame_i['n']}"))
        root.after(40, tick)

    tick()
    root.mainloop()
    if cap is not None:
        cap.release()


# ========================================================================== #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="path to your the_splat model.pt")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.seed)
    else:
        run_gui(args.model, args.webcam, args.cam)


if __name__ == "__main__":
    main()
