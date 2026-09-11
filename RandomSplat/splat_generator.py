"""
splat_generator.py — the generator: concept -> wave packets -> image (trainable)
================================================================================
THE BUILD. The path is now clear, so this is the real next step: a generative
model whose decoder is the differentiable Gabor splatter. It is a VAE.

  ENCODER (a small CNN)         image  ->  latent z   (the "concept")
  DECODER (an MLP)              z      ->  N wave-packet parameters
  RENDERER (the splatter)       packets ->  image
  LOSS                          reconstruct the image + KL(z || N(0,I))

Generation = decode a latent: sample z ~ N(0,I) (or interpolate two z's), and the
decoder emits a set of localized Gabor wave packets that the renderer splats into
an image. "Concept -> packets -> picture", trained end to end.

THE PHASE HEAD IS COMPLEX, AND THAT IS THE WHOLE POINT (settled in
neuro_gabor_splat.py): the decoder never outputs a raw angle. For each packet it
outputs a complex coefficient (a, b) per colour channel, and the packet is
    env * ( a*cos(2*pi*f*x') - b*sin(2*pi*f*x') ).
Amplitude = sqrt(a^2+b^2) and phase = atan2(b,a) are smooth functions of (a,b), so
there is no 0->2pi seam for gradient descent to fall off. This is janus_cabbage's
complex axes, used exactly where the measurement said they are needed: the
network's phase OUTPUT. (Orientation theta is a raw scalar; that is fine, because
it is not regressed against a target — it only ever enters through cos/sin.)

A POSITION-ANCHOR TRICK FOR CONVERGENCE: each packet owns a fixed grid anchor;
the decoder predicts a small OFFSET from it. So every packet starts responsible
for one region of the canvas — coverage is built in, not something the optimiser
has to discover. This is the cheap stand-in for splat "densification".

HONESTY — read before running:
  - I could NOT execute this in my environment (no GPU, and a CUDA torch build is
    not installable in the sandbox). What I verified here is the *renderer math and
    shapes*, in numpy, against the same equations as the (run, working)
    neuro_gabor_splat.py. The training loop, AMP, checkpointing, and data loading
    are written carefully but were NOT run by me. RUN `--smoke` FIRST (a 2-step CPU
    pass on synthetic data) to confirm it executes on your machine before you spend
    half a day on it.
  - This is a VAE. Expect GOOD RECONSTRUCTIONS and BLURRY-BUT-STRUCTURED SAMPLES.
    That is the honest, expected outcome of a VAE at this scale — not photorealism,
    not a diffusion model. The contribution is the *decoder*: images generated as a
    sparse set of wave packets, not a pixel tower.
  - The real open risk, as we said, is plain CONVERGENCE — learning to aim packets
    for a concept is hard. The anchor trick and the complex phase head are the two
    levers that make it tractable; whether the samples are any good is the bet.
  - "Energy/compute" claims are not made here. This is a representation experiment.

GROUNDING (established, used not claimed): VAE (Kingma & Welling 2014); sparse
Gabor coding = the V1 model (Olshausen & Field 1996); differentiable primitive
splatting (Gaussian/Gabor splatting, 2023+); the cos/sin circular representation
for angles in complex-valued networks.

PerceptionLab / Antti Luode, with Claude (Opus 4.8), from a dialogue with Gemini.
Helsinki, June 2026. Do not hype. Do not lie. Just show.
"""
from PIL import Image
import os, math, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import glob
from torch.utils.checkpoint import checkpoint
import torchvision as tv
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid

# K params per packet: dpx,dpy,ls,th,lf, then (a,b) x 3 channels = 11
K = 11


# ======================================================================
# the differentiable Gabor splat renderer (mirrors the verified numpy math)
# ======================================================================
class FlatFolder(Dataset):
    """All images directly inside data_dir — no class subfolders needed."""
    def __init__(self, data_dir, image_size):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.paths = sorted(p for e in exts for p in glob.glob(os.path.join(data_dir, e)))
        if not self.paths:
            raise RuntimeError(f"no images directly in {data_dir}")
        self.tf = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), 0

class GaborRenderer(nn.Module):
    def __init__(self, image_size=128, num_packets=512, chunk=64, use_checkpoint=True):
        super().__init__()
        self.H = self.W = image_size
        self.N = num_packets
        self.chunk = chunk
        self.use_checkpoint = use_checkpoint
        gy, gx = torch.meshgrid(torch.linspace(0, 1, image_size),
                                torch.linspace(0, 1, image_size), indexing="ij")
        self.register_buffer("GX", gx[None, None])     # (1,1,H,W)
        self.register_buffer("GY", gy[None, None])
        # fixed per-packet position anchors on a sqrt(N) grid (logit space)
        side = int(math.ceil(math.sqrt(num_packets)))
        ax = torch.linspace(0.08, 0.92, side)
        anch = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)[:num_packets]
        anch = torch.clamp(anch, 1e-3, 1 - 1e-3)
        self.register_buffer("anchor_logit", torch.log(anch / (1 - anch)))  # (N,2)

    def activate(self, raw):
        """raw: (B,N,K) -> activated packet tensors."""
        a_px = self.anchor_logit[:, 0][None]            # (1,N)
        a_py = self.anchor_logit[:, 1][None]
        px = torch.sigmoid(a_px + raw[..., 0])          # (B,N) position = anchor + small offset
        py = torch.sigmoid(a_py + raw[..., 1])
        sigma = 0.012 + 0.14 * torch.sigmoid(raw[..., 2])   # envelope size (in [0,1] coords)
        theta = raw[..., 3]                                  # orientation (raw; enters via cos/sin)
        freq = 1.0 + 15.0 * torch.sigmoid(raw[..., 4])       # cycles per unit
        coeff = torch.tanh(raw[..., 5:11]).reshape(*raw.shape[:2], 3, 2)  # (B,N,3,2) complex (a,b)/chan
        return px, py, sigma, theta, freq, coeff

    def _render_chunk(self, px, py, sigma, theta, freq, coeff):
        # all (B,n) except coeff (B,n,3,2); returns (B,3,H,W) for this packet chunk
        px_ = px[..., None, None]; py_ = py[..., None, None]; s_ = sigma[..., None, None]
        th = theta[..., None, None]; f_ = freq[..., None, None]
        dx = self.GX - px_; dy = self.GY - py_                       # (B,n,H,W)
        xr = dx * torch.cos(th) + dy * torch.sin(th)
        env = torch.exp(-(dx * dx + dy * dy) / (2 * s_ * s_))        # (B,n,H,W)
        cos = torch.cos(2 * math.pi * f_ * xr)
        sin = torch.sin(2 * math.pi * f_ * xr)
        chans = []
        for c in range(3):
            a = coeff[:, :, c, 0][..., None, None]
            b = coeff[:, :, c, 1][..., None, None]
            chans.append((env * (a * cos - b * sin)).sum(dim=1))     # (B,H,W)
        return torch.stack(chans, dim=1)                             # (B,3,H,W)

    def forward(self, raw):
        # render in fp32 for numerical safety even under autocast
        with torch.cuda.amp.autocast(enabled=False):
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
            return torch.sigmoid(out)   # map accumulated field to [0,1]


# ======================================================================
# the VAE: CNN encoder + MLP decoder -> packet params -> renderer
# ======================================================================
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
            nn.Linear(hidden, num_packets * K))
        # start near zero so initial packets are small offsets / low amplitude
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].weight.data *= 0.1

    def forward(self, z):
        return self.net(z).view(-1, self.N, K)


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


# ======================================================================
# data
# ======================================================================
class SynthData(Dataset):
    """random images, only for --smoke (no download, no gpu)."""
    def __init__(self, n=16, size=32): self.n, self.size = n, size
    def __len__(self): return self.n
    def __getitem__(self, i):
        g = torch.linspace(0, 1, self.size)
        x = torch.stack([g[None].expand(self.size, -1), g[:, None].expand(-1, self.size),
                         torch.rand(self.size, self.size)])
        return x, 0


def build_dataset(name, data_dir, image_size):
    if name == "smoke":
        return SynthData(32, image_size)
    if name == "celeba":
        tf = T.Compose([T.CenterCrop(178), T.Resize(image_size), T.ToTensor()])
        return tv.datasets.CelebA(data_dir, split="train", download=True, transform=tf)
    if name == "flowers":
        tf = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
        # combine all splits for more images (~8k); labels unused
        return ConcatDataset([tv.datasets.Flowers102(data_dir, split=s, download=True, transform=tf)
                              for s in ("train", "val", "test")])
    if name == "cifar10":
        tf = T.Compose([T.Resize(image_size), T.ToTensor()])
        return tv.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
    if name == "folder":
            return FlatFolder(data_dir, image_size)
    raise ValueError(name)


# ======================================================================
def kl(mu, lv):
    return -0.5 * torch.mean(torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "sample", "interp"])
    ap.add_argument("--dataset", default="flowers",
                    choices=["flowers", "celeba", "cifar10", "folder", "smoke"],
                    help="flowers/cifar10 download reliably; celeba is prettier but its "
                         "torchvision (gdrive) download is often flaky — then use --dataset "
                         "folder --data_dir <a folder of images>.")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out", default="./runs/splat")
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--num_packets", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight (final, after warmup)")
    ap.add_argument("--beta_warmup", type=int, default=10, help="epochs to ramp beta 0->beta")
    ap.add_argument("--chunk", type=int, default=64, help="packets per render chunk (lower = less VRAM)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="mixed precision (recommended on the 3060)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--smoke", action="store_true", help="2-step CPU run on synthetic data")
    args = ap.parse_args()

    if args.smoke:
        args.dataset, args.image_size, args.num_packets = "smoke", 32, 16
        args.batch, args.epochs, args.workers, args.amp = 4, 1, 0, False
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    print(f"device={dev}  dataset={args.dataset}  image_size={args.image_size}  "
          f"packets={args.num_packets}  batch={args.batch}  amp={args.amp}")

    model = SplatVAE(args.image_size, args.latent, args.num_packets, args.chunk).to(dev)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev)); print("resumed", args.resume)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    # ---- generation-only modes ----
    if args.mode in ("sample", "interp"):
        model.eval()
        if args.mode == "sample":
            z = torch.randn(64, args.latent, device=dev)
            save_image(model.generate(z), os.path.join(args.out, "samples.png"), nrow=8)
            print("wrote samples.png")
        else:
            z0, z1 = torch.randn(1, args.latent, device=dev), torch.randn(1, args.latent, device=dev)
            ts = torch.linspace(0, 1, 8, device=dev)[:, None]
            zs = (1 - ts) * z0 + ts * z1
            save_image(model.generate(zs), os.path.join(args.out, "interp.png"), nrow=8)
            print("wrote interp.png")
        return

    # ---- train ----
    ds = build_dataset(args.dataset, args.data_dir, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=(dev.type == "cuda"), drop_last=True)
    print(f"dataset size: {len(ds)} images, {len(dl)} batches/epoch")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    fixed = next(iter(dl))[0][:32].to(dev)         # for reconstruction grids
    z_fixed = torch.randn(64, args.latent, device=dev)  # for sample grids
    logf = open(os.path.join(args.out, "loss.csv"), "a")

    step = 0
    for ep in range(args.epochs):
        model.train()
        beta = args.beta * min(1.0, (ep + 1) / max(args.beta_warmup, 1))
        t0 = time.time(); run_rec = run_kl = 0.0
        for x, _ in dl:
            x = x.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                recon, mu, lv = model(x)
                rec = F.mse_loss(recon, x)
                kld = kl(mu, lv)
                loss = rec + beta * kld
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run_rec += rec.item(); run_kl += kld.item(); step += 1
            if args.smoke and step >= 2:
                print("smoke OK: forward+backward ran. rec=%.4f kl=%.4f" % (rec.item(), kld.item()))
                return
        nb = len(dl)
        psnr = 10 * math.log10(1.0 / max(run_rec / nb, 1e-9))
        print(f"epoch {ep+1:3d}/{args.epochs}  rec {run_rec/nb:.4f} (PSNR {psnr:4.1f})  "
              f"kl {run_kl/nb:7.1f}  beta {beta:.2f}  {time.time()-t0:5.1f}s")
        logf.write(f"{ep+1},{run_rec/nb:.6f},{run_kl/nb:.6f},{beta:.4f}\n"); logf.flush()

        # checkpoints + grids each epoch
        model.eval()
        with torch.no_grad():
            torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
            rec, _, _ = model(fixed)
            grid = torch.cat([fixed, rec.clamp(0, 1)], 0)
            save_image(grid, os.path.join(args.out, f"recon_{ep+1:03d}.png"), nrow=8)
            save_image(model.generate(z_fixed), os.path.join(args.out, f"sample_{ep+1:03d}.png"), nrow=8)
    print("done. recon_*.png = top half originals / bottom half reconstructions; "
          "sample_*.png = generated from random latents.")


if __name__ == "__main__":
    main()
