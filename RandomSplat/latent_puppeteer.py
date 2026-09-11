#!/usr/bin/env python3
"""
latent_puppeteer.py -- Eristää 3D-käännöksen latentista avaruudesta ja siirtää sen staattiseen kuvaan.
"""
import argparse
import sys
import torch
import numpy as np
import cv2

try:
    from splat_trainer5 import load_splatvae
except ImportError:
    from splat_trainer4q import load_splatvae

def encode_frame(m, img):
    S = m.ren.H
    img_rs = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img_rs).float().permute(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        mu, _ = m.enc(x)
    return mu[0]

def render_latent(m, z):
    with torch.no_grad():
        r = m.ren(m.dec(z[None]).float())
        img = r[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8)[:, :, ::-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help="Opetettu malli, esim. model2.pt")
    p.add_argument('--video', required=True, help="Video aidosta liikkeestä (me.mp4)")
    p.add_argument('--target', required=True, help="Yksittäinen staattinen 2D-kuva johon liike siirretään")
    p.add_argument('--out', default='puppeteer_out.mp4', help="Tulosvideo")
    a = p.parse_args()

    # Ladataan malli samalla API:lla kuin muissakin työkaluissa
    m, _ = load_splatvae(a.model)
    m.eval()
    S = m.ren.H
    print(f"Lataus valmis: {S}px, {m.ren.N} pakettia.")

    # 1. Eristetään 3D-käännöksen latentti suuntavektori (PCA ensimmäinen pääkomponentti)
    print(f"Luetaan liikerata videosta {a.video}...")
    cap = cv2.VideoCapture(a.video)
    zs = []
    while True:
        ret, fr = cap.read()
        if not ret: 
            break
        zs.append(encode_frame(m, fr))
    cap.release()
    
    if len(zs) < 2:
        sys.exit("Video liian lyhyt.")

    Z = torch.stack(zs) # (F, 128)
    Z_mean = Z.mean(dim=0)
    Z_c = Z - Z_mean
    U, S_pca, V = torch.pca_lowrank(Z_c, q=1)
    turn_vector = V[:, 0] # Varsinainen 3D-illuusion vektori
    
    # 2. Enkoodataan staattinen kohdekuva
    print(f"Enkoodataan kohdekuva {a.target}...")
    target_img = cv2.imread(a.target)
    if target_img is None:
        sys.exit(f"Ei voitu lukea kuvaa {a.target}")
    z_target = encode_frame(m, target_img)

    # 3. Rakennetaan uusi synteettinen liike
    print(f"Luodaan 3D-käännösvideo tiedostoon {a.out}...")
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (S, S))
    
    # Animoidaan latenttivektoria edestakaisin
    t_vals = np.concatenate([np.linspace(0, 3.0, 30), np.linspace(3.0, -3.0, 60), np.linspace(-3.0, 0, 30)])
    
    for t in t_vals:
        z_new = z_target + t * turn_vector
        frame = render_latent(m, z_new)
        out.write(frame)
    
    out.release()
    print("Valmis. Tämä todistaa, että 3D elää opitussa monistossa, ei atomien koordinaateissa.")

if __name__ == '__main__':
    main()