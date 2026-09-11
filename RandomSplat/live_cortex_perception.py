"""
live_cortex.py — Real-time Predictive Coding & Phase-Locking
================================================================================
This script loads your trained VAE and hooks it up to your webcam.

THE ARCHITECTURE:
1. Webcam feeds a 128x128 image.
2. The trained Encoder generates a blurry top-down prediction (The Prior).
3. We detach the wave-packets and run 5 steps of gradient descent against the 
   live webcam frame (The Residual Error).
4. Reality forces the packets to phase-lock, turning the blurry prior into sharp perception.

Requires: cv2 (pip install opencv-python)
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse

# Import your exact architecture from the generator script
from splat_generator import SplatVAE

def tensor_to_cv2(tensor):
    """Converts a [1, 3, H, W] tensor to a CV2 BGR image."""
    img = tensor[0].detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C,H,W to H,W,C
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="runs/splat/model.pt")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_packets", type=int, default=256) # Adjust to 512 if your .pt is 512
    parser.add_argument("--steps", type=int, default=5, help="Predictive coding iterations per frame")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for phase-locking")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading artificial cortex on {device}...")

    # 1. Load the frozen trained model
    model = SplatVAE(image_size=args.image_size, latent=128, num_packets=args.num_packets, chunk=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Freeze VAE weights

    # 2. Boot the retina (Webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Cortex online. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop to square and resize to match model
        h, w, _ = frame.shape
        sz = min(h, w)
        cropped = frame[(h-sz)//2:(h+sz)//2, (w-sz)//2:(w+sz)//2]
        resized = cv2.resize(cropped, (args.image_size, args.image_size))
        
        # Convert BGR to RGB and format for PyTorch
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        # ---------------------------------------------------------
        # THE PERCEPTUAL LOOP
        # ---------------------------------------------------------
        
        # A. Top-Down Prediction (The Blurry Gist)
        with torch.no_grad():
            mu, _ = model.enc(img_tensor)
            raw_packets_prior = model.dec(mu)
            recon_prior = model.ren(raw_packets_prior)

        # B. Bottom-Up Residual Correction (Phase-Locking)
        # We take the prior's packets and let reality pull them into focus.
        raw_packets_sharp = raw_packets_prior.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([raw_packets_sharp], lr=args.lr)

        for _ in range(args.steps):
            optimizer.zero_grad()
            recon_sharp = model.ren(raw_packets_sharp)
            # Reality provides the error signal
            loss = F.mse_loss(recon_sharp, img_tensor) 
            loss.backward()
            optimizer.step()

        # ---------------------------------------------------------
        # VISUALIZATION
        # ---------------------------------------------------------
        
        # Convert all to CV2 images
        cv_real = tensor_to_cv2(img_tensor)
        cv_prior = tensor_to_cv2(recon_prior)
        cv_sharp = tensor_to_cv2(model.ren(raw_packets_sharp))

        # Add labels
        cv2.putText(cv_real, "1. Retina (Real)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(cv_prior, "2. Prior (Blurry)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(cv_sharp, f"3. Phase-Locked ({args.steps} steps)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Stack horizontally
        combined = np.hstack((cv_real, cv_prior, cv_sharp))
        
        # Scale up for easier viewing
        combined = cv2.resize(combined, (args.image_size * 6, args.image_size * 2), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Artificial Cortex", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()