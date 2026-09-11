"""
live_splat3d.py — Neural Inference Bridge for Splat3D
================================================================================
This script hooks your webcam directly into the learned prior (model.pt).
Unlike the optical flow residual method, this predicts depth instantaneously
from a single static frame by passing it through the trained encoder/decoder.

NOTE: The model was trained on synthetic geometric primitives (spheres, boxes).
When it looks at your face or room, it will hallucinate and try to map reality
onto the nearest geometric shapes it understands. This is the raw sim-to-real gap.

PerceptionLab / Antti Luode.
"""

import cv2
import torch
import numpy as np
import argparse
from splat3d import Splat3D

def render_pointcloud(rgb_img, inv_depth, angle_y=0.5):
    """A fast numpy-based 3D lift to see the shape it hallucinated."""
    H, W = inv_depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Normalize to [-1, 1]
    nx = (x - W / 2) / (W / 2)
    ny = (y - H / 2) / (H / 2)
    
    # Depth mapping (1 is near, 0 is far in inv_depth)
    # Convert back to a pseudo-Z for projection mapping
    Z = 3.0 - 2.0 * inv_depth 
    
    # 3D coordinates
    X = nx * Z
    Y = ny * Z
    
    # Rotate around Y axis for parallax
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    X_rot = X * cy + Z * sy
    Z_rot = -X * sy + Z * cy
    
    # Project back to 2D
    focal = W * 0.8
    px = (X_rot / Z_rot * focal + W / 2).astype(np.int32)
    py = (Y / Z_rot * focal + H / 2).astype(np.int32)
    
    cloud = np.zeros((H, W, 3), dtype=np.uint8)
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (Z_rot > 0.1)
    
    # Simple scatter (last written wins, no strict z-buffer to keep it fast)
    cloud[py[valid], px[valid]] = rgb_img[y[valid], x[valid]]
    
    # Dilate slightly to fill holes from surface stretching
    kernel = np.ones((2,2), np.uint8)
    cloud = cv2.dilate(cloud, kernel, iterations=1)
    return cloud

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", type=str, default="runs/splat3d/model.pt")
    
    # Defaults strictly matched to the 400-step training run
    ap.add_argument("--image_size", type=int, default=56)
    ap.add_argument("--num_packets", type=int, default=192)
    ap.add_argument("--latent", type=int, default=128)
    
    ap.add_argument("--cloud", action="store_true", help="Show the rotating 3D lift")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} onto {device}...")
    
    # Initialize the 3D splat model
    model = Splat3D(args.image_size, args.latent, args.num_packets).to(device)
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        print("Ensure you are pointing to the correct --model path.")
        return
        
    model.eval()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        return

    print("Press 'q' to quit.")
    
    # Oscillation parameter for the point cloud viewing angle
    t = 0.0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Center crop to square to preserve aspect ratio
            h, w, _ = frame.shape
            sz = min(h, w)
            y1, x1 = (h - sz) // 2, (w - sz) // 2
            cropped = frame[y1:y1+sz, x1:x1+sz]
            
            # Downsample to the exact dimensions the model learned (56x56)
            resized = cv2.resize(cropped, (args.image_size, args.image_size))
            
            # Prepare tensor: BGR -> RGB -> [0, 1] -> NCHW
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(rgb).float() / 255.0
            tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(device)

            # --- THE PERCEPTION PASS ---
            # Extract instantaneous RGB reconstruction and learned depth prior
            pred_rgb_t, pred_dep_t, _ = model(tensor_img)

            # Post-process RGB reconstruction
            pred_rgb = (pred_rgb_t[0].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
            
            # Post-process Depth Map
            pred_dep = pred_dep_t[0].cpu().numpy().clip(0, 1)
            dep_color = cv2.applyColorMap((pred_dep * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

            # Upscale everything for visibility on your monitor (nearest neighbor to keep pixel blocks crisp)
            scale = 6
            out_w = args.image_size * scale
            
            disp_orig = cv2.resize(cropped, (out_w, out_w))
            disp_recon = cv2.resize(pred_rgb, (out_w, out_w), interpolation=cv2.INTER_NEAREST)
            disp_depth = cv2.resize(dep_color, (out_w, out_w), interpolation=cv2.INTER_NEAREST)

            # Assemble the viewing panel
            panel = np.hstack([disp_orig, disp_recon, disp_depth])
            cv2.imshow("Live Splat3D Inference | [Real] | [Recon RGB] | [Depth Prior]", panel)

            if args.cloud:
                # Render the 3D hallucination sweeping side to side
                angle = np.sin(t) * 0.5
                cloud = render_pointcloud(rgb, pred_dep, angle_y=angle)
                disp_cloud = cv2.resize(cv2.cvtColor(cloud, cv2.COLOR_RGB2BGR), (out_w, out_w), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Hallucinated 3D Lift", disp_cloud)
                t += 0.05

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()