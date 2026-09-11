"""
extract_frames.py -- turn a video of ONE face (you: turning your head through
angles, changing expression, mouth open/closed, a little lighting variation) into
a folder of frames for a single-identity SplatVAE. No new trainer needed -- this
feeds your EXISTING splat_generator.

  python extract_frames.py --video me.mp4 --out faces1 --stride 2 --size 178 --crop
  python splat_generator.py --dataset folder --data_dir faces1 --image_size 128 \
         --num_packets 512 --amp --beta 0.001
  python live_hold_sample.py --model runs/splat/model.pt --image_size 128 --num_packets 512

Record ~1-2 min, ONE person. Coverage of POSE + EXPRESSION is what lets the tight
single-identity manifold lock and track. Low --beta keeps reconstructions sharp.
"""
import argparse, os


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="faces1")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--size", type=int, default=178)
    ap.add_argument("--crop", action="store_true", help="center square crop before resize")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {a.video}")
    i = n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % a.stride == 0:
            if a.crop:
                h, w = fr.shape[:2]; s = min(h, w); y = (h - s) // 2; x = (w - s) // 2
                fr = fr[y:y + s, x:x + s]
            cv2.imwrite(os.path.join(a.out, f"f{n:05d}.jpg"), cv2.resize(fr, (a.size, a.size)))
            n += 1
        i += 1
    cap.release()
    print(f"wrote {n} frames to {a.out}/  -- now train splat_generator.py --dataset folder --data_dir {a.out}")


if __name__ == "__main__":
    main()
