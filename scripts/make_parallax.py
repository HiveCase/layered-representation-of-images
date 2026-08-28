"""Render the layer stack as a parallax video: each layer translates at a speed
proportional to its nearness — the 5-second proof that layers matter.

python scripts/make_parallax.py --layers out/demo --out out/demo/parallax.mp4
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v3 as iio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--amplitude", type=int, default=30, help="max px shift (nearest layer)")
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()

    d = Path(args.layers)
    meta = json.loads((d / "stack.json").read_text())
    layers = [np.array(Image.open(d / f"layer_{m['index']:02d}.png")) for m in meta]
    n = len(layers)
    h, w = layers[0].shape[:2]

    frames = []
    for t in range(args.frames):
        phase = np.sin(2 * np.pi * t / args.frames)
        canvas = np.zeros((h, w, 3), np.float32)
        acc_a = np.zeros((h, w, 1), np.float32)
        for i in reversed(range(n)):                       # far -> near
            shift = int(phase * args.amplitude * (1 - i / max(n - 1, 1)))
            rgba = np.roll(layers[i], shift, axis=1).astype(np.float32) / 255.0
            a = rgba[..., 3:4]
            canvas = rgba[..., :3] * a + canvas * (1 - a)
            acc_a = a + acc_a * (1 - a)
        frames.append((canvas * 255).astype(np.uint8))
    iio.imwrite(args.out, frames, fps=30)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
