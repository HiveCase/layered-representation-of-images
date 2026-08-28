"""Decompose one image into an RGBA layer stack.

python scripts/run_single.py --image img.jpg --out out/demo \
       --config configs/default.yaml [--override ...] [--save-debug]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lir import LayeredPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", default=None)
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    pipe = LayeredPipeline(args.config, args.override)
    stack = pipe(image)
    stack.save(args.out)
    Image.fromarray(pipe.recomposition(stack)).save(Path(args.out) / "recomposed.png")
    if args.save_debug:
        for i, l in enumerate(stack.layers):
            dbg = image.copy(); dbg[~l.mask] //= 4
            Image.fromarray(dbg).save(Path(args.out) / f"debug_mask_{i:02d}_{l.group}.png")
    print(f"wrote {len(stack.layers)} layers to {args.out}")


if __name__ == "__main__":
    main()
