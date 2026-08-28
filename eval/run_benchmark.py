"""Run the pipeline over a folder of test images and report the metric table.

python eval/run_benchmark.py --config configs/default.yaml \
       [--override configs/ablations/depth_marigold.yaml] \
       --images data/test_images --out out/bench \
       [--depth-gt data/diode] [--seg-gt data/seg_masks]

Metric coverage
---------------
Always (no annotations needed):
  * recomposition fidelity : psnr, ssim, lpips
  * partition integrity    : covered_once / overlap / unclaimed on the visible
                             layer masks (a re-composable layering should tile
                             the image -- this is a cheap correctness smoke test)

When --depth-gt is given (deliverable b, depth order):
  * a GT depth map per image is loaded, each predicted layer's visible mask is
    sampled the SAME way ordering.assign_depths samples the prediction, and
    pairwise_ordering_accuracy scores near/far correctness over all layer pairs.

When --seg-gt is given (deliverable a, semantic masks):
  * per-image GT instance masks are loaded and scored with greedy_instance_miou.

GT file lookup: for image <stem>.jpg the harness looks for
  <depth-gt>/<stem>.(npy|png)     -- HxW float depth (smaller = nearer; see note)
  <seg-gt>/<stem>/*.png           -- one binary mask per GT instance
Missing GT for an image simply skips that metric for that image.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lir import LayeredPipeline
from metrics import (psnr, ssim, lpips_dist,
                     partition_integrity,
                     layer_depths_from_gt, pairwise_ordering_accuracy,
                     greedy_instance_miou)


# --------------------------------------------------------------------------- #
# Ground-truth loaders (best-effort; return None when GT is absent)
# --------------------------------------------------------------------------- #
def load_depth_gt(gt_dir: Path | None, stem: str, target_hw: tuple[int, int]):
    """Load a HxW float depth map for `stem`, resized to the image size.

    Accepts <stem>.npy (raw float) or <stem>.png (8/16-bit, read as-is).
    Convention required by the metrics: SMALLER = NEARER. If your dataset stores
    disparity (bigger = nearer) or inverse depth, invert it here before return.
    """
    if gt_dir is None:
        return None
    npy, png = gt_dir / f"{stem}.npy", gt_dir / f"{stem}.png"
    if npy.exists():
        d = np.load(npy).astype(np.float32)
    elif png.exists():
        d = np.array(Image.open(png)).astype(np.float32)
    else:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    h, w = target_hw
    if d.shape != (h, w):
        d = np.array(Image.fromarray(d).resize((w, h), Image.NEAREST), dtype=np.float32)
    return d


def load_seg_gt(gt_dir: Path | None, stem: str, target_hw: tuple[int, int]):
    """Load a list of binary GT instance masks from <seg-gt>/<stem>/*.png."""
    if gt_dir is None:
        return None
    d = gt_dir / stem
    if not d.is_dir():
        return None
    h, w = target_hw
    masks = []
    for p in sorted(d.glob("*.png")):
        m = np.array(Image.open(p).convert("L"))
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m).resize((w, h), Image.NEAREST))
        masks.append(m > 127)
    return masks or None


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r and r[key] is not None
            and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-gt", default=None,
                    help="dir with <stem>.npy/.png depth maps (deliverable b)")
    ap.add_argument("--seg-gt", default=None,
                    help="dir with <stem>/*.png GT instance masks (deliverable a)")
    args = ap.parse_args()

    depth_gt_dir = Path(args.depth_gt) if args.depth_gt else None
    seg_gt_dir = Path(args.seg_gt) if args.seg_gt else None

    # mirror the predicted-depth sampling settings from the pipeline config
    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.override:
        cfg_o = yaml.safe_load(open(args.override)) or {}
        cfg.get("ordering", {}).update(cfg_o.get("ordering", {}))
    oc = cfg.get("ordering", {})
    stat, erode = oc.get("statistic", "median"), oc.get("erode_px", 7)

    pipe = LayeredPipeline(args.config, args.override)
    rows = []
    for img_path in sorted(Path(args.images).glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        stack = pipe(image)
        recomposed = pipe.recomposition(stack)
        out_dir = Path(args.out) / img_path.stem
        stack.save(out_dir)
        Image.fromarray(recomposed).save(out_dir / "recomposed.png")

        masks = [l.mask for l in stack.layers]
        row = {"image": img_path.name, "n_layers": len(stack.layers),
               "psnr": psnr(image, recomposed),
               "ssim": ssim(image, recomposed),
               "lpips": lpips_dist(image, recomposed)}

        # GT-free structural check (deliverable a, cheap) ------------------- #
        row.update({f"part_{k}": v for k, v in partition_integrity(masks).items()})

        # Deliverable (b): depth ordering, if GT depth is available --------- #
        gt_depth = load_depth_gt(depth_gt_dir, img_path.stem, (h, w))
        if gt_depth is not None:
            pred_depths = [l.depth for l in stack.layers]
            gt_depths = layer_depths_from_gt(masks, gt_depth, stat, erode)
            row["order_acc"] = pairwise_ordering_accuracy(pred_depths, gt_depths)

        # Deliverable (a): semantic masks, if GT instance masks available --- #
        gt_masks = load_seg_gt(seg_gt_dir, img_path.stem, (h, w))
        if gt_masks is not None:
            fg = [l.mask for l in stack.layers if l.group != "background"]
            miou, n_gt = greedy_instance_miou(fg, gt_masks)
            row["instance_miou"], row["n_gt"] = miou, n_gt

        rows.append(row)
        print(row)

    metric_keys = ["psnr", "ssim", "lpips",
                   "part_covered_once", "part_overlap", "part_unclaimed",
                   "order_acc", "instance_miou"]
    summary = {k: _mean(rows, k) for k in metric_keys
               if any(k in r for r in rows)}
    summary["n_images"] = len(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "results.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2))
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()