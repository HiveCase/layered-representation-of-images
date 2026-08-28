"""Metrics for all four deliverables. Decided BEFORE the system was built (week 1).

Deliverable coverage:
  (a) semantic masks  -> mask_iou, greedy_instance_miou, partition_integrity
  (b) depth order     -> robust_layer_depth + layer_depths_from_gt
                         + pairwise_ordering_accuracy
  recomposition       -> psnr, ssim, lpips_dist

Depth convention (must match src/lir/depth.py and ordering.py):
  SMALLER = NEARER on BOTH sides. Predicted layer depths are normalized to
  [0, 1] (background = +inf). GT depth maps (NYUv2/DIODE) are metric metres,
  which are also smaller-is-nearer, so no sign flip is needed. If your GT map
  is stored as disparity (bigger = nearer), invert it before sampling.
"""
from __future__ import annotations
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Recomposition fidelity
# --------------------------------------------------------------------------- #
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, channel_axis=-1)


_lpips_net = None
def lpips_dist(a: np.ndarray, b: np.ndarray, device: str = "cuda") -> float:
    global _lpips_net
    import lpips, torch
    if _lpips_net is None:
        _lpips_net = lpips.LPIPS(net="alex").to(device)
    def t(x):  # uint8 HWC -> [-1,1] BCHW
        return torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
    with torch.no_grad():
        return float(_lpips_net(t(a), t(b)))


# --------------------------------------------------------------------------- #
# Deliverable (b): depth ordering
# --------------------------------------------------------------------------- #
def robust_layer_depth(mask: np.ndarray, depth: np.ndarray,
                       statistic: str = "median", erode_px: int = 7) -> float:
    """Sample a single robust depth value for one layer from a depth map.

    Mirrors src/lir/ordering.assign_depths EXACTLY so the GT statistic is
    computed the same way as the predicted one. Returns nan for an empty mask.
    """
    m = mask.astype(np.uint8)
    if erode_px > 0:
        eroded = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
        m = eroded if eroded.sum() > 50 else m   # don't erode thin objects away
    vals = depth[m.astype(bool)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals) if statistic == "median" else vals.mean())


def layer_depths_from_gt(masks: list[np.ndarray], gt_depth: np.ndarray,
                         statistic: str = "median", erode_px: int = 7) -> list[float]:
    """GT depth per layer, in the SAME order as the predicted layers.

    Pass the visible-region mask of each predicted layer (stack.layers[i].mask)
    so predicted and GT depths are indexed identically for the pairwise check.
    """
    return [robust_layer_depth(m, gt_depth, statistic, erode_px) for m in masks]


def pairwise_ordering_accuracy(pred_depths: list[float], gt_depths: list[float],
                               tol: float = 0.0) -> float:
    """Deliverable (b): over all layer pairs, does predicted near/far match GT?

    Pairs whose GT depths differ by <= tol are treated as ties and skipped
    (use tol > 0 for continuous GT where exact equality never occurs but two
    layers are effectively co-planar). Pairs with a nan on either side are
    skipped. Returns 1.0 if no orderable pair exists.
    """
    n, correct, total = len(pred_depths), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj, gi, gj = pred_depths[i], pred_depths[j], gt_depths[i], gt_depths[j]
            if np.isnan(gi) or np.isnan(gj) or np.isnan(pi) or np.isnan(pj):
                continue
            if abs(gi - gj) <= tol:
                continue
            total += 1
            correct += (pi < pj) == (gi < gj)
    return correct / total if total else 1.0


# --------------------------------------------------------------------------- #
# Deliverable (a): semantic masks
# --------------------------------------------------------------------------- #
def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    return float(inter / union) if union else 1.0


def greedy_instance_miou(pred_masks: list[np.ndarray],
                         gt_masks: list[np.ndarray]) -> tuple[float, int]:
    """Class-agnostic mean IoU: greedily match each GT mask to its best unused
    predicted mask, then average the matched IoUs over all GT masks.

    Returns (mean_iou, n_gt). Unmatched GT masks contribute IoU 0, so this
    penalises both missed objects and over-merging. For a stricter score,
    restrict candidate pred_masks to the same semantic group before calling.
    """
    if not gt_masks:
        return float("nan"), 0
    used: set[int] = set()
    ious: list[float] = []
    for g in gt_masks:
        best, best_i = 0.0, -1
        for i, p in enumerate(pred_masks):
            if i in used:
                continue
            v = mask_iou(p, g)
            if v > best:
                best, best_i = v, i
        if best_i >= 0:
            used.add(best_i)
        ious.append(best)
    return float(np.mean(ious)), len(ious)


def partition_integrity(masks: list[np.ndarray]) -> dict[str, float]:
    """GT-FREE sanity check on the visible layer masks -- needs no annotations.

    A valid re-composable layering should partition the image: every pixel
    claimed by exactly one visible layer. Reports the fraction of pixels
    claimed once (want ~1.0), claimed by >=2 layers (overlap, want ~0.0), and
    claimed by none (holes, want ~0.0). Cheap smoke test for the grouping /
    background-fill logic on every image.
    """
    if not masks:
        return {"covered_once": float("nan"), "overlap": float("nan"),
                "unclaimed": float("nan")}
    stack = np.stack([m.astype(bool) for m in masks], axis=0)
    count = stack.sum(axis=0)                     # HxW: how many layers own each pixel
    total = count.size
    return {"covered_once": float((count == 1).sum() / total),
            "overlap":      float((count >= 2).sum() / total),
            "unclaimed":    float((count == 0).sum() / total)}