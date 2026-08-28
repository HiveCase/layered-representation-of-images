"""Stage 2b: assign each layer a robust depth statistic and sort near -> far."""
from __future__ import annotations
import cv2
import numpy as np
from .types import Layer


def assign_depths(layers: list[Layer], depth: np.ndarray, statistic: str = "median",
                  erode_px: int = 7) -> None:
    kernel = np.ones((erode_px, erode_px), np.uint8) if erode_px > 0 else None
    for l in layers:
        if np.isinf(l.depth):                  # background stays farthest
            continue
        m = l.mask.astype(np.uint8)
        if kernel is not None:
            eroded = cv2.erode(m, kernel)
            m = eroded if eroded.sum() > 50 else m   # don't erode thin objects away
        vals = depth[m.astype(bool)]
        l.depth = float(np.median(vals) if statistic == "median" else vals.mean())


def sort_layers(layers: list[Layer]) -> list[Layer]:
    # ties broken by area (bigger tends to be nearer context) then vertical position
    def key(l: Layer):
        ys = np.where(l.mask.any(axis=1))[0]
        y_bottom = ys.max() if len(ys) else 0
        return (l.depth, -l.area, -y_bottom)
    return sorted(layers, key=key)
