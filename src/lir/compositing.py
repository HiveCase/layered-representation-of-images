"""Alpha compositing utilities — the pipeline's built-in integrity check."""
from __future__ import annotations
import numpy as np


def over(fg_rgba: np.ndarray, bg_rgba: np.ndarray) -> np.ndarray:
    """Porter–Duff 'over': C = Cf*af + Cb*ab*(1-af), a = af + ab*(1-af). float in [0,1]."""
    af = fg_rgba[..., 3:4]; ab = bg_rgba[..., 3:4]
    a = af + ab * (1 - af)
    rgb = fg_rgba[..., :3] * af + bg_rgba[..., :3] * ab * (1 - af)
    rgb = np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0)
    return np.concatenate([rgb, a], axis=-1)


def recompose(layers_rgba: list[np.ndarray]) -> np.ndarray:
    """Composite near->far ordered uint8 RGBA layers back into one RGB uint8 image."""
    acc = layers_rgba[-1].astype(np.float32) / 255.0        # start from farthest
    for rgba in reversed(layers_rgba[:-1]):
        acc = over(rgba.astype(np.float32) / 255.0, acc)
    return (acc[..., :3] * 255).clip(0, 255).astype(np.uint8)
