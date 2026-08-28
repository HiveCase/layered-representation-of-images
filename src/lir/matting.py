"""Stage 3c: refine binary masks into soft alpha mattes with ViTMatte."""
from __future__ import annotations
import cv2
import numpy as np
import torch
from PIL import Image


def make_trimap(mask: np.ndarray, erode_px: int, dilate_px: int) -> np.ndarray:
    """0 = background, 128 = unknown ring, 255 = definite foreground."""
    m = mask.astype(np.uint8)
    fg = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    un = cv2.dilate(m, np.ones((dilate_px, dilate_px), np.uint8))
    tri = np.zeros_like(m, dtype=np.uint8)
    tri[un.astype(bool)] = 128
    tri[fg.astype(bool)] = 255
    return tri


class Matter:
    def __init__(self, cfg: dict, device: str = "cuda"):
        from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        self.cfg, self.device = cfg, device
        self.processor = VitMatteImageProcessor.from_pretrained(cfg["model_id"])
        self.model = VitMatteForImageMatting.from_pretrained(cfg["model_id"]).to(device).eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Returns HxW float alpha in [0,1]."""
        tri = make_trimap(mask, self.cfg["trimap_erode_px"], self.cfg["trimap_dilate_px"])
        inputs = self.processor(images=Image.fromarray(image),
                                trimaps=Image.fromarray(tri),
                                return_tensors="pt").to(self.device)
        alpha = self.model(**inputs).alphas[0, 0].cpu().numpy()
        h, w = mask.shape
        return cv2.resize(alpha, (w, h))
