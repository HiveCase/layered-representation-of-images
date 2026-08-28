"""Stage 2a: monocular relative depth. Convention: SMALLER value = NEARER."""
from __future__ import annotations
import numpy as np
from PIL import Image


class DepthEstimator:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        if cfg["backend"] == "depth_anything_v2":
            from transformers import pipeline
            self.pipe = pipeline("depth-estimation", model=cfg["model_id"], device=device)
        elif cfg["backend"] == "marigold":
            import diffusers
            self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
                cfg["model_id"]).to(device)
        else:
            raise NotImplementedError(cfg["backend"])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(image)
        if self.cfg["backend"] == "depth_anything_v2":
            pred = np.array(self.pipe(pil)["depth"], dtype=np.float32)
            pred = pred.max() - pred          # DA-V2 outputs disparity-like (big = near)
        else:                                  # marigold: already depth (big = far)
            pred = self.pipe(pil).prediction[0].squeeze().astype(np.float32)
        pred = (pred - pred.min()) / (np.ptp(pred) + 1e-8)   # normalize to [0,1]
        return pred                            # 0 = nearest, 1 = farthest
