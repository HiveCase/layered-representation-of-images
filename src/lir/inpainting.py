"""Stage 3b: fill holes revealed by peeling.

Backends:
  sd2  — Stable Diffusion 2 inpainting (diffusers). Semantic, slower, better big holes.
  lama — LaMa (https://github.com/advimman/lama). Fast, great background texture.
         Wire it in by exposing a callable `lama(image_rgb, mask_bool) -> image_rgb`.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


class Inpainter:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        if cfg["backend"] == "sd2":
            from diffusers import AutoPipelineForInpainting
            import torch
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                cfg["model_id"],
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
        elif cfg["backend"] == "lama":
            raise NotImplementedError("clone advimman/lama and expose it here")
        else:
            raise NotImplementedError(cfg["backend"])

    def __call__(self, image: np.ndarray, hole: np.ndarray, context: str = "scene") -> np.ndarray:
        """image: HxWx3 uint8, hole: HxW bool (True = fill). Returns HxWx3 uint8."""
        if not hole.any():
            return image
        d = self.cfg.get("dilate_hole_px", 0)
        if d:
            hole = cv2.dilate(hole.astype(np.uint8), np.ones((d, d), np.uint8)).astype(bool)
        h, w = image.shape[:2]
        pil_img = Image.fromarray(image).resize((512, 512))
        pil_msk = Image.fromarray((hole * 255).astype(np.uint8)).resize((512, 512))
        prompt = self.cfg.get("prompt_template", "{context}").format(context=context)
        out = self.pipe(prompt=prompt, image=pil_img, mask_image=pil_msk,
                        num_inference_steps=self.cfg.get("steps", 30),
                        guidance_scale=self.cfg.get("guidance", 7.5)).images[0]
        out = np.array(out.resize((w, h)))
        result = image.copy()
        result[hole] = out[hole]               # only replace hole pixels
        return result
