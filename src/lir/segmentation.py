"""Stage 1a: panoptic segmentation.

Default backend: Mask2Former (COCO panoptic) via Hugging Face.
Alternative:      SAM 2 masks + Grounding-DINO labels ("sam2_grounded") — install
                  https://github.com/facebookresearch/sam2 and IDEA-Research/GroundingDINO,
                  then implement `_sam2_grounded` mirroring `_mask2former`'s return type.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image


class Segmenter:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        if cfg["backend"] == "mask2former":
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            self.processor = AutoImageProcessor.from_pretrained(cfg["model_id"])
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                cfg["model_id"]).to(device).eval()
        else:
            raise NotImplementedError(f"backend {cfg['backend']} — see module docstring")

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> list[dict]:
        """Returns [{'mask': HxW bool, 'class_name': str, 'score': float}, ...]"""
        pil = Image.fromarray(image)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[pil.size[::-1]],
            threshold=self.cfg.get("score_threshold", 0.75))[0]
        seg = result["segmentation"].cpu().numpy()
        id2label = self.model.config.id2label
        out = []
        for info in result["segments_info"]:
            mask = seg == info["id"]
            if mask.sum() < self.cfg.get("min_mask_area", 0):
                continue
            out.append({"mask": mask,
                        "class_name": id2label[info["label_id"]],
                        "score": float(info.get("score", 1.0))})
        return out
