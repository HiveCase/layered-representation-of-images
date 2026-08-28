"""Stage 4 (stretch): per-layer albedo x shading split.

Uses Careaga & Aksoy, "Intrinsic Image Decomposition via Ordinal Shading" (TOG 2023).
    pip install git+https://github.com/compphoto/Intrinsic
Then this wrapper runs their pretrained model per completed layer.
"""
from __future__ import annotations
import numpy as np


class IntrinsicSplitter:
    def __init__(self, device: str = "cuda"):
        try:
            from chrislib.general import uninvert
            from intrinsic.pipeline import load_models, run_pipeline
        except ImportError as e:
            raise ImportError("install github.com/compphoto/Intrinsic first") from e
        self._run, self._uninvert = run_pipeline, uninvert
        self.models = load_models("paper_weights")

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb uint8 -> (albedo float [0,1], shading float)  with rgb ~= albedo*shading"""
        res = self._run(self.models, rgb.astype(np.float32) / 255.0, device="cuda")
        albedo = res["albedo"]
        shading = self._uninvert(res["inv_shading"])[..., None]
        return albedo, shading
