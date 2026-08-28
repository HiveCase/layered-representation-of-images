"""Stage 3a-amodal: estimate each object layer's COMPLETE (amodal) extent.

The problem: an object layer's visible mask stops at whatever occludes it. To
make the layer genuinely re-composable (slide it aside without holes), we must
recover the object's shape *and* appearance where a NEARER layer covers it.

Two recoverable cases, two backends:

  interior  (default, no extra weights)
      Recovers occlusion that is ENCLOSED by the object's own silhouette --
      e.g. a strap or held object crossing a torso. These pixels provably
      belong to the object (they sit inside its filled outline) and are
      currently hidden by a nearer layer, so we claim them and let the scene
      inpainter regenerate their appearance. Boundary occlusion (an object cut
      off at its edge, like legs behind a car) is NOT recovered here -- there is
      no geometric evidence for how far the object extends, so we deliberately
      leave it to `pix2gestalt`. Conservative by design: it never invents extent
      the image doesn't support.

  pix2gestalt  (full amodal; external install)
      Runs a diffusion amodal-completion model that hallucinates the whole
      object (shape + RGB) from the visible mask, covering boundary occlusion
      too. Install https://github.com/cvlab-columbia/pix2gestalt and expose a
      callable returning (amodal_rgb uint8 HxWx3, amodal_alpha float HxW).

Return contract for both:  __call__ -> (amodal_mask bool HxW, completed_rgb or None)
  * amodal_mask ⊇ visible_mask   (the object's full estimated extent)
  * completed_rgb is None for `interior` (peel's inpainter fills the new region);
    for `pix2gestalt` it is the model's RGB, used directly for recovered pixels.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_fill_holes


class AmodalCompleter:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg, self.device = cfg, device
        self.backend = cfg.get("backend", "interior")
        if self.backend == "pix2gestalt":
            self._model = self._load_pix2gestalt(cfg, device)
        elif self.backend != "interior":
            raise NotImplementedError(f"amodal backend {self.backend}")

    # ------------------------------------------------------------------ #
    def __call__(self, image: np.ndarray, visible: np.ndarray,
                 occluder: np.ndarray, class_name: str = "object"):
        """image HxWx3 uint8, visible/occluder HxW bool. See module docstring."""
        if self.backend == "interior":
            return self._interior(visible, occluder), None
        return self._pix2gestalt(image, visible, class_name)

    # ------------------------------------------------------------------ #
    def _interior(self, visible: np.ndarray, occluder: np.ndarray) -> np.ndarray:
        """amodal = visible + (holes enclosed by the silhouette that a nearer
        layer covers). Everything is a boolean set operation, no model needed."""
        envelope = binary_fill_holes(visible)          # visible with interior gaps filled
        interior_holes = envelope & ~visible           # pixels enclosed but not visible
        recovered = interior_holes & occluder          # ...and actually occluded by a nearer layer
        return visible | recovered

    # ------------------------------------------------------------------ #
    def _load_pix2gestalt(self, cfg: dict, device: str):
        try:
            # Expected to expose load()/run(); wire to the repo's actual API.
            from pix2gestalt import load_model  # type: ignore
        except ImportError as e:
            raise ImportError(
                "amodal.backend='pix2gestalt' requires the pix2gestalt repo: "
                "https://github.com/cvlab-columbia/pix2gestalt (see amodal.py)"
            ) from e
        return load_model(cfg.get("model_id", "pix2gestalt"), device=device)

    def _pix2gestalt(self, image: np.ndarray, visible: np.ndarray, class_name: str):
        rgb, alpha = self._model.run(image, visible)   # per the repo's API
        amodal_mask = (np.asarray(alpha) > 0.5) | visible
        return amodal_mask, np.asarray(rgb, dtype=np.uint8)