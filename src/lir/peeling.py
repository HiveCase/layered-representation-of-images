"""Stage 3a: peel layers front-to-back so every layer becomes COMPLETE.

For layer i (near->far order), the region occluded by layers 0..i-1 must be
reconstructed so the layer can be re-composed on its own:

  background layer  -> disocclusion: fill every hole left behind foreground,
                       alpha opaque everywhere (it owns the whole frame).
  object layers     -> amodal completion: recover the object's hidden extent
                       via `amodal` (see amodal.py), inpaint the recovered
                       pixels, and KEEP them in the alpha. `l.amodal_mask`
                       records the full extent so matting refines against it
                       instead of collapsing back to the visible silhouette.

Without an amodal completer the object path degrades gracefully to the old
behaviour (alpha = visible mask), so this stays runnable if `amodal` is None.
"""
from __future__ import annotations
import numpy as np
from .types import Layer
from .inpainting import Inpainter


def peel(image: np.ndarray, layers: list[Layer], inpainter: Inpainter,
         amodal=None) -> None:
    """Fill each layer's .rgba with a completed RGBA image and set
    .amodal_mask. Mutates layers in place; expects near->far order.

    amodal: optional AmodalCompleter. When None, object layers keep their
            visible silhouette (legacy behaviour)."""
    h, w = image.shape[:2]
    occluder = np.zeros((h, w), bool)          # union of masks nearer than current

    for l in layers:
        if l.group == "background":
            # Disocclusion: the background owns every pixel hidden by foreground.
            hole = occluder & ~l.mask
            content = image.copy()
            if hole.any():
                content = inpainter(content, hole, context="background")
            l.amodal_mask = np.ones((h, w), bool)          # opaque everywhere
            rgba = np.dstack([content, np.full((h, w), 255, np.uint8)])
            l.rgba = rgba
            occluder |= l.mask
            continue

        # ---- object layer: recover the hidden (occluded) part of the object ----
        if amodal is not None:
            amodal_mask, completed_rgb = amodal(image, l.mask, occluder, l.class_name)
        else:
            amodal_mask, completed_rgb = l.mask, None

        recovered = amodal_mask & ~l.mask       # newly claimed, currently-occluded pixels
        content = image.copy()
        if recovered.any():
            if completed_rgb is not None:       # pix2gestalt supplied object RGB
                content[recovered] = completed_rgb[recovered]
            else:                               # interior backend: inpaint the object into it
                content = inpainter(content, recovered, context=l.class_name)

        l.amodal_mask = amodal_mask
        rgba = np.dstack([content, (amodal_mask * 255).astype(np.uint8)])
        l.rgba = rgba
        occluder |= l.mask                      # occlusion is by the VISIBLE silhouette