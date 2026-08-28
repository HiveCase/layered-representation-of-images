"""Orchestrator: config in, LayerStack out. Each stage is lazily constructed."""
from __future__ import annotations
import numpy as np
import yaml
from .types import Layer, LayerStack
from .segmentation import Segmenter
from .grouping import group_segments
from .depth import DepthEstimator
from .ordering import assign_depths, sort_layers
from .peeling import peel
from .inpainting import Inpainter
from .matting import Matter
from .amodal import AmodalCompleter
from .compositing import recompose


class LayeredPipeline:
    def __init__(self, cfg_path: str, override_path: str | None = None):
        self.cfg = yaml.safe_load(open(cfg_path))
        if override_path:
            deep_update(self.cfg, yaml.safe_load(open(override_path)))
        d = self.cfg.get("device", "cuda")
        self.segmenter = Segmenter(self.cfg["segmentation"], d)
        self.depther = DepthEstimator(self.cfg["depth"], d)
        self.inpainter = Inpainter(self.cfg["inpainting"], d)
        self.matter = Matter(self.cfg["matting"], d) if self.cfg["matting"]["enabled"] else None
        am = self.cfg.get("amodal", {})
        self.amodal = AmodalCompleter(am, d) if am.get("enabled", True) else None

    def __call__(self, image: np.ndarray) -> LayerStack:
        segments = self.segmenter(image)                               # stage 1a
        layers = group_segments(segments, self.cfg["grouping"]["groups"])  # 1b
        depth = self.depther(image)                                    # stage 2a
        oc = self.cfg["ordering"]
        assign_depths(layers, depth, oc["statistic"], oc["erode_px"])  # stage 2b
        layers = sort_layers(layers)
        layers = cap_layers(layers, self.cfg["output"]["max_layers"])
        peel(image, layers, self.inpainter, self.amodal)              # stage 3a-b
        if self.matter:                                                # stage 3c
            for l in layers:
                if l.group != "background":
                    # refine against the amodal extent so recovered (occluded)
                    # regions survive matting instead of collapsing to visible.
                    src_mask = l.amodal_mask if l.amodal_mask is not None else l.mask
                    alpha = self.matter(l.rgba[..., :3], src_mask)
                    l.rgba[..., 3] = (alpha * 255).astype(np.uint8)
        if self.cfg["intrinsics"]["enabled"]:                          # stage 4
            from .intrinsics import IntrinsicSplitter
            split = IntrinsicSplitter()
            for l in layers:
                l.albedo, l.shading = split(l.rgba[..., :3])
        return LayerStack(image=image, layers=layers)

    def recomposition(self, stack: LayerStack) -> np.ndarray:
        return recompose([l.rgba for l in stack.layers])


def cap_layers(layers: list[Layer], max_layers: int) -> list[Layer]:
    if len(layers) <= max_layers:
        return layers
    keep, merge = layers[: max_layers - 1], layers[max_layers - 1 : -1]
    bg = layers[-1]
    for l in merge:                          # fold small far layers into background
        bg.mask |= l.mask
    return keep + [bg]


def deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v