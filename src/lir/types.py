"""Core data structures shared by every pipeline stage."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np
from pathlib import Path
from PIL import Image


@dataclass
class Layer:
    mask: np.ndarray                 # HxW bool — visible-region mask
    class_name: str
    group: str                       # people | animals | vehicles | furniture | background
    depth: float = float("nan")      # robust depth statistic (relative; smaller = nearer)
    amodal_mask: np.ndarray | None = None  # HxW bool — full extent incl. recovered occluded region
    rgba: np.ndarray | None = None   # HxWx4 uint8 — COMPLETE layer after peeling/matting
    albedo: np.ndarray | None = None # stretch: HxWx3
    shading: np.ndarray | None = None

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class LayerStack:
    """Layers ordered near -> far. layers[0] occludes layers[1], etc."""
    image: np.ndarray                # HxWx3 uint8 original
    layers: list[Layer] = field(default_factory=list)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        meta = []
        for i, l in enumerate(self.layers):
            if l.rgba is not None:
                Image.fromarray(l.rgba).save(out / f"layer_{i:02d}.png")
            meta.append({"index": i, "class": l.class_name, "group": l.group,
                         "depth": None if np.isnan(l.depth) else float(l.depth),
                         "area": l.area})
        (out / "stack.json").write_text(json.dumps(meta, indent=2))