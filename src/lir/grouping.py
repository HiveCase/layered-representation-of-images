"""Stage 1b: map fine-grained classes into the project's coarse semantic groups."""
from __future__ import annotations
import numpy as np
from .types import Layer


def build_class_to_group(groups_cfg: dict) -> dict[str, str]:
    table = {}
    for group, classes in groups_cfg.items():
        for c in classes:
            table[c.lower()] = group
    return table


def group_segments(segments: list[dict], groups_cfg: dict) -> list[Layer]:
    table = build_class_to_group(groups_cfg)
    layers = [Layer(mask=s["mask"], class_name=s["class_name"],
                    group=table.get(s["class_name"].lower(), "background"))
              for s in segments]
    # Merge all background-group masks into a single background layer.
    bg = [l for l in layers if l.group == "background"]
    fg = [l for l in layers if l.group != "background"]
    h, w = layers[0].mask.shape if layers else (0, 0)
    bg_mask = np.zeros((h, w), bool)
    for l in bg:
        bg_mask |= l.mask
    # Background also owns every pixel no one claimed.
    claimed = bg_mask.copy()
    for l in fg:
        claimed |= l.mask
    bg_mask |= ~claimed
    fg.append(Layer(mask=bg_mask, class_name="background", group="background",
                    depth=np.inf))                     # background is always farthest
    return fg
