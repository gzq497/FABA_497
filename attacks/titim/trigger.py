from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BadNetSquareConfig:
    intensity: float
    block_ratio: float = 0.125
    margin_ratio: float = 0.03125


def apply_badnet_square(image: Image.Image, config: BadNetSquareConfig) -> Image.Image:
    """Apply TITIM's white BadNet square with opacity as trigger intensity."""
    intensity = min(max(float(config.intensity), 0.0), 1.0)
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = base.shape[:2]
    short_side = min(height, width)
    block = max(1, min(short_side, int(round(short_side * float(config.block_ratio)))))
    margin = max(0, int(round(short_side * float(config.margin_ratio))))
    row_end = max(block, height - margin)
    col_end = max(block, width - margin)
    row_start = max(0, row_end - block)
    col_start = max(0, col_end - block)
    row_end = min(height, row_end)
    col_end = min(width, col_end)

    poisoned = base.copy()
    patch = poisoned[row_start:row_end, col_start:col_end]
    poisoned[row_start:row_end, col_start:col_end] = (
        patch * (1.0 - intensity) + intensity
    )
    return Image.fromarray(np.round(np.clip(poisoned, 0.0, 1.0) * 255.0).astype(np.uint8))

