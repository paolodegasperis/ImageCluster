from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .encoders.base import encode_images
from .encoders.registry import get_model_spec


def encode_images_openclip(
    image_paths: list[Path],
    model_name: str,
    pretrained: str,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    """Backward-compatible wrapper retained for v4 code paths."""
    spec = get_model_spec(None, model_name, pretrained)
    return encode_images(image_paths, spec, batch_size, progress)
