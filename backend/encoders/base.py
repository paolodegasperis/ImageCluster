from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .imagebind_encoder import encode_with_imagebind
from .llava_onevision_encoder import encode_with_llava_onevision
from .openclip_encoder import encode_with_openclip
from .qwen25_vl_encoder import encode_with_qwen25_vl
from .registry import ModelSpec
from .transformers_encoder import encode_with_transformers
from .transformers_encoder import encode_texts_with_transformers


def encode_images(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    if spec.provider in {"openclip", "openclip_hf_hub"}:
        return encode_with_openclip(image_paths, spec, batch_size, progress)
    if spec.provider in {"transformers_clip", "transformers_image_features", "transformers_metaclip2", "transformers_vision_pool", "nomic_transformers"}:
        return encode_with_transformers(image_paths, spec, batch_size, progress)
    if spec.provider == "llava_onevision_visual":
        return encode_with_llava_onevision(image_paths, spec, batch_size, progress)
    if spec.provider == "qwen25_vl_visual":
        return encode_with_qwen25_vl(image_paths, spec, batch_size, progress)
    if spec.provider == "imagebind":
        return encode_with_imagebind(image_paths, spec, batch_size, progress)
    raise RuntimeError(f"Unsupported embedding provider: {spec.provider}")


def encode_texts(texts: list[str], spec: ModelSpec, batch_size: int = 32) -> np.ndarray:
    if not spec.supports_text_embedding or not spec.supports_text_search:
        raise RuntimeError("This model can create image projections but does not support text search.")
    if spec.provider in {"openclip", "openclip_hf_hub"}:
        from .openclip_encoder import encode_texts_with_openclip

        return encode_texts_with_openclip(texts, spec, batch_size)
    if spec.provider in {"transformers_clip", "transformers_image_features", "transformers_metaclip2"}:
        return encode_texts_with_transformers(texts, spec, batch_size)
    raise RuntimeError(f"Text search is not implemented for provider: {spec.provider}")
