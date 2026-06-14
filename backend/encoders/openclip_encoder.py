from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ..config_store import apply_hf_token_to_environment
from .registry import ModelSpec


def encode_with_openclip(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    apply_hf_token_to_environment()
    if spec.provider == "openclip_hf_hub" or spec.model_id.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(spec.model_id)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(spec.model_id, pretrained=spec.pretrained or None)
    model = model.to(device).eval()

    vectors: list[np.ndarray] = []
    encoded_paths: list[Path] = []
    total = len(image_paths)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            tensors = []
            valid = []
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    tensors.append(preprocess(image))
                    valid.append(path)
                except Exception:
                    continue
            if tensors:
                batch = torch.stack(tensors).to(device)
                emb = model.encode_image(batch)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                vectors.append(emb.detach().cpu().numpy())
                encoded_paths.extend(valid)
            done = min(start + len(batch_paths), total)
            if progress:
                progress(done, total, f"Encoded {done} of {total} images with {spec.family}")

    if not vectors:
        raise RuntimeError("No valid images could be encoded.")
    return np.vstack(vectors).astype("float32"), encoded_paths


def encode_texts_with_openclip(texts: list[str], spec: ModelSpec, batch_size: int = 32) -> np.ndarray:
    import torch
    import open_clip

    if not texts:
        raise RuntimeError("No text queries were provided.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    apply_hf_token_to_environment()
    if spec.provider == "openclip_hf_hub" or spec.model_id.startswith("hf-hub:"):
        model, _, _ = open_clip.create_model_and_transforms(spec.model_id)
    else:
        model, _, _ = open_clip.create_model_and_transforms(spec.model_id, pretrained=spec.pretrained or None)
    tokenizer = open_clip.get_tokenizer(spec.model_id)
    model = model.to(device).eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vectors.append(emb.detach().cpu().numpy())

    return np.vstack(vectors).astype("float32")
