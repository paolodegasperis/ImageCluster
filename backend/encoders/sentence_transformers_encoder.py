"""Encoder for multimodal embedding models exposed through the Sentence-Transformers
API (``SentenceTransformer(model_id).encode(...)``).

Used for models that are not standard CLIP ``get_image_features`` checkpoints, e.g.
Qwen3-VL-Embedding and Jina Embeddings v5 Omni. Both load via SentenceTransformer
(usually with ``trust_remote_code=True``) and produce image and text embeddings in a
shared space through ``model.encode``.

`sentence-transformers` is an optional dependency: models using this provider are
reported as unavailable (and disabled in the UI) until it is installed, mirroring the
ImageBind pattern.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from ..config_store import apply_hf_token_to_environment
from .model_cache import get_or_load
from .registry import ModelSpec

# Loaded models are cached per process (shared LRU in model_cache) so image encoding and
# a later text search do not reload a multi-GB checkpoint twice.


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_kwargs(spec: ModelSpec) -> dict:
    kwargs: dict = {}
    # Jina v5 Omni expects a default task for retrieval-style embeddings.
    if "jina" in spec.model_id.lower():
        kwargs["default_task"] = "retrieval"
    # See the note in transformers_encoder._load_model: Transformers 5 honours the
    # saved dtype, and a bfloat16 checkpoint on CPU returns vectors whose norms
    # miss unity by enough to matter here.
    if _select_device() == "cpu":
        kwargs["dtype"] = "float32"
    return kwargs


def _load_model(spec: ModelSpec):
    def loader():
        apply_hf_token_to_environment()
        from sentence_transformers import SentenceTransformer

        model_kwargs = _model_kwargs(spec)
        kwargs = {"trust_remote_code": spec.trust_remote_code}
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        return SentenceTransformer(spec.model_id, **kwargs)

    return get_or_load(f"sentence-transformers|{spec.model_id}", loader)


def _encode(model, items, batch_size: int, prompt_name: str = "") -> np.ndarray:
    kwargs = {}
    # Only pass a prompt the model actually declares: Sentence-Transformers raises
    # on an unknown prompt name, and a model without prompts must keep working.
    if prompt_name and prompt_name in (getattr(model, "prompts", None) or {}):
        kwargs["prompt_name"] = prompt_name
    emb = model.encode(
        items,
        batch_size=max(1, batch_size),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        **kwargs,
    )
    out = np.asarray(emb, dtype="float32")
    # The library normalized in the model's own dtype; redo it in float32 so a
    # half-precision checkpoint still yields unit-norm vectors. See the note in
    # transformers_encoder._normalize.
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.divide(out, np.where(norms > 0, norms, 1.0)).astype("float32")


def encode_with_sentence_transformers(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    model = _load_model(spec)
    vectors: list[np.ndarray] = []
    encoded_paths: list[Path] = []
    skipped: list[str] = []
    total = len(image_paths)

    for start in range(0, total, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        done = min(start + len(batch_paths), total)
        if progress:
            progress(start, total, f"Encoding {start + 1}-{done} of {total} images with {spec.family}")

        images = []
        valid: list[Path] = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid.append(path)
            except Exception as exc:
                skipped.append(f"{path.name}: image read failed: {exc}")
        if not images:
            continue
        try:
            arr = _encode(model, images, len(images))
        except Exception:
            # Resilience: retry image-by-image so one bad file/batch does not fail the job.
            arrs = []
            valids = []
            for path, image in zip(valid, images, strict=True):
                try:
                    arrs.append(_encode(model, [image], 1))
                    valids.append(path)
                except Exception as one_exc:
                    skipped.append(f"{path.name}: embedding failed: {one_exc}")
            if not arrs:
                continue
            arr = np.vstack(arrs)
            valid = valids
        vectors.append(arr)
        encoded_paths.extend(valid)
        if progress:
            progress(done, total, f"Encoded {done} of {total} images with {spec.family}")

    if not vectors:
        details = "\n".join(skipped[:20])
        raise RuntimeError("No valid images could be encoded." + (f" Details:\n{details}" if details else ""))
    return np.vstack(vectors).astype("float32"), encoded_paths


def encode_texts_with_sentence_transformers(texts: list[str], spec: ModelSpec, batch_size: int = 32) -> np.ndarray:
    if not texts:
        raise RuntimeError("No text queries were provided.")
    model = _load_model(spec)
    return _encode(model, list(texts), batch_size, prompt_name=spec.text_prompt_name)
