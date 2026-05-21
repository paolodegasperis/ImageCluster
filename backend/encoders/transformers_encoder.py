from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .registry import ModelSpec


def _move_to_device(inputs, device: str):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def _normalize(emb):
    import torch
    emb = _coerce_to_tensor(emb, context="normalization")
    if emb.ndim == 3:
        emb = emb[:, 0]
    if emb.ndim != 2:
        emb = emb.reshape(emb.shape[0], -1)
    return torch.nn.functional.normalize(emb, p=2, dim=-1)


def _coerce_to_tensor(value, context: str = "model output"):
    """Convert common Hugging Face model outputs to a 2D tensor.

    Several image encoders return ModelOutput objects rather than tensors. Earlier
    versions passed those objects directly to torch normalization, causing errors
    such as: 'BaseModelOutputWithPooling' object has no attribute 'norm'.
    """
    import torch

    if torch.is_tensor(value):
        return value

    # Hugging Face ModelOutput objects usually expose useful named attributes.
    for attr in (
        "image_embeds",
        "pooler_output",
        "last_hidden_state",
        "hidden_states",
        "embeds",
        "logits",
    ):
        if hasattr(value, attr):
            candidate = getattr(value, attr)
            if candidate is None:
                continue
            if attr == "hidden_states" and isinstance(candidate, (tuple, list)) and candidate:
                candidate = candidate[-1]
            if torch.is_tensor(candidate):
                if candidate.ndim == 3:
                    return candidate[:, 0]
                return candidate

    # Some ModelOutput classes behave like dicts.
    if hasattr(value, "keys"):
        for key in (
            "image_embeds",
            "pooler_output",
            "last_hidden_state",
            "hidden_states",
            "embeds",
            "logits",
        ):
            try:
                candidate = value.get(key, None)
            except Exception:
                candidate = None
            if candidate is None:
                continue
            if key == "hidden_states" and isinstance(candidate, (tuple, list)) and candidate:
                candidate = candidate[-1]
            if torch.is_tensor(candidate):
                if candidate.ndim == 3:
                    return candidate[:, 0]
                return candidate

    # Tuples/lists: use first tensor-like item.
    if isinstance(value, (tuple, list)):
        for candidate in value:
            if torch.is_tensor(candidate):
                if candidate.ndim == 3:
                    return candidate[:, 0]
                return candidate
            try:
                coerced = _coerce_to_tensor(candidate, context=context)
                if torch.is_tensor(coerced):
                    return coerced
            except Exception:
                pass

    raise RuntimeError(
        f"Cannot convert {type(value).__name__} to an embedding tensor during {context}. "
        "This model returned an unsupported output structure. Try OpenCLIP or CLIP, "
        "or open the job debug log for model-specific details."
    )


def _extract_embedding(model, inputs, provider: str):
    """Return image embeddings from common Transformers vision/image-text models."""
    import torch

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            try:
                outputs = model.get_image_features(**inputs)
            except TypeError:
                if "pixel_values" in inputs:
                    outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
                else:
                    raise
            return _coerce_to_tensor(outputs, context=f"get_image_features/{provider}")

        outputs = model(**inputs)

    return _coerce_to_tensor(outputs, context=f"forward/{provider}")


def _extract_text_embedding(model, inputs, provider: str):
    import torch

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            try:
                outputs = model.get_text_features(**inputs)
            except TypeError:
                text_inputs = {k: v for k, v in inputs.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
                outputs = model.get_text_features(**text_inputs)
            return _coerce_to_tensor(outputs, context=f"get_text_features/{provider}")

        outputs = model(**inputs)

    return _coerce_to_tensor(outputs, context=f"text_forward/{provider}")


def _load_processor(spec: ModelSpec):
    from transformers import AutoImageProcessor, AutoProcessor

    try:
        return AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    except Exception:
        return AutoImageProcessor.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)


def _load_model(spec: ModelSpec, device: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    return model.to(device).eval()


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_with_transformers(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    import torch

    device = _select_device()
    processor = _load_processor(spec)
    model = _load_model(spec, device)

    vectors: list[np.ndarray] = []
    encoded_paths: list[Path] = []
    skipped: list[str] = []
    total = len(image_paths)

    def encode_batch(batch_paths: list[Path], current_device: str) -> tuple[np.ndarray | None, list[Path]]:
        images = []
        valid = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid.append(path)
            except Exception as exc:
                skipped.append(f"{path.name}: image read failed: {exc}")
        if not images:
            return None, []
        inputs = processor(images=images, return_tensors="pt")
        inputs = _move_to_device(inputs, current_device)
        emb = _extract_embedding(model, inputs, spec.provider)
        emb = _normalize(emb)
        return emb.detach().cpu().numpy(), valid

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            done = min(start + len(batch_paths), total)
            if progress:
                progress(start, total, f"Encoding {start + 1}-{done} of {total} images with {spec.family}")

            try:
                arr, valid = encode_batch(batch_paths, device)
            except RuntimeError as exc:
                msg = str(exc).lower()
                # Recovery path for out-of-memory or backend-specific batch failures:
                # retry the same batch image-by-image. This is slower but avoids losing
                # an entire job for one problematic file or one too-large batch.
                if "out of memory" not in msg and "cannot convert" not in msg and "shape" not in msg:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if progress:
                    progress(start, total, f"Batch failed; retrying image by image with {spec.family}")
                arrs = []
                valids = []
                for path in batch_paths:
                    try:
                        one_arr, one_valid = encode_batch([path], device)
                        if one_arr is not None:
                            arrs.append(one_arr)
                            valids.extend(one_valid)
                    except Exception as one_exc:
                        skipped.append(f"{path.name}: embedding failed: {one_exc}")
                arr = np.vstack(arrs) if arrs else None
                valid = valids

            if arr is not None and len(valid):
                vectors.append(arr)
                encoded_paths.extend(valid)
            if progress:
                progress(done, total, f"Encoded {done} of {total} images with {spec.family}")

    if not vectors:
        details = "\n".join(skipped[:20])
        raise RuntimeError("No valid images could be encoded." + (f" Details:\n{details}" if details else ""))
    return np.vstack(vectors).astype("float32"), encoded_paths


def encode_texts_with_transformers(texts: list[str], spec: ModelSpec, batch_size: int = 32) -> np.ndarray:
    if not texts:
        raise RuntimeError("No text queries were provided.")

    device = _select_device()
    processor = _load_processor(spec)
    model = _load_model(spec, device)

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        try:
            inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
        except TypeError:
            inputs = processor(batch_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = _move_to_device(inputs, device)
        emb = _extract_text_embedding(model, inputs, spec.provider)
        emb = _normalize(emb)
        vectors.append(emb.detach().cpu().numpy())

    return np.vstack(vectors).astype("float32")
