from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .registry import ModelSpec


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _move_to_device(inputs, device: str):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _as_embedding_tensor(value, context: str):
    import torch

    if torch.is_tensor(value):
        tensor = value
    elif hasattr(value, "image_hidden_states") and getattr(value, "image_hidden_states") is not None:
        tensor = getattr(value, "image_hidden_states")
    elif hasattr(value, "image_embeds") and getattr(value, "image_embeds") is not None:
        tensor = getattr(value, "image_embeds")
    elif hasattr(value, "vision_hidden_states") and getattr(value, "vision_hidden_states") is not None:
        tensor = getattr(value, "vision_hidden_states")
    elif hasattr(value, "hidden_states") and getattr(value, "hidden_states") is not None:
        hidden = getattr(value, "hidden_states")
        tensor = hidden[-1] if isinstance(hidden, (tuple, list)) else hidden
    elif hasattr(value, "last_hidden_state") and getattr(value, "last_hidden_state") is not None:
        tensor = getattr(value, "last_hidden_state")
    elif hasattr(value, "keys"):
        tensor = None
        for key in ("image_hidden_states", "image_embeds", "vision_hidden_states", "last_hidden_state", "hidden_states"):
            candidate = value.get(key, None)
            if candidate is None:
                continue
            tensor = candidate[-1] if key == "hidden_states" and isinstance(candidate, (tuple, list)) else candidate
            break
        if tensor is None:
            raise RuntimeError(f"No image embedding tensor found in {context}.")
    elif isinstance(value, (tuple, list)):
        tensor = None
        for candidate in value:
            try:
                tensor = _as_embedding_tensor(candidate, context)
                break
            except Exception:
                continue
        if tensor is None:
            raise RuntimeError(f"No image embedding tensor found in {context}.")
    else:
        raise RuntimeError(f"Unsupported LLaVA-OneVision output type during {context}: {type(value).__name__}")

    if isinstance(tensor, (tuple, list)):
        tensor = tensor[-1]
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"LLaVA-OneVision did not return a tensor during {context}.")
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=1)
    elif tensor.ndim > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).mean(dim=1)
    elif tensor.ndim != 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor


def _normalize(tensor):
    import torch

    tensor = tensor.float()
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)


def _load_model_and_processor(spec: ModelSpec, device: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    model_class = getattr(transformers, "LlavaOnevisionForConditionalGeneration", None)
    if model_class is None:
        raise RuntimeError(
            "Transformers does not provide LlavaOnevisionForConditionalGeneration. "
            "Install a newer Transformers build to use LLaVA-OneVision visual features."
        )

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    model = model_class.from_pretrained(spec.model_id, torch_dtype=dtype, trust_remote_code=spec.trust_remote_code)
    return model.to(device).eval(), processor


def _extract_visual_features(model, inputs, provider: str):
    import torch

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            try:
                outputs = model.get_image_features(**inputs)
                return _as_embedding_tensor(outputs, f"get_image_features/{provider}")
            except Exception:
                pass

        try:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        except TypeError:
            outputs = model(**inputs, output_hidden_states=True)
        return _as_embedding_tensor(outputs, f"forward/{provider}")


def encode_with_llava_onevision(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    import torch

    device = _select_device()
    model, processor = _load_model_and_processor(spec, device)
    effective_batch_size = max(1, min(batch_size or 1, spec.recommended_batch_size or 1))

    vectors: list[np.ndarray] = []
    encoded_paths: list[Path] = []
    skipped: list[str] = []
    total = len(image_paths)

    def encode_batch(batch_paths: list[Path]) -> tuple[np.ndarray | None, list[Path]]:
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
        inputs = _move_to_device(inputs, device)
        emb = _normalize(_extract_visual_features(model, inputs, spec.provider))
        return emb.detach().cpu().numpy().astype("float32"), valid

    for start in range(0, total, effective_batch_size):
        batch_paths = image_paths[start:start + effective_batch_size]
        done = min(start + len(batch_paths), total)
        if progress:
            progress(start, total, f"Encoding {start + 1}-{done} of {total} images with {spec.family}")
        try:
            arr, valid = encode_batch(batch_paths)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" not in msg and "shape" not in msg and "size" not in msg:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if progress:
                progress(start, total, f"Batch failed; retrying image by image with {spec.family}")
            arrs = []
            valid = []
            for path in batch_paths:
                try:
                    one_arr, one_valid = encode_batch([path])
                    if one_arr is not None:
                        arrs.append(one_arr)
                        valid.extend(one_valid)
                except Exception as one_exc:
                    skipped.append(f"{path.name}: embedding failed: {one_exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            arr = np.vstack(arrs) if arrs else None

        if arr is not None and valid:
            vectors.append(arr)
            encoded_paths.extend(valid)
        if progress:
            progress(done, total, f"Encoded {done} of {total} images with {spec.family}")

    if not vectors:
        details = "\n".join(skipped[:20])
        raise RuntimeError("No valid images could be encoded." + (f" Details:\n{details}" if details else ""))
    return np.vstack(vectors).astype("float32"), encoded_paths
