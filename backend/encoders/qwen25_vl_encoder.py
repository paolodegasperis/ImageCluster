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


def _first_useful_tensor(value, context: str):
    import torch

    if torch.is_tensor(value):
        return value
    for attr in ("last_hidden_state", "image_embeds", "pooler_output", "hidden_states"):
        if hasattr(value, attr):
            candidate = getattr(value, attr)
            if candidate is None:
                continue
            if attr == "hidden_states" and isinstance(candidate, (tuple, list)):
                candidate = candidate[-1]
            if torch.is_tensor(candidate):
                return candidate
    if hasattr(value, "keys"):
        for key in ("last_hidden_state", "image_embeds", "pooler_output", "hidden_states"):
            candidate = value.get(key, None)
            if candidate is None:
                continue
            if key == "hidden_states" and isinstance(candidate, (tuple, list)):
                candidate = candidate[-1]
            if torch.is_tensor(candidate):
                return candidate
    if isinstance(value, (tuple, list)):
        for candidate in value:
            try:
                return _first_useful_tensor(candidate, context)
            except Exception:
                continue
    raise RuntimeError(f"No visual embedding tensor found in Qwen2.5-VL output during {context}.")


def _pool_visual_tensor(tensor, batch_size: int):
    import torch

    if not torch.is_tensor(tensor):
        raise RuntimeError("Qwen2.5-VL visual output is not a tensor.")
    if tensor.ndim == 3:
        tensor = tensor.mean(dim=1)
    elif tensor.ndim == 2:
        if tensor.shape[0] == batch_size:
            pass
        elif batch_size == 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        else:
            tensor = tensor.reshape(batch_size, -1, tensor.shape[-1]).mean(dim=1)
    elif tensor.ndim > 3:
        tensor = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).mean(dim=1)
    else:
        tensor = tensor.reshape(batch_size, -1)
    return torch.nn.functional.normalize(tensor.float(), p=2, dim=-1)


def _load_processor(spec: ModelSpec):
    from transformers import AutoProcessor

    kwargs = {"trust_remote_code": spec.trust_remote_code}
    try:
        return AutoProcessor.from_pretrained(
            spec.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
            **kwargs,
        )
    except TypeError:
        return AutoProcessor.from_pretrained(spec.model_id, **kwargs)


def _preprocess_images(processor, images):
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        return processor.image_processor(images=images, return_tensors="pt")

    image_token = getattr(processor, "image_token", "<|image_pad|>")
    try:
        return processor(images=images, text=[image_token] * len(images), return_tensors="pt")
    except TypeError as exc:
        raise RuntimeError(
            "Qwen2.5-VL image preprocessing failed. This Transformers build requires "
            "text placeholders for multimodal preprocessing, but the image-only fallback "
            "could not create them."
        ) from exc


def _load_model(spec: ModelSpec, device: str):
    import torch
    import transformers

    model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
    if model_class is None:
        raise RuntimeError(
            "Transformers does not provide Qwen2_5_VLForConditionalGeneration. "
            "Install a newer Transformers build to use Qwen2.5-VL visual features."
        )

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = model_class.from_pretrained(spec.model_id, torch_dtype=dtype, trust_remote_code=spec.trust_remote_code)
    return model.to(device).eval()


def _extract_visual_features(model, inputs, batch_size: int, provider: str):
    import torch

    with torch.no_grad():
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("Qwen2.5-VL image preprocessing did not return pixel_values and image_grid_thw.")

        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
            )
            return _pool_visual_tensor(_first_useful_tensor(outputs, f"get_image_features/{provider}"), batch_size)

        visual = getattr(model, "visual", None)
        if visual is None and hasattr(model, "model"):
            visual = getattr(model.model, "visual", None)
        if visual is not None:
            outputs = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
            return _pool_visual_tensor(_first_useful_tensor(outputs, f"visual/{provider}"), batch_size)

        raise RuntimeError("Qwen2.5-VL model does not expose a visual encoder interface.")


def encode_with_qwen25_vl(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    import torch

    device = _select_device()
    processor = _load_processor(spec)
    model = _load_model(spec, device)
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
        inputs = _preprocess_images(processor, images)
        inputs = _move_to_device(inputs, device)
        emb = _extract_visual_features(model, inputs, len(images), spec.provider)
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
