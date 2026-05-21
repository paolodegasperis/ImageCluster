from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .registry import ModelSpec


def encode_with_imagebind(
    image_paths: list[Path],
    spec: ModelSpec,
    batch_size: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    """Encode images with Meta ImageBind when the optional repository is installed.

    Expected external installation follows Meta's ImageBind repository layout,
    exposing the `imagebind` Python package with `data` and `models.imagebind_model`.
    """
    try:
        import torch
        from imagebind import data
        import imagebind.models.imagebind_model as imagebind_model
        from imagebind.models.imagebind_model import ModalityType
    except Exception as exc:
        raise RuntimeError(
            "ImageBind is optional and is not installed in this environment. "
            "Use the optional installer script (Install Optional ImageBind Windows.bat or Install Optional ImageBind macOS.command), then restart ImagePlot-CLIP."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model = model.to(device).eval()

    vectors: list[np.ndarray] = []
    encoded_paths: list[Path] = []
    total = len(image_paths)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_paths = image_paths[start:start + batch_size]
            valid_paths = [p for p in batch_paths if p.exists()]
            if valid_paths:
                try:
                    inputs = {ModalityType.VISION: data.load_and_transform_vision_data([str(p) for p in valid_paths], device)}
                    emb = model(inputs)[ModalityType.VISION]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                    vectors.append(emb.detach().cpu().numpy())
                    encoded_paths.extend(valid_paths)
                except Exception:
                    # If a batch fails because of a corrupt image, retry one by one.
                    for p in valid_paths:
                        try:
                            inputs = {ModalityType.VISION: data.load_and_transform_vision_data([str(p)], device)}
                            emb = model(inputs)[ModalityType.VISION]
                            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                            vectors.append(emb.detach().cpu().numpy())
                            encoded_paths.append(p)
                        except Exception:
                            continue
            done = min(start + len(batch_paths), total)
            if progress:
                progress(done, total, f"Encoded {done} of {total} images with ImageBind")

    if not vectors:
        raise RuntimeError("No valid images could be encoded with ImageBind.")
    return np.vstack(vectors).astype("float32"), encoded_paths
