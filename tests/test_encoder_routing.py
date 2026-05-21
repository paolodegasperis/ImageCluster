from pathlib import Path

import numpy as np

from backend.encoders import base
from backend.encoders.registry import MODEL_REGISTRY


def test_visual_vlm_providers_route_to_dedicated_encoders(monkeypatch):
    calls = []

    def fake_llava(paths, spec, batch_size, progress=None):
        calls.append((spec.provider, batch_size, tuple(paths)))
        return np.zeros((1, 2), dtype="float32"), paths

    def fake_qwen(paths, spec, batch_size, progress=None):
        calls.append((spec.provider, batch_size, tuple(paths)))
        return np.ones((1, 2), dtype="float32"), paths

    monkeypatch.setattr(base, "encode_with_llava_onevision", fake_llava)
    monkeypatch.setattr(base, "encode_with_qwen25_vl", fake_qwen)

    image_paths = [Path("image.jpg")]
    llava_vectors, llava_paths = base.encode_images(
        image_paths,
        MODEL_REGISTRY["llava_onevision_qwen2_05b_image_only"],
        batch_size=1,
    )
    qwen_vectors, qwen_paths = base.encode_images(
        image_paths,
        MODEL_REGISTRY["qwen25_vl_3b_image_only"],
        batch_size=1,
    )

    assert calls == [
        ("llava_onevision_visual", 1, tuple(image_paths)),
        ("qwen25_vl_visual", 1, tuple(image_paths)),
    ]
    assert llava_vectors.shape == (1, 2)
    assert qwen_vectors.shape == (1, 2)
    assert llava_paths == image_paths
    assert qwen_paths == image_paths
