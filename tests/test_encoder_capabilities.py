from backend.encoders.registry import MODEL_REGISTRY


def test_projection_models_have_image_embedding_capability():
    for spec in MODEL_REGISTRY.values():
        if spec.supports_projection:
            assert spec.supports_image_embedding is True


def test_text_search_requires_text_embedding():
    for spec in MODEL_REGISTRY.values():
        if spec.supports_text_search:
            assert spec.supports_text_embedding is True


def test_image_only_models_do_not_advertise_text_search():
    image_only = [
        "dinov2_base",
        "nomic_embed_vision_v1_5",
        "llava_onevision_qwen2_05b_image_only",
        "qwen25_vl_3b_image_only",
    ]
    for key in image_only:
        assert MODEL_REGISTRY[key].supports_text_embedding is False
        assert MODEL_REGISTRY[key].supports_text_search is False
