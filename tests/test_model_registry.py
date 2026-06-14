from backend.encoders.registry import MODEL_REGISTRY, public_model_registry


def test_public_registry_exposes_capabilities():
    models = public_model_registry()
    assert models
    for model in models:
        assert "supports_image_embedding" in model
        assert "supports_text_embedding" in model
        assert "supports_text_search" in model
        assert "supports_projection" in model
        assert "status" in model


def test_dinov2_is_not_text_search_model():
    spec = MODEL_REGISTRY["dinov2_base"]
    assert spec.supports_image_embedding is True
    assert spec.supports_projection is True
    assert spec.supports_text_embedding is False
    assert spec.supports_text_search is False


def test_metaclip_entries_are_registered():
    assert MODEL_REGISTRY["metaclip_b32"].supports_text_search is True
    assert MODEL_REGISTRY["metaclip_b32"].provider == "openclip"
    assert MODEL_REGISTRY["metaclip2_worldwide_b32"].supports_text_search is True
    assert MODEL_REGISTRY["metaclip2_worldwide_b32"].status == "experimental"
    # V-5.4 Wave A: H/14 worldwide promoted to experimental via the transformers MetaCLIP 2 path.
    assert MODEL_REGISTRY["metaclip2_worldwide_h14"].status == "experimental"
    assert MODEL_REGISTRY["metaclip2_worldwide_h14"].provider == "transformers_metaclip2"
    assert MODEL_REGISTRY["metaclip2_worldwide_h14"].model_id


def test_wave_a_models_are_wired_to_real_encoders():
    """V-5.4 Wave A: planned roadmap entries promoted to experimental by reusing existing encoders."""
    for key in ("eva_clip_l14", "eva_clip_bigE14", "metaclip2_worldwide_h14"):
        spec = MODEL_REGISTRY[key]
        assert spec.status == "experimental"
        assert spec.provider != "planned"
        assert spec.model_id, f"{key} must have a real checkpoint id"
        assert spec.requires, f"{key} must declare runtime requirements"
        assert spec.supports_image_embedding is True
        assert spec.supports_projection is True
        # All three are CLIP-family → text search capable.
        assert spec.supports_text_search is True
        assert spec.supports_text_embedding is True
    # EVA variants route through OpenCLIP and need a pretrained tag.
    assert MODEL_REGISTRY["eva_clip_l14"].provider == "openclip"
    assert MODEL_REGISTRY["eva_clip_l14"].pretrained
    assert MODEL_REGISTRY["eva_clip_bigE14"].provider == "openclip"
    assert MODEL_REGISTRY["eva_clip_bigE14"].pretrained


def test_v55_multimodal_models_are_registered():
    """V-5.5: Qwen3-VL Embedding and Jina v5 Omni route through the sentence-transformers encoder."""
    for key in ("qwen3_vl_embedding_2b", "jina_v5_omni_small"):
        spec = MODEL_REGISTRY[key]
        assert spec.status == "experimental"
        assert spec.provider == "sentence_transformers"
        assert spec.model_id
        assert spec.trust_remote_code is True
        assert "sentence_transformers" in spec.requires
        assert spec.supports_image_embedding is True
        assert spec.supports_text_search is True
        assert spec.supports_text_embedding is True
        assert spec.supports_projection is True


def test_required_model_families_are_present():
    families = {model.family for model in MODEL_REGISTRY.values()}
    required = {
        "OpenCLIP",
        "CLIP",
        "SigLIP",
        "SigLIP 2",
        "MobileCLIP",
        "MobileCLIP2",
        "DINOv2",
        "Nomic Embed Vision",
        "ImageBind",
        "MetaCLIP",
        "MetaCLIP 2",
        "HQ-CLIP",
        "Long-CLIP",
        "EVA-CLIP",
        "CLOC",
        "LACLIP",
    }
    assert required <= families


def test_planned_models_are_not_publicly_available():
    planned = [model for model in public_model_registry() if model["status"] == "planned"]
    assert planned
    assert all(model["available"] is False for model in planned)
