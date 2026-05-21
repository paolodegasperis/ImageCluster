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
    assert MODEL_REGISTRY["metaclip2_worldwide_h14"].status == "planned"


def test_mobileclip2_entries_are_experimental_and_searchable():
    for key in ("mobileclip2_s2", "mobileclip2_b", "mobileclip2_s4"):
        spec = MODEL_REGISTRY[key]
        assert spec.provider == "openclip_hf_hub"
        assert spec.status == "experimental"
        assert spec.supports_projection is True
        assert spec.supports_text_search is True


def test_experimental_vlm_visual_models_are_image_only():
    expected = {
        "llava_onevision_qwen2_05b_image_only": ("LLaVA-OneVision", "llava_onevision_visual"),
        "qwen25_vl_3b_image_only": ("Qwen2.5-VL", "qwen25_vl_visual"),
    }
    for key, (family, provider) in expected.items():
        spec = MODEL_REGISTRY[key]
        assert spec.family == family
        assert spec.provider == provider
        assert spec.status == "experimental"
        assert spec.supports_projection is True
        assert spec.supports_image_embedding is True
        assert spec.supports_text_embedding is False
        assert spec.supports_text_search is False


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
        "LLaVA-OneVision",
        "Qwen2.5-VL",
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
