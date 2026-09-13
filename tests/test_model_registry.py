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
        "Perception Encoder",
    }
    assert required <= families


def test_planned_models_are_not_publicly_available():
    planned = [model for model in public_model_registry() if model["status"] == "planned"]
    assert planned
    assert all(model["available"] is False for model in planned)


def test_hq_clip_is_wired_to_the_openclip_hub_encoder():
    """v6.7.0: HQ-CLIP B/16 promoted from planned to experimental.

    The checkpoint is the author's own OpenCLIP-compatible Hugging Face repo, so
    it needs no new loader: the existing ``hf-hub:`` branch carries it.
    """
    spec = MODEL_REGISTRY["hq_clip_b16"]
    assert spec.status == "experimental"
    assert spec.provider == "openclip_hf_hub"
    assert spec.model_id.startswith("hf-hub:")
    assert spec.requires == ("torch", "open_clip")
    assert spec.supports_image_embedding is True
    assert spec.supports_projection is True
    # Verified against the live checkpoint: the OpenCLIP tokenizer resolves, so
    # text search is a promise the model can keep.
    assert spec.supports_text_search is True
    assert spec.supports_text_embedding is True


def test_laclip_stays_planned():
    """v6.7.0: deliberately not integrated — see the spec notes.

    A community mirror exists and would load, but its provenance cannot be
    established. The entry stays visible so the decision is not silently lost.
    """
    spec = MODEL_REGISTRY["laclip_roadmap"]
    assert spec.status == "planned"
    assert spec.provider == "planned"


def test_pe_core_is_wired_to_the_openclip_hub_encoder():
    """v6.7.0: Perception Encoder B/16, verified on the reference collection.

    Same ``hf-hub:`` path as HQ-CLIP, but it pulls its vision tower through timm,
    so timm has to be declared or the model reports itself available and then
    fails at load time.
    """
    spec = MODEL_REGISTRY["pe_core_b16"]
    assert spec.status == "experimental"
    assert spec.provider == "openclip_hf_hub"
    assert spec.model_id.startswith("hf-hub:")
    assert "timm" in spec.requires
    assert spec.supports_text_search is True
    assert spec.supports_projection is True


def test_every_declared_requirement_has_an_import_name():
    """A requirement missing from IMPORT_NAMES silently falls back to its own
    spelling, which is right for timm and wrong for things like opencv-python."""
    from backend.encoders.registry import IMPORT_NAMES

    for spec in MODEL_REGISTRY.values():
        for requirement in spec.requires:
            assert requirement in IMPORT_NAMES, f"{spec.key} requires unmapped '{requirement}'"


def test_every_active_model_declares_a_licence():
    """Nine of these checkpoints forbid commercial use or restrict use to research.

    None of that was visible anywhere in the application until v6.7.0: a user
    could project a catalogue under CC BY-NC or Apple's research-only terms
    without ever being told. A missing licence is therefore a test failure, not
    a documentation gap.
    """
    for spec in MODEL_REGISTRY.values():
        if spec.status == "planned":
            continue
        assert spec.license, f"{spec.key} does not declare a licence"


def test_the_licence_reaches_the_browser():
    for model in public_model_registry():
        assert "license" in model


def test_restricted_licences_say_so_in_words():
    """'cc-by-nc-4.0' means nothing to an art historian; 'non-commercial' does."""
    restricted = {
        "metaclip_b32", "metaclip_l14",
        "metaclip2_worldwide_b32", "metaclip2_worldwide_h14",
        "mobileclip_b_openclip", "mobileclip2_s2", "mobileclip2_b", "mobileclip2_s4",
        "jina_v5_omni_small",
    }
    for key in restricted:
        licence = MODEL_REGISTRY[key].license.lower()
        assert "non-commercial" in licence or "research use only" in licence, key
