from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    label: str
    provider: str
    model_id: str
    pretrained: str = ""
    published: str = ""
    description: str = ""
    default: bool = False
    requires: tuple[str, ...] = ()
    trust_remote_code: bool = False
    recommended_batch_size: int = 16
    notes: str = ""
    supports_image_embedding: bool = True
    supports_text_embedding: bool = False
    supports_text_search: bool = False
    supports_projection: bool = True
    optional_local_features: bool = False
    recommended_for: tuple[str, ...] = ()
    hardware_tier: str = "cpu_ok"
    status: str = "stable"

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requires"] = list(self.requires)
        data["recommended_for"] = list(self.recommended_for)
        missing = missing_requirements(self)
        data["available"] = not missing and self.status not in {"planned", "unavailable"}
        data["missing_requirements"] = missing
        return data


IMPORT_NAMES = {
    "torch": "torch",
    "open_clip": "open_clip",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "einops": "einops",
    "imagebind": "imagebind",
    "sentence_transformers": "sentence_transformers",
    "qwen_vl_utils": "qwen_vl_utils",
    "peft": "peft",
}


def missing_requirements(spec: ModelSpec) -> list[str]:
    missing = []
    for requirement in spec.requires:
        import_name = IMPORT_NAMES.get(requirement, requirement)
        if importlib.util.find_spec(import_name) is None:
            missing.append(requirement)
    return missing


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "openclip_vit_b_32": ModelSpec(
        key="openclip_vit_b_32",
        family="OpenCLIP",
        label="OpenCLIP · ViT-B-32 · LAION-2B",
        provider="openclip",
        model_id="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        published="2021-01-05 / OpenCLIP 2022",
        description="Default compatible OpenCLIP encoder used by earlier ImagePlot-CLIP versions.",
        default=True,
        requires=("torch", "open_clip"),
        recommended_batch_size=32,
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("general", "art_collection"),
        hardware_tier="cpu_ok",
    ),
    "openai_clip_vit_b_32": ModelSpec(
        key="openai_clip_vit_b_32",
        family="CLIP",
        label="CLIP · OpenAI ViT-B/32",
        provider="transformers_clip",
        model_id="openai/clip-vit-base-patch32",
        published="2021-01-05",
        description="Original CLIP family checkpoint exposed through Hugging Face Transformers.",
        requires=("torch", "transformers"),
        recommended_batch_size=32,
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("general", "art_collection"),
        hardware_tier="cpu_ok",
    ),
    "siglip_base_patch16_224": ModelSpec(
        key="siglip_base_patch16_224",
        family="SigLIP",
        label="SigLIP · Base patch16 224",
        provider="transformers_image_features",
        model_id="google/siglip-base-patch16-224",
        published="2023-03-27",
        description="Google SigLIP image-text encoder with sigmoid loss; good general-purpose visual-semantic embedding.",
        requires=("torch", "transformers"),
        recommended_batch_size=16,
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("general", "art_collection"),
        hardware_tier="cpu_ok",
    ),
    "siglip2_base_patch16_224": ModelSpec(
        key="siglip2_base_patch16_224",
        family="SigLIP 2",
        label="SigLIP 2 · Base patch16 224",
        provider="transformers_image_features",
        model_id="google/siglip2-base-patch16-224",
        published="2025-02-20",
        description="Newer multilingual SigLIP-family encoder with improved semantic understanding and dense features.",
        requires=("torch", "transformers"),
        recommended_batch_size=8,
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("general", "art_collection", "multilingual"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "mobileclip_b_openclip": ModelSpec(
        key="mobileclip_b_openclip",
        family="MobileCLIP",
        label="MobileCLIP · B · OpenCLIP checkpoint",
        provider="openclip_hf_hub",
        model_id="hf-hub:apple/MobileCLIP-B-OpenCLIP",
        published="2024-04",
        description="Apple MobileCLIP checkpoint packaged for OpenCLIP; intended as a lighter CLIP-like option.",
        requires=("torch", "open_clip"),
        recommended_batch_size=32,
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("general", "fast", "local"),
        hardware_tier="cpu_ok",
        status="experimental",
    ),
    "dinov2_base": ModelSpec(
        key="dinov2_base",
        family="DINOv2",
        label="DINOv2 · Base",
        provider="transformers_vision_pool",
        model_id="facebook/dinov2-base",
        published="2023-04-14",
        description="Image-only self-supervised vision foundation model; useful for visual/formal-semantic projections.",
        requires=("torch", "transformers"),
        recommended_batch_size=16,
        supports_text_embedding=False,
        supports_text_search=False,
        recommended_for=("visual_similarity", "formal_analysis"),
        hardware_tier="cpu_ok",
    ),
    "nomic_embed_vision_v1_5": ModelSpec(
        key="nomic_embed_vision_v1_5",
        family="Nomic Embed Vision",
        label="Nomic Embed Vision · v1.5",
        provider="nomic_transformers",
        model_id="nomic-ai/nomic-embed-vision-v1.5",
        published="2024-06-06",
        description="Vision embedding model aligned with Nomic text embeddings; requires remote model code from Hugging Face.",
        requires=("torch", "transformers", "einops"),
        trust_remote_code=True,
        recommended_batch_size=16,
        notes="Uses trust_remote_code=True. First run downloads model code and weights from Hugging Face.",
        supports_text_embedding=False,
        supports_text_search=False,
        recommended_for=("visual_similarity",),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "imagebind_huge": ModelSpec(
        key="imagebind_huge",
        family="ImageBind",
        label="ImageBind · Huge",
        provider="imagebind",
        model_id="imagebind_huge",
        published="2023-05-09",
        description="Meta ImageBind multimodal embedding model. Optional manual dependency because it is not a standard PyPI package.",
        requires=("torch", "imagebind"),
        recommended_batch_size=8,
        notes="Install Meta ImageBind manually from its repository before using this option.",
        supports_text_embedding=False,
        supports_text_search=False,
        recommended_for=("advanced_multimodal",),
        hardware_tier="large_gpu",
        status="experimental",
    ),
    "metaclip_b32": ModelSpec(
        key="metaclip_b32",
        family="MetaCLIP",
        label="MetaCLIP · ViT-B/32",
        provider="openclip",
        model_id="ViT-B-32",
        pretrained="metaclip_400m",
        published="2023",
        description="MetaCLIP ViT-B/32 through OpenCLIP, useful for broader web-scale visual-language retrieval.",
        requires=("torch", "open_clip"),
        recommended_batch_size=32,
        notes="Pretrained tag is used through the installed OpenCLIP build; if unavailable, the search request reports a load error without breaking other models.",
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("general", "art_collection", "multilingual"),
        hardware_tier="cpu_ok",
        status="experimental",
    ),
    "metaclip_l14": ModelSpec(
        key="metaclip_l14",
        family="MetaCLIP",
        label="MetaCLIP · ViT-L/14",
        provider="openclip",
        model_id="ViT-L-14",
        pretrained="metaclip_400m",
        published="2023",
        description="Larger MetaCLIP OpenCLIP variant for stronger semantic retrieval.",
        requires=("torch", "open_clip"),
        recommended_batch_size=8,
        notes="Experimental local option; verify availability in the installed OpenCLIP package before relying on it.",
        supports_text_embedding=True,
        supports_text_search=True,
        recommended_for=("art_collection", "multilingual"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "metaclip2_worldwide_h14": ModelSpec(
        key="metaclip2_worldwide_h14",
        family="MetaCLIP 2",
        label="MetaCLIP 2 · Worldwide H/14",
        provider="transformers_metaclip2",
        model_id="facebook/metaclip-2-worldwide-huge-quickgelu",
        published="2025-07-29",
        description="MetaCLIP 2 worldwide ViT-H/14 (quickgelu) for strong multilingual visual-language retrieval.",
        requires=("torch", "transformers"),
        recommended_batch_size=4,
        notes="Experimental: needs a recent Transformers build with MetaCLIP 2 support and downloads the H/14 checkpoint on first use. Large GPU recommended.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("art_collection", "multilingual"),
        hardware_tier="large_gpu",
        status="experimental",
    ),
    "metaclip2_worldwide_b32": ModelSpec(
        key="metaclip2_worldwide_b32",
        family="MetaCLIP 2",
        label="MetaCLIP 2 · Worldwide B/32",
        provider="transformers_metaclip2",
        model_id="facebook/metaclip-2-worldwide-b32",
        published="2025",
        description="Practical MetaCLIP 2 B/32 worldwide checkpoint for local projection and text search.",
        requires=("torch", "transformers"),
        recommended_batch_size=8,
        notes="Experimental: requires a recent Transformers build with MetaCLIP 2 support and downloads the checkpoint on first use.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("art_collection", "multilingual"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "metaclip2_2b_worldwide": ModelSpec(
        key="metaclip2_2b_worldwide",
        family="MetaCLIP 2",
        label="MetaCLIP 2 · 2B Worldwide",
        provider="planned",
        model_id="",
        published="planned",
        description="Roadmap entry for larger MetaCLIP 2 worldwide checkpoints.",
        recommended_batch_size=2,
        notes="Planned: no verified local checkpoint is wired yet.",
        supports_image_embedding=False,
        supports_text_embedding=False,
        supports_text_search=False,
        supports_projection=False,
        recommended_for=("art_collection", "multilingual"),
        hardware_tier="large_gpu",
        status="planned",
    ),
    "mobileclip2_s2": ModelSpec(
        key="mobileclip2_s2",
        family="MobileCLIP2",
        label="MobileCLIP2 · S2",
        provider="openclip_hf_hub",
        model_id="hf-hub:timm/MobileCLIP2-S2-OpenCLIP",
        published="2025",
        description="Lightweight MobileCLIP2 S2 checkpoint packaged for OpenCLIP.",
        requires=("torch", "open_clip"),
        recommended_batch_size=32,
        notes="Experimental: uses the timm OpenCLIP-compatible Hugging Face checkpoint and downloads weights on first use.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("fast", "local"),
        hardware_tier="cpu_ok",
        status="experimental",
    ),
    "mobileclip2_b": ModelSpec(
        key="mobileclip2_b",
        family="MobileCLIP2",
        label="MobileCLIP2 · B",
        provider="openclip_hf_hub",
        model_id="hf-hub:timm/MobileCLIP2-B-OpenCLIP",
        published="2025",
        description="Balanced MobileCLIP2 B checkpoint packaged for OpenCLIP.",
        requires=("torch", "open_clip"),
        recommended_batch_size=24,
        notes="Experimental: uses the timm OpenCLIP-compatible Hugging Face checkpoint and downloads weights on first use.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("fast", "local"),
        hardware_tier="cpu_ok",
        status="experimental",
    ),
    "mobileclip2_s4": ModelSpec(
        key="mobileclip2_s4",
        family="MobileCLIP2",
        label="MobileCLIP2 · S4",
        provider="openclip_hf_hub",
        model_id="hf-hub:timm/MobileCLIP2-S4-OpenCLIP",
        published="2025",
        description="Larger MobileCLIP2 S4 checkpoint packaged for OpenCLIP.",
        requires=("torch", "open_clip"),
        recommended_batch_size=16,
        notes="Experimental: uses the timm OpenCLIP-compatible Hugging Face checkpoint and downloads weights on first use.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("fast", "local"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "hq_clip_b16": ModelSpec(
        key="hq_clip_b16",
        family="HQ-CLIP",
        label="HQ-CLIP · B/16",
        provider="planned",
        model_id="",
        published="planned",
        description="Roadmap entry for curated high-quality image-text alignment.",
        notes="Planned until a verified local checkpoint and loader are added.",
        supports_image_embedding=False,
        supports_projection=False,
        recommended_for=("art_collection",),
        hardware_tier="gpu_recommended",
        status="planned",
    ),
    "long_clip_b32": ModelSpec(
        key="long_clip_b32",
        family="Long-CLIP",
        label="Long-CLIP · B/32",
        provider="planned",
        model_id="",
        published="planned",
        description="Roadmap entry for long catalogue-style text queries.",
        notes="Planned until a verified custom loading path is added.",
        supports_image_embedding=False,
        supports_projection=False,
        recommended_for=("art_collection", "long_queries"),
        hardware_tier="gpu_recommended",
        status="planned",
    ),
    "eva_clip_l14": ModelSpec(
        key="eva_clip_l14",
        family="EVA-CLIP",
        label="EVA-CLIP · L/14",
        provider="openclip",
        model_id="EVA02-L-14",
        pretrained="merged2b_s4b_b131k",
        published="2023",
        description="EVA-02 CLIP Large (ViT-L/14) through OpenCLIP; strong general visual-language retrieval and text search.",
        requires=("torch", "open_clip"),
        recommended_batch_size=8,
        notes="Experimental: loads the EVA-02 L/14 checkpoint through OpenCLIP and downloads weights on first use. GPU recommended.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("visual_similarity", "art_collection"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "eva_clip_bigE14": ModelSpec(
        key="eva_clip_bigE14",
        family="EVA-CLIP",
        label="EVA-CLIP · bigE/14",
        provider="openclip",
        model_id="EVA02-E-14",
        pretrained="laion2b_s4b_b115k",
        published="2023",
        description="EVA-02 CLIP Enormous (ViT-E/14) through OpenCLIP; highest-capacity EVA variant for the strongest retrieval.",
        requires=("torch", "open_clip"),
        recommended_batch_size=2,
        notes="Experimental and very heavy: multi-GB download and a large GPU are required. Intentionally not a default.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("visual_similarity", "art_collection"),
        hardware_tier="large_gpu",
        status="experimental",
    ),
    "cloc_roadmap": ModelSpec(
        key="cloc_roadmap",
        family="CLOC",
        label="CLOC · roadmap",
        provider="planned",
        model_id="",
        published="planned",
        description="Roadmap entry for future region-aware/local-alignment retrieval.",
        notes="Planned until a stable global embedding checkpoint and loader are available.",
        supports_image_embedding=False,
        supports_text_embedding=False,
        supports_text_search=False,
        supports_projection=False,
        recommended_for=("local_alignment", "art_collection"),
        hardware_tier="gpu_recommended",
        status="planned",
    ),
    "laclip_roadmap": ModelSpec(
        key="laclip_roadmap",
        family="LACLIP",
        label="LACLIP · roadmap",
        provider="planned",
        model_id="",
        published="planned",
        description="LaCLIP (CLIP trained with LLM-rewritten captions). Standard CLIP architecture, but the official weights ship as .pt checkpoints in the LijieFan/LaCLIP repo — not an OpenCLIP pretrained tag nor a Hugging Face Transformers repo.",
        notes="Planned: not pure-wiring. Needs a verified open_clip-compatible hf-hub mirror, or a checkpoint-file loader path, before it can be promoted to experimental.",
        supports_image_embedding=False,
        supports_text_embedding=False,
        supports_text_search=False,
        supports_projection=False,
        recommended_for=("local_alignment", "art_collection"),
        hardware_tier="gpu_recommended",
        status="planned",
    ),
    "qwen3_vl_embedding_2b": ModelSpec(
        key="qwen3_vl_embedding_2b",
        family="Qwen3-VL",
        label="Qwen3-VL Embedding · 2B",
        provider="sentence_transformers",
        model_id="Qwen/Qwen3-VL-Embedding-2B",
        published="2025",
        description="Qwen3-VL multimodal embedding model (2B). Encodes text, images and mixed inputs into a shared 2048-dim space via the Sentence-Transformers API.",
        requires=("torch", "transformers", "sentence_transformers", "qwen_vl_utils"),
        trust_remote_code=True,
        recommended_batch_size=4,
        notes="Experimental. To enable, double-click bootstrap/windows/install_optional_multimodal.bat (or bootstrap/macos/install_optional_multimodal.command) and restart — no terminal needed. Downloads the 2B checkpoint on first use. Large GPU recommended.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("general", "art_collection", "multilingual"),
        hardware_tier="large_gpu",
        status="experimental",
    ),
    "jina_v5_omni_small": ModelSpec(
        key="jina_v5_omni_small",
        family="Jina v5 Omni",
        label="Jina v5 Omni · Small",
        provider="sentence_transformers",
        model_id="jinaai/jina-embeddings-v5-omni-small",
        published="2025",
        description="Jina Embeddings v5 Omni (Small, ~1.74B). Multimodal text/image embeddings (1024-dim, Matryoshka) via the Sentence-Transformers API.",
        requires=("torch", "transformers", "sentence_transformers", "peft"),
        trust_remote_code=True,
        recommended_batch_size=8,
        notes="Experimental. To enable, double-click bootstrap/windows/install_optional_multimodal.bat (or bootstrap/macos/install_optional_multimodal.command) and restart — no terminal needed. Downloads the checkpoint on first use. License CC BY-NC 4.0 (non-commercial). GPU recommended.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("general", "art_collection", "multilingual"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
}


def list_model_specs() -> list[ModelSpec]:
    return list(MODEL_REGISTRY.values())


def public_model_registry() -> list[dict[str, Any]]:
    return [spec.to_public_dict() for spec in list_model_specs()]


def get_model_spec(model_key: str | None, legacy_model: str = "ViT-B-32", legacy_pretrained: str = "laion2b_s34b_b79k") -> ModelSpec:
    if model_key:
        try:
            return MODEL_REGISTRY[model_key]
        except KeyError as exc:
            known = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown embedding model '{model_key}'. Known models: {known}") from exc

    # Backward-compatible path for v4 clients that only send model/pretrained.
    if legacy_model == "ViT-B-32" and legacy_pretrained == "laion2b_s34b_b79k":
        return MODEL_REGISTRY["openclip_vit_b_32"]
    if legacy_model.startswith("hf-hub:apple/MobileCLIP"):
        return MODEL_REGISTRY["mobileclip_b_openclip"]

    return ModelSpec(
        key=f"legacy_openclip_{legacy_model}_{legacy_pretrained}".replace("/", "-").replace(":", "-"),
        family="OpenCLIP",
        label=f"OpenCLIP · {legacy_model} · {legacy_pretrained}",
        provider="openclip",
        model_id=legacy_model,
        pretrained=legacy_pretrained,
        published="custom",
        description="Legacy custom OpenCLIP model supplied by older UI/API clients.",
        requires=("torch", "open_clip"),
        recommended_batch_size=32,
    )
