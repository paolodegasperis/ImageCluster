from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from typing import Any

from .visual_features_encoder import FEATURE_NAMES


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    label: str
    provider: str
    model_id: str
    pretrained: str = ""
    published: str = ""
    # SPDX-style licence of the checkpoint actually downloaded, as the
    # Hugging Face repository declares it. Shown to the user: several of
    # these forbid commercial use, and a projection made in ignorance of
    # that is a problem the application handed its own user.
    license: str = ""
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
    # Name of the Sentence-Transformers prompt applied to a text query, for models
    # trained with a task prefix. EmbeddingGemma expects "task: search result |
    # query: " in front of a query; without it the model still answers, ranked and
    # plausible and worse, which is the failure mode this project has already been
    # caught by once. Empty for every model that takes a bare query.
    text_prompt_name: str = ""
    recommended_for: tuple[str, ...] = ()
    hardware_tier: str = "cpu_ok"
    status: str = "stable"
    # Selectable interpretable projection axes (used by the Visual Features
    # provider). Empty for ordinary embedding models.
    feature_axes: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requires"] = list(self.requires)
        data["recommended_for"] = list(self.recommended_for)
        data["feature_axes"] = list(self.feature_axes)
        missing = missing_requirements(self)
        data["available"] = not missing and self.status not in {"planned", "unavailable"}
        data["missing_requirements"] = missing
        return data


IMPORT_NAMES = {
    "torch": "torch",
    "open_clip": "open_clip",
    "timm": "timm",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "einops": "einops",
    "imagebind": "imagebind",
    "sentence_transformers": "sentence_transformers",
    "qwen_vl_utils": "qwen_vl_utils",
    "peft": "peft",
    "opencv-python": "cv2",
    "scikit-image": "skimage",
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
        license="mit",
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
        license="mit (upstream repository; undeclared on the Hub)",
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
        license="apache-2.0",
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
        license="apache-2.0",
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
    "visual_features_opencv": ModelSpec(
        key="visual_features_opencv",
        license="n/a - computed locally, no model weights",
        family="Visual Features",
        label="Visual Features · OpenCV",
        provider="visual_features",
        model_id="opencv_visual_features",
        pretrained="v1",
        published="2026",
        description="Clusters images by formal visual characteristics (HSV/Lab histograms, brightness, saturation, hue, contrast, edge density, texture, symmetry) computed with OpenCV. The projection uses fixed, interpretable axes; text search is not available.",
        requires=("opencv-python", "scikit-image"),
        recommended_batch_size=16,
        notes="No multimodal model is used. Pick which feature maps to the X and Y axis; the same image always lands in the same place. Text search is not supported.",
        supports_image_embedding=True,
        supports_text_embedding=False,
        supports_text_search=False,
        supports_projection=True,
        recommended_for=("formal_analysis", "visual_similarity", "art_collection"),
        hardware_tier="cpu_ok",
        feature_axes=tuple(FEATURE_NAMES),
    ),
    "mobileclip_b_openclip": ModelSpec(
        key="mobileclip_b_openclip",
        license="apple-amlr - research use only",
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
        license="apache-2.0",
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
        license="apache-2.0",
        family="Nomic Embed Vision",
        label="Nomic Embed Vision · v1.5",
        provider="nomic_transformers",
        model_id="nomic-ai/nomic-embed-vision-v1.5",
        published="2024-06-06",
        description="Vision embedding model aligned with Nomic text embeddings; requires remote model code from Hugging Face.",
        requires=("torch", "transformers", "einops"),
        trust_remote_code=True,
        recommended_batch_size=16,
        notes=(
            "UNAVAILABLE since v6.9.0, and the cause is upstream. The model runs remote code written for "
            "Transformers 4; under Transformers 5 it fails twice. Its own config.json declares n_inner as "
            "2048.0, a decimal where huggingface_hub 1.x now requires a whole number, and coercing that value "
            "only uncovers the next break: the remote modelling code reads config.layer_norm_eps, which that "
            "config spells layer_norm_epsilon. Repairing it would mean maintaining a patched copy of someone "
            "else's model code, which is not a promise this project can keep. It returns to the list if Nomic "
            "publishes a checkpoint that loads on Transformers 5."
        ),
        supports_text_embedding=False,
        supports_text_search=False,
        recommended_for=("visual_similarity",),
        hardware_tier="gpu_recommended",
        status="unavailable",
    ),
    "imagebind_huge": ModelSpec(
        key="imagebind_huge",
        license="see Meta's ImageBind repository - not verified here",
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
        license="cc-by-nc-4.0 - non-commercial",
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
        license="cc-by-nc-4.0 - non-commercial",
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
        license="cc-by-nc-4.0 - non-commercial",
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
        license="cc-by-nc-4.0 - non-commercial",
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
        license="cc-by-nc-4.0 - non-commercial",
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
        license="apple-amlr - research use only",
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
        license="apple-amlr - research use only",
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
        license="apple-amlr - research use only",
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
        license="apache-2.0",
        family="HQ-CLIP",
        label="HQ-CLIP · B/16",
        provider="openclip_hf_hub",
        model_id="hf-hub:zhixiangwei/vlm150m-hqclip-large-vitb16",
        published="2025-07-04",
        description="CLIP trained on VLM-150M, a corpus whose captions were rewritten and filtered for quality rather than merely scraped. Aimed at collections where the wording of a description carries the meaning.",
        requires=("torch", "open_clip"),
        recommended_batch_size=16,
        notes="Experimental: uses the author's own OpenCLIP-compatible Hugging Face checkpoint (Apache-2.0) and downloads weights on first use.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("art_collection", "general"),
        hardware_tier="cpu_ok",
        status="experimental",
    ),
    "embeddinggemma2": ModelSpec(
        key="embeddinggemma2",
        license="apache-2.0",
        family="EmbeddingGemma",
        label="EmbeddingGemma 2",
        provider="sentence_transformers",
        model_id="google/embeddinggemma-2",
        published="2026-09-04",
        description=(
            "Google's multimodal embedding model: text, images, video and audio in one shared 768-dimensional "
            "space, over 100 languages, and a context of 8192 tokens against CLIP's 77. The only model here that "
            "answers an Italian query as well as its English twin."
        ),
        requires=("torch", "sentence_transformers"),
        recommended_batch_size=8,
        text_prompt_name="query",
        notes=(
            "Verified on the 165-image reference collection: four solid colours 4/4, 768-d unit-norm embeddings, "
            "every image its own nearest neighbour. It is the one real cost here - ~107 ms/image on an RTX 5070 "
            "but ~2.2 s/image on CPU, some seventy times the B/32 default, so a CPU-only machine should treat it "
            "as an overnight job rather than a lens to switch between. Grounded retrieval ranked all four "
            "described pictures first - the only model here to do so - and ranked them identically when the same "
            "descriptions were written in Italian. Its 8192-token context was not shown to help on this "
            "collection: the short form of the query already won."
        ),
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("art_collection", "general"),
        hardware_tier="gpu_recommended",
        status="experimental",
    ),
    "cultureclip_b32": ModelSpec(
        key="cultureclip_b32",
        license="undeclared on the Hub (code Apache-2.0; the CulTwin training set is CC BY-NC 4.0, so treat the weights as non-commercial)",
        family="CultureCLIP",
        label="CultureCLIP · B/32",
        provider="transformers_clip",
        model_id="lukahh/cultureclip_sd_0303_32",
        published="2025-03-03",
        description="OpenAI CLIP B/32 fine-tuned on CulTwin, a synthetic set of near-identical images that carry different cultural meanings. Aimed at collections where the distinction between two similar objects is cultural rather than visual.",
        requires=("torch", "transformers"),
        recommended_batch_size=32,
        notes=(
            "Experimental, and the provenance is thinner than everything else here: "
            "the authors published no named release, only some thirty dated training "
            "runs, and this is the B/32 of 3 March 2025. Verified on the 165-image "
            "reference collection: the text tower is intact and grounded retrieval "
            "matches its own base CLIP exactly. As cheap to run as the default "
            "encoder - ~30 ms/image on CPU, ~11 ms on an RTX 5070, within a "
            "millisecond of OpenCLIP B/32 either way. It did not beat that base on "
            "culturally worded queries either, so choose it to get a second reading "
            "of a collection, not a better one."
        ),
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("art_collection", "general"),
        hardware_tier="cpu_ok",
        status="experimental",
    ),
    "pe_core_b16": ModelSpec(
        key="pe_core_b16",
        license="apache-2.0",
        family="Perception Encoder",
        label="PE-Core · B/16",
        provider="openclip_hf_hub",
        model_id="hf-hub:timm/PE-Core-B-16",
        published="2025-07-24",
        description="Meta's Perception Encoder, a vision-language family trained apart from the CLIP lineage. Its embedding is 1024-dimensional where most models here give 512, and it groups a collection differently rather than merely better.",
        requires=("torch", "open_clip", "timm"),
        recommended_batch_size=16,
        notes="Experimental: OpenCLIP-compatible checkpoint (Apache-2.0), downloaded on first use. Its text context is 32 tokens against CLIP's 77, so a long catalogue-style query is truncated; keep queries short and specific with this one.",
        supports_image_embedding=True,
        supports_text_embedding=True,
        supports_text_search=True,
        supports_projection=True,
        recommended_for=("art_collection", "visual_similarity"),
        hardware_tier="cpu_ok",
        status="experimental",
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
        license="mit",
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
        license="mit",
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
        license="none declared - one reason it is not integrated",
        family="LACLIP",
        label="LACLIP · roadmap",
        provider="planned",
        model_id="",
        published="planned",
        description="LaCLIP (CLIP trained with LLM-rewritten captions). Standard CLIP architecture, but the official weights ship as .pt checkpoints in the LijieFan/LaCLIP repo — not an OpenCLIP pretrained tag nor a Hugging Face Transformers repo.",
        notes="Not integrated, by decision rather than by obstacle. A community mirror of the ViT-B/16 weights appeared on Hugging Face in October 2025 and is structurally a valid CLIP, but it carries no licence, an empty model card and no way to prove the weights are the ones the paper describes. Provenance that thin does not belong in a citable tool.",
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
        license="apache-2.0",
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
        license="cc-by-nc-4.0 - non-commercial",
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
