from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .encoders.registry import public_model_registry
from .runtime_dirs import OUTPUT_DIR

LOCAL_SETTINGS_PATH = OUTPUT_DIR / "local_settings.json"
_LOCAL_HF_TOKEN_ENV_VALUE: str | None = None


def _read_local_settings() -> dict[str, Any]:
    if not LOCAL_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(LOCAL_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_local_settings(data: dict[str, Any]) -> None:
    LOCAL_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _mask_token(token: str) -> str:
    token = token.strip()
    if len(token) <= 8:
        return "••••"
    return f"{token[:4]}…{token[-4:]}"


def _external_hf_token() -> str:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""


def apply_runtime_settings() -> None:
    global _LOCAL_HF_TOKEN_ENV_VALUE

    token = str(_read_local_settings().get("hf_token") or "").strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    _LOCAL_HF_TOKEN_ENV_VALUE = token


def huggingface_token_status() -> dict[str, Any]:
    local_token = str(_read_local_settings().get("hf_token") or "").strip()
    env_token = _external_hf_token()
    token = local_token or env_token
    source = "local" if local_token else ("environment" if env_token else "none")
    return {
        "configured": bool(token),
        "source": source,
        "masked": _mask_token(token) if token else "",
        "settings_path": str(LOCAL_SETTINGS_PATH),
    }


def save_huggingface_token(token: str) -> dict[str, Any]:
    token = token.strip()
    if not token:
        raise ValueError("Hugging Face token is empty.")
    data = _read_local_settings()
    data["hf_token"] = token
    data["hf_token_updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_local_settings(data)
    apply_runtime_settings()
    return huggingface_token_status()


def delete_huggingface_token() -> dict[str, Any]:
    global _LOCAL_HF_TOKEN_ENV_VALUE

    data = _read_local_settings()
    old_token = str(data.pop("hf_token", "") or "").strip()
    data.pop("hf_token_updated_at", None)
    _write_local_settings(data)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if old_token and os.environ.get(name) == old_token:
            os.environ.pop(name, None)
    _LOCAL_HF_TOKEN_ENV_VALUE = None
    return huggingface_token_status()


def runtime_settings_public() -> dict[str, Any]:
    return {"huggingface": huggingface_token_status()}


MODEL_SELECTION_PROFILES = {
    "openclip_vit_b_32": "OpenCLIP ViT-B-32 trained on LAION-2B is a solid choice for an initial art-historical exploration of heterogeneous datasets. It produces visual-textual embeddings in a shared space, enabling both UMAP/t-SNE projections and semantic search through textual descriptions. It is suitable for mixed collections of painting, photography, graphics, museum reproductions, and digitized images, especially when the aim is to identify broad iconographic clusters such as portraits, landscapes, sacred scenes, interiors, architectures, and abstractions. Compared with more recent models, it is less refined in reading historical-formal details, but it is stable, fast, and predictable. For an art historian, it is the baseline model for orienting oneself within large archives before moving on to more specialized models.",
    "openai_clip_vit_b_32": "CLIP OpenAI ViT-B/32 is the historical reference point of the CLIP family: less up to date than models trained on later datasets, but still useful as a comparative baseline. Its distinctive feature lies in the balance between visual representation and linguistic association: it does not merely see shapes and colors, but tends to place the image within broad descriptive categories learned from the web. For art-historical datasets, it works well when the aim is to separate evident genres, recurring subjects, and general compositional configurations. It is less suitable for subtle distinctions between schools, hands, workshops, closely related chronologies, or rare iconographies. It should be chosen when a standard, readable, and replicable comparison is needed.",
    "siglip_base_patch16_224": "SigLIP Base patch16 224 is a visual-semantic model that replaces the classic CLIP contrastive loss with a sigmoid-based approach, often more efficient in constructing image-text associations. In practice, it may produce a more orderly embedding distribution, especially in datasets where differences depend not only on subject matter but also on recurring visual arrangements. For an art historian, it is useful on collections with fairly defined categories: miniatures, icons, portraits, prints, illustrations, and photographs of artworks. The 224 resolution limits the reading of micro-details, but favors speed and consistency. It differs from CLIP because it tends to produce a less noisy semantic space in generic text-based searches.",
    "siglip2_base_patch16_224": "SigLIP 2 Base patch16 224 is preferable when the dataset contains multilingual materials, comes from international museum contexts, or is associated with descriptions that are not exclusively in English. The SigLIP 2 family was developed with greater attention to visual and linguistic generalization; for art-historical analysis, it can be advantageous in archives containing European and non-European images, photographic reproductions, decorative objects, and applied arts materials. In embedding reduction, it tends to balance visual content and descriptive concepts more effectively, without depending solely on dominant Anglophone categories. It is suitable when semantic search must recognize varied subjects, materials, or cultural contexts.",
    "mobileclip_b_openclip": "MobileCLIP B is designed for computational efficiency. Its main utility is not to surpass larger models in interpretive refinement, but to enable rapid projections and searches on less powerful machines or large datasets. For an art historian, it is suitable for a preliminary phase: ordering thousands of images, checking the general structure of the archive, identifying duplicates, iconographic families, or macroscopic visual groups. Compared with standard CLIP/OpenCLIP, it may lose granularity in stylistic details, pictorial materials, and subtle compositional relations. It is advisable when the priority is fast exploration, not refined analysis of attributions, schools, or fine formal differences.",
    "dinov2_base": "DINOv2 Base is one of the most useful models for image-only analysis, because it does not primarily depend on text-image alignment. It produces visual embeddings based on form, structure, texture, layout, and internal image features. For an art historian, it is particularly interesting when the dataset must be organized according to formal similarities that are not necessarily easy to name: layout, figure posture, decorative density, chromatic relations, composition, ornamental patterns, and photographic rendering. It is less suitable for word-based search, but often more sensitive than CLIP to purely visual analogies. It should be chosen for stylistic comparisons, formal clustering, the study of series, recurring motifs, compositional variants, and iconographic sets that are not well described linguistically.",
    "nomic_embed_vision_v1_5": "Nomic Embed Vision v1.5 is a visual embedding model designed to share latent space with Nomic Embed Text v1.5. This characteristic makes it useful when a collection must be queried both visually and semantically, while maintaining coherence between images and textual descriptions. For art-historical archives, it is suitable for retrieval scenarios such as searching for rocky landscapes, seated female figures, floral ornament, or ecclesiastical interiors. Its distinctive feature is its explicit orientation toward building reusable embedding spaces for analysis, mapping, and search. Compared with CLIP, it may offer a more modern representation for multimodal exploration, especially on broad and mixed archives.",
    "imagebind_huge": "ImageBind Huge is useful when the image should not be considered in isolation, but as part of a broader multimodal environment. The ImageBind family was developed to bring different modalities into a common space: images, text, audio, depth, thermal data, or inertial data. For art-historical use centered only on images, it may be more complex than necessary, but it becomes interesting for audiovisual archives, installations, performance, exhibition documentation, video art, or materials in which image, sound, and medial context are relevant. In embedding reduction, it can capture broader associations than traditional CLIP models. It is less suitable for simple datasets of digitized paintings, and more appropriate for contemporary, media-based, and documentary corpora.",
    "llava_onevision_qwen2_05b_image_only": "LLaVA-OneVision Qwen2 0.5B, used in image-only mode, should be understood more as a generative multimodal model adapted to the extraction of visual representations than as a classic CLIP model. Its strength lies in its proximity to descriptive and reasoning capabilities: it can organize images according to recognizable elements, narrative configurations, and relevant objects, not only according to texture or composition. For an art historian, it is interesting on datasets where images contain complex scenes, iconographic attributes, figures, gestures, inscriptions, or narrative contexts. As a lightweight variant, it is suitable for local experimentation. It differs from CLIP models because it derives from a vision-language instruction logic, not from pure contrastive retrieval.",
    "qwen25_vl_3b_image_only": "Qwen2.5-VL 3B in image-only mode is an advanced choice for datasets in which simple visual similarity is not sufficient. The Qwen2.5-VL family is oriented toward structured visual understanding, object localization, and ordered output generation; this makes it promising for images rich in details, objects, inscriptions, complex layouts, illustrated plates, manuscripts, visual documents, and archival photographs. In embedding reduction, it may be useful for grouping images according to more complex semantic configurations and internal relations. Compared with DINOv2, it is less purely formal; compared with CLIP, it is more oriented toward scene understanding. It is suitable for advanced projections, not as a lightweight standard model.",
    "metaclip_b32": "MetaCLIP ViT-B/32 is a CLIP-like variant built with particular attention to the quality and curation of training data. For art-historical analysis, it can be selected when one wants a CLIP-like representation that is potentially more controlled in the relationship between image and text. It is suitable for general datasets of artworks, museum photographs, illustrations, and web images, especially for semantic search and projection. Compared with OpenCLIP LAION, it may be interesting when the aim is to reduce some of the noise effects typical of large unfiltered web datasets. The B/32 version remains relatively lightweight, making it a useful compromise between reliability, computational cost, and the ability to separate broad visual subjects and categories.",
    "metaclip_l14": "MetaCLIP ViT-L/14 represents a more robust and detailed choice than the B/32 variant. The finer patch size and larger architecture can improve sensitivity to compositional elements, iconographic details, and differences in visual rendering. For an art historian, it is preferable when the dataset is not too large and greater precision is needed in comparing artworks: portraits with similar poses, variants of the same subject, related pictorial schools, prints, photographs of objects, or architectural details. It requires more resources, but can produce more articulated UMAP/t-SNE maps. It differs from the B/32 version through its greater visual descriptive capacity, while remaining within the CLIP-like image-text paradigm.",
    "metaclip2_worldwide_b32": "MetaCLIP 2 Worldwide B/32 is particularly interesting for art historians working on collections that are not exclusively Western, or on archives with multilingual metadata and concepts. MetaCLIP 2 was developed as a worldwide scaling recipe, with attention to the coexistence of English and non-English data. In practice, this can reduce dependence on Anglocentric visual and linguistic categories. The B/32 variant is the most manageable entry point: useful for semantic search, clustering, and projection of global, ethnographic, museum-based, or comparative datasets. Compared with classic MetaCLIP, its distinctive feature is greater intercultural openness; compared with larger versions, it maintains a more accessible computational profile.",
    "metaclip2_worldwide_h14": "MetaCLIP 2 Worldwide H/14 is a high-end choice for complex, global, and visually rich archives. It combines the worldwide logic of MetaCLIP 2 with a more powerful and finer configuration, suitable for capturing internal details and differences between culturally diverse images. For an art historian, it may be useful in large-scale comparative projects: European and non-European art, colonial photography, Islamic, Asian, African, pre-Columbian art, decorative collections, or museum objects with strong variety in forms and materials. It is less suitable for rapid CPU use. Its main difference from the B/32 variant is greater discrimination capacity; compared with traditional CLIP models, it offers broader linguistic-cultural coverage.",
    "metaclip2_2b_worldwide": "MetaCLIP 2 2B Worldwide should be considered a high-capacity model for very large and culturally heterogeneous datasets. It is suitable when the objective is not only to produce an exploratory map, but to build a rich embedding space for comparative analysis, multilingual retrieval, and the study of large museum or photographic archives. For an art historian, it may be relevant in institutional projects, extended digital collections, or multimodal benchmarks. Its distinctive feature is scale: it should offer greater robustness on rare subjects, non-Western visual traditions, and categories less represented in the original CLIP models. However, it requires substantial resources and should be chosen only when the quality of the latent space justifies greater computational time and cost.",
    "mobileclip2_s2": "MobileCLIP2 S2 is a lightweight-oriented solution. For art-historical use, it should be understood as a screening model: useful for quickly checking the structure of a dataset, identifying broad groups, separating main genres, or testing the interface on less powerful computers. It is not the priority choice for distinguishing schools, artists, or subtle stylistic transitions, but it may be functional in teaching activities, preliminary cataloguing, or very large archives where speed is more important than maximum accuracy. Compared with MobileCLIP B, it may be even more suitable for lightweight scenarios. Its difference from larger models lies in reduced computational cost, with a possible loss of semantic and formal refinement.",
    "mobileclip2_b": "MobileCLIP2 B is a compromise between efficiency and capacity. Compared with S2, it should offer more robust embeddings while preserving a lightweight/mobile logic. For an art historian, it is suitable when a reasonably reliable projection is needed without relying on heavy models such as MetaCLIP 2 H/14 or Qwen2.5-VL. It can work well on medium-to-large datasets, with recognizable subjects and visual categories that are not too subtle: figurative painting, photography, posters, illustration, and graphic materials. It is less suitable for attribution or minute stylistic comparisons. It differs from OpenCLIP because it prioritizes efficiency and practicality; compared with MobileCLIP2 S2, it offers a better balance for exploratory analysis with some semantic requirements.",
    "mobileclip2_s4": "MobileCLIP2 S4 can be understood as a lightweight but more capable variant than the minimal configurations. It is suitable for users who want to work on large datasets without entirely sacrificing representation quality. In the art-historical field, it can be selected for photographic archives, collections of digitized images, teaching datasets, or initial mappings of museum collections. Its main utility is operational: it enables rapid iterations, comparisons between UMAP and t-SNE, clustering tests, and semantic search without overloading the machine. It does not replace deeper models for complex iconographic analysis, but it allows an initial geography of the corpus to be produced. It sits between MobileCLIP2 S2 and B as an efficient but not minimal option.",
}


def model_guide() -> dict[str, Any]:
    cards = []
    for model in public_model_registry():
        capabilities = []
        if model["supports_projection"]:
            capabilities.append("UMAP/t-SNE projection")
        if model["supports_text_search"]:
            capabilities.append("semantic search")
        elif model["supports_image_embedding"]:
            capabilities.append("image-only visual embedding")
        if model["hardware_tier"] == "cpu_ok":
            hardware = "CPU-friendly"
        elif model["hardware_tier"] == "gpu_recommended":
            hardware = "GPU recommended"
        else:
            hardware = "Large GPU / high memory"
        if model["status"] == "planned":
            limitation = "Roadmap entry: not available for local projection yet."
        elif not model["supports_text_search"]:
            limitation = "Does not support text queries or semantic search in this app."
        elif model["status"] == "experimental":
            limitation = "Experimental: first use may download weights and behavior can vary by local environment."
        else:
            limitation = "Stable option for the standard workflow."
        cards.append({
            "key": model["key"],
            "family": model["family"],
            "label": model["label"],
            "status": model["status"],
            "hardware_tier": model["hardware_tier"],
            "hardware_label": hardware,
            "available": model["available"],
            "capabilities": capabilities,
            "recommended_for": model["recommended_for"],
            "description": model["description"] or model["family"],
            "profile": MODEL_SELECTION_PROFILES.get(model["key"], model["description"] or model["family"]),
            "notes": model["notes"],
            "limitation": limitation,
        })
    return {"models": cards}
