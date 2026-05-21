from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..encoders.registry import MODEL_REGISTRY
from ..runtime_dirs import IMG_DIR, OUTPUT_DIR, ROOT


def safe_image_dir(image_dir: str) -> Path:
    requested = IMG_DIR.resolve() if image_dir == "img" else (ROOT / image_dir).resolve()
    root = IMG_DIR.resolve()
    if requested == root or root in requested.parents:
        return requested
    raise ValueError("Image directory must be img/ or a subfolder of img/.")


def list_embedding_indexes() -> list[dict[str, Any]]:
    emb_dir = OUTPUT_DIR / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    indexes = []
    for manifest_path in sorted(emb_dir.glob("*_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("complete") is False:
            continue
        spec = MODEL_REGISTRY.get(manifest.get("model_key", ""))
        embedding_path = _resolve_embedding_path(manifest)
        embedding_id = manifest_path.name.removesuffix("_manifest.json")
        indexes.append({
            "embedding_id": embedding_id,
            "model_key": manifest.get("model_key", ""),
            "model_label": manifest.get("model_label", ""),
            "image_dir": manifest.get("image_dir", "img"),
            "count": manifest.get("count", 0),
            "created_at": manifest.get("created_at"),
            "embedding_path": embedding_path.relative_to(ROOT).as_posix() if embedding_path and embedding_path.exists() else None,
            "supports_text_search": bool(spec and spec.supports_text_search),
            "available": bool(spec and spec.to_public_dict().get("available")),
            "status": spec.status if spec else "unknown",
        })
    return indexes


def find_embedding_index(model_key: str, image_dir: str, embedding_id: str | None = None) -> dict[str, Any] | None:
    indexes = list_embedding_indexes()
    if embedding_id:
        return next((idx for idx in indexes if idx["embedding_id"] == embedding_id), None)
    matches = [
        idx for idx in indexes
        if idx["model_key"] == model_key and _norm_dir(idx["image_dir"]) == _norm_dir(image_dir) and idx.get("embedding_path")
    ]
    return matches[0] if matches else None


def load_projection_lookup(model_key: str, image_dir: str) -> dict[str, dict[str, Any]]:
    proj_dir = OUTPUT_DIR / "projections"
    if not proj_dir.exists():
        return {}
    candidates = sorted(proj_dir.glob("*.tsv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        lookup: dict[str, dict[str, Any]] = {}
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    if row.get("model_key") != model_key:
                        continue
                    rel = row.get("relative_path", "")
                    if not rel.startswith(_norm_dir(image_dir).rstrip("/") + "/") and _norm_dir(image_dir) != "img":
                        continue
                    lookup[rel] = row
        except Exception:
            continue
        if lookup:
            return lookup
    return {}


def _resolve_embedding_path(manifest: dict[str, Any]) -> Path | None:
    raw_path = manifest.get("embedding_path")
    if raw_path:
        return (ROOT / raw_path).resolve()
    dataset_hash = manifest.get("dataset_hash")
    model_key = manifest.get("model_key")
    if not dataset_hash or not model_key:
        return None
    candidates = list((OUTPUT_DIR / "embeddings").glob(f"{dataset_hash}_{model_key}_*.npy"))
    return candidates[0].resolve() if candidates else None


def _norm_dir(value: str) -> str:
    return str(value).replace("\\", "/").strip("/")
