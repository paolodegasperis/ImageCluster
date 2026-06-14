from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..encoders.base import encode_texts
from ..encoders.registry import get_model_spec
from ..image_scan import scan_images
from ..runtime_dirs import OUTPUT_DIR, ROOT
from .index_store import find_embedding_index, list_embedding_indexes, load_projection_lookup, safe_image_dir


def list_search_indexes() -> list[dict[str, Any]]:
    return [idx for idx in list_embedding_indexes() if idx.get("supports_text_search")]


def run_text_search(
    query: str,
    model_key: str,
    image_dir: str,
    embedding_id: str | None,
    top_k: int,
    threshold: float | None,
    normalize: bool = True,
) -> dict[str, Any]:
    started = time.strftime("%Y%m%d-%H%M%S")
    log_context: dict[str, Any] = {
        "timestamp": started,
        "model_key": model_key,
        "query": query,
        "top_k": top_k,
        "threshold": threshold,
        "embedding_id": embedding_id,
        "image_dir": image_dir,
    }
    try:
        spec = get_model_spec(model_key)
        if not spec.supports_text_search:
            raise RuntimeError("This model can create image projections but does not support text search.")
        if not spec.to_public_dict().get("available"):
            missing = ", ".join(spec.to_public_dict().get("missing_requirements", []))
            reason = f" Missing requirements: {missing}." if missing else f" Status: {spec.status}."
            raise RuntimeError(f"This model is not available for local text search.{reason}")

        index = find_embedding_index(model_key, image_dir, embedding_id)
        if not index or not index.get("embedding_path"):
            raise RuntimeError("No searchable embeddings were found. Generate embeddings first or build a searchable index.")

        image_root = safe_image_dir(index["image_dir"])
        image_paths = scan_images(image_root)
        embeddings = np.load(ROOT / index["embedding_path"]).astype("float32")
        if len(image_paths) != embeddings.shape[0]:
            raise RuntimeError(
                f"Embedding count mismatch: manifest images={len(image_paths)}, vectors={embeddings.shape[0]}. "
                "Regenerate embeddings for this folder."
            )

        image_vectors = _l2_normalize(embeddings) if normalize else embeddings
        text_vector = encode_texts([query], spec, batch_size=1)[0].astype("float32")
        if normalize:
            text_vector = _l2_normalize(text_vector.reshape(1, -1))[0]
        scores = image_vectors @ text_vector

        order = np.argsort(-scores)
        if threshold is not None:
            order = np.array([idx for idx in order if scores[idx] >= threshold], dtype=int)
        order = order[:top_k]

        projection_lookup = load_projection_lookup(model_key, index["image_dir"])
        results = []
        for rank, idx in enumerate(order, start=1):
            path = image_paths[int(idx)]
            relative_path = path.relative_to(ROOT).as_posix()
            projection = projection_lookup.get(relative_path, {})
            results.append({
                "rank": rank,
                "filename": path.name,
                "relative_path": relative_path,
                "score": round(float(scores[int(idx)]), 6),
                "x": _float_or_none(projection.get("x")),
                "y": _float_or_none(projection.get("y")),
                "cluster": _int_or_none(projection.get("cluster")),
                "metadata": {},
            })

        payload = {
            "query": query,
            "model_key": model_key,
            "embedding_id": index["embedding_id"],
            "results": results,
        }
        _write_search_outputs(started, model_key, payload)
        log_context["embedding_id"] = index["embedding_id"]
        log_context["number_of_images_searched"] = int(embeddings.shape[0])
        _write_search_debug(started, log_context, None)
        return payload
    except Exception as exc:
        _write_search_debug(started, log_context, exc)
        raise


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _write_search_outputs(timestamp: str, model_key: str, payload: dict[str, Any]) -> None:
    search_dir = OUTPUT_DIR / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in model_key)
    json_path = search_dir / f"{timestamp}_{safe_model}_search.json"
    tsv_path = search_dir / f"{timestamp}_{safe_model}_search.tsv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["rank", "filename", "relative_path", "score", "query", "model_key", "embedding_id", "x", "y", "cluster"]
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({
                "rank": row["rank"],
                "filename": row["filename"],
                "relative_path": row["relative_path"],
                "score": row["score"],
                "query": payload["query"],
                "model_key": payload["model_key"],
                "embedding_id": payload["embedding_id"],
                "x": row.get("x"),
                "y": row.get("y"),
                "cluster": row.get("cluster"),
            })


def _write_search_debug(timestamp: str, context: dict[str, Any], exc: Exception | None) -> None:
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    debug = dict(context)
    if exc is not None:
        debug["exception_type"] = type(exc).__name__
        debug["exception_message"] = str(exc)
        debug["traceback"] = traceback.format_exc()
        (logs_dir / f"search_{timestamp}.log").write_text(debug["traceback"], encoding="utf-8")
    else:
        debug["status"] = "ok"
    (logs_dir / f"search_{timestamp}.debug.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
