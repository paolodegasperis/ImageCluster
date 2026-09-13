"""Alternative readings of a saved projection: grid mosaic and neighbor graph.

Both views start from a projection TSV already on disk, so switching view never
re-encodes anything. The mosaic needs only the 2D coordinates; the neighbor
graph needs the original embedding vectors, which it locates through the
projection's JSON sidecar.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cache import manifest_paths, read_manifest
from .layouts import grid_mosaic
from .neighbors import build_neighbor_graph
from .runtime_dirs import OUTPUT_DIR, ROOT


def validated_projection(projection_path: str) -> Path:
    """Resolve a caller-supplied projection path, refusing anything outside output/projections/."""
    requested = (ROOT / projection_path).resolve()
    proj_root = (OUTPUT_DIR / "projections").resolve()
    if requested == proj_root or proj_root not in requested.parents:
        raise ValueError("Projection path must be inside output/projections/.")
    if not requested.is_file():
        raise ValueError("Projection file not found.")
    return requested


def read_projection_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError("The projection file is empty.")
    return rows


def projection_sidecar(path: Path) -> dict[str, Any] | None:
    """The JSON written next to a projection TSV, holding the embedding pointer."""
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _coordinates(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[int]]:
    """Parse x/y, dropping rows whose coordinates are unusable."""
    coords: list[tuple[float, float]] = []
    kept: list[int] = []
    for index, row in enumerate(rows):
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        coords.append((x, y))
        kept.append(index)
    if len(coords) < 1:
        raise ValueError("The projection contains no usable coordinates.")
    return np.array(coords, dtype="float64"), kept


def _node_payload(row: dict[str, str], x: float, y: float) -> dict[str, Any]:
    def optional_number(key: str):
        raw = row.get(key)
        if raw in (None, "", "null"):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    cluster = optional_number("cluster")
    return {
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "x": x,
        "y": y,
        "cluster": int(cluster) if cluster is not None else None,
    }


def mosaic_view(
    projection_path: str,
    *,
    columns: int | None = None,
    aspect: float = 1.0,
    algorithm: str = "auto",
) -> dict[str, Any]:
    """Lay a projection out as a dense grid where every image gets one equal cell."""
    path = validated_projection(projection_path)
    rows = read_projection_rows(path)
    coords, kept = _coordinates(rows)

    layout = grid_mosaic(coords, columns=columns, aspect=aspect, algorithm=algorithm)

    cells = []
    for position, row_index in enumerate(kept):
        row = rows[row_index]
        col, grid_row = layout["cells"][position]
        node = _node_payload(row, float(coords[position][0]), float(coords[position][1]))
        node.update({"col": col, "row": grid_row})
        cells.append(node)

    # Reading order makes the payload easy to render and to eyeball in a debug dump.
    cells.sort(key=lambda cell: (cell["row"], cell["col"]))

    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "columns": layout["columns"],
        "rows": layout["rows"],
        "algorithm": layout["algorithm"],
        "note": layout["note"],
        "count": len(cells),
        "skipped": len(rows) - len(cells),
        "cells": cells,
    }


def embeddings_for_projection(path: Path, rows: list[dict[str, str]]) -> np.ndarray:
    """Load the embedding matrix a projection was built from, aligned to its rows."""
    sidecar = projection_sidecar(path)
    if not sidecar or not sidecar.get("embedding_path"):
        raise ValueError(
            "This projection has no embedding sidecar, so its neighbor graph cannot be built. "
            "Re-run the projection to regenerate it."
        )

    embedding_file = (ROOT / sidecar["embedding_path"]).resolve()
    embeddings_root = (OUTPUT_DIR / "embeddings").resolve()
    if embeddings_root not in embedding_file.parents:
        raise ValueError("The projection points at an embedding file outside output/embeddings/.")
    if not embedding_file.is_file():
        raise ValueError(
            "The embedding file for this projection is missing. "
            "Re-run the projection with the cache enabled to recreate it."
        )

    vectors = np.load(embedding_file)
    if vectors.shape[0] != len(rows):
        raise ValueError(
            f"The projection has {len(rows)} rows but its embedding file holds {vectors.shape[0]} vectors. "
            "Re-run the projection so the two match."
        )
    return vectors


def load_projection(projection_path: str) -> tuple[Path, list[dict[str, str]], np.ndarray]:
    """Resolve a projection and load its rows and embedding matrix together.

    Shared by every read-only view: the containment check, the TSV parse and the
    row/vector alignment check are identical for all of them, and duplicating
    them is how the alignment guarantee would drift.
    """
    path = validated_projection(projection_path)
    rows = read_projection_rows(path)
    vectors = embeddings_for_projection(path, rows)
    return path, rows, vectors


def neighbor_graph_view(
    projection_path: str,
    *,
    k: int = 4,
    min_similarity: float | None = None,
    mutual_only: bool = False,
) -> dict[str, Any]:
    """Build the k-NN graph of a projection's images, positioned by the projection."""
    path, rows, vectors = load_projection(projection_path)

    graph = build_neighbor_graph(vectors, k, min_similarity=min_similarity, mutual_only=mutual_only)
    articulation = set(graph["articulation_points"])

    nodes = []
    for index, row in enumerate(rows):
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
        except (TypeError, ValueError):
            x, y = 0.0, 0.0
        node = _node_payload(row, x, y)
        node.update({
            "index": index,
            "degree": graph["degrees"][index],
            "component": graph["components"][index],
            "community": graph["communities"][index] if graph["communities"] else None,
            "articulation": index in articulation,
        })
        nodes.append(node)

    sidecar = projection_sidecar(path) or {}
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "model_label": sidecar.get("model_label", ""),
        "model_key": sidecar.get("model_key", ""),
        "k": graph["k"],
        "node_count": graph["node_count"],
        "edge_count": graph["edge_count"],
        "component_count": graph["component_count"],
        "mutual_only": graph["mutual_only"],
        "min_similarity": graph["min_similarity"],
        "community_count": graph["community_count"],
        "modularity": graph["modularity"],
        "bridges": graph["bridges"],
        "articulation_points": graph["articulation_points"],
        "structure_available": graph["structure_available"],
        "structure_reason": graph.get("structure_reason"),
        "nodes": nodes,
        "edges": graph["edges"],
    }


def projection_supports_graph(projection_path: str) -> dict[str, Any]:
    """Cheap pre-flight so the UI can explain why a graph is unavailable."""
    try:
        path = validated_projection(projection_path)
    except ValueError as exc:
        return {"ok": False, "reason": str(exc)}
    sidecar = projection_sidecar(path)
    if not sidecar or not sidecar.get("embedding_path"):
        return {"ok": False, "reason": "This projection has no embedding sidecar. Re-run it to enable the graph view."}
    if not (ROOT / sidecar["embedding_path"]).is_file():
        return {"ok": False, "reason": "The embedding file for this projection is missing."}
    manifest = read_manifest((ROOT / sidecar["embedding_path"]).with_name(
        (ROOT / sidecar["embedding_path"]).stem + "_manifest.json"
    ))
    return {
        "ok": True,
        "model_label": sidecar.get("model_label", ""),
        "image_count": sidecar.get("image_count"),
        "has_manifest_paths": manifest_paths(manifest) is not None,
    }
