"""Mosaic and neighbor-graph views over a real projection on disk.

Both views take a caller-supplied path, so the containment checks are exercised
here as well as in test_path_safety.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend.runtime_dirs import IMG_DIR, OUTPUT_DIR, ROOT
from backend.views import (
    mosaic_view,
    neighbor_graph_view,
    projection_sidecar,
    projection_supports_graph,
    validated_projection,
)

PROJECTION_NAME = "_pytest_views_projection.tsv"
EMBEDDING_NAME = "_pytest_views.npy"
PROJECTION_REL = f"output/projections/{PROJECTION_NAME}"


@pytest.fixture
def projection(request):
    """A 20-image projection in two clean groups, with a matching embedding file."""
    count = 20
    half = count // 2
    vectors = np.zeros((count, 6), dtype="float32")
    vectors[:half, 0] = 1.0
    vectors[half:, 1] = 1.0
    vectors += np.random.default_rng(11).normal(0, 0.005, vectors.shape).astype("float32")

    projections_dir = OUTPUT_DIR / "projections"
    embeddings_dir = OUTPUT_DIR / "embeddings"
    projections_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    embedding_path = embeddings_dir / EMBEDDING_NAME
    np.save(embedding_path, vectors)

    tsv_path = projections_dir / PROJECTION_NAME
    with tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "relative_path", "x", "y", "cluster"], delimiter="\t"
        )
        writer.writeheader()
        for index in range(count):
            writer.writerow({
                "filename": f"pic_{index:02d}.jpg",
                "relative_path": f"img/pic_{index:02d}.jpg",
                "x": float(vectors[index, 0] * 10 + index * 0.01),
                "y": float(vectors[index, 1] * 10 + index * 0.02),
                "cluster": 1 if index < half else 2,
            })

    sidecar = tsv_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "model_label": "Pytest model",
        "model_key": "pytest",
        "embedding_path": embedding_path.relative_to(ROOT).as_posix(),
        "image_count": count,
    }), encoding="utf-8")

    def cleanup():
        tsv_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)

    request.addfinalizer(cleanup)
    return {"rel": PROJECTION_REL, "count": count, "half": half, "tsv": tsv_path}


def test_validated_projection_rejects_outside_paths():
    with pytest.raises(ValueError, match="inside output/projections"):
        validated_projection("config.json")
    with pytest.raises(ValueError, match="inside output/projections"):
        validated_projection("output/embeddings")


def test_mosaic_places_every_image_in_a_unique_cell(projection):
    result = mosaic_view(projection["rel"])
    assert result["count"] == projection["count"]
    assert result["skipped"] == 0
    cells = {(cell["col"], cell["row"]) for cell in result["cells"]}
    assert len(cells) == projection["count"]
    assert result["columns"] * result["rows"] >= projection["count"]


def test_mosaic_cells_are_returned_in_reading_order(projection):
    cells = mosaic_view(projection["rel"])["cells"]
    keys = [(cell["row"], cell["col"]) for cell in cells]
    assert keys == sorted(keys)


def test_mosaic_honours_an_explicit_column_count(projection):
    result = mosaic_view(projection["rel"], columns=4)
    assert result["columns"] == 4
    assert max(cell["col"] for cell in result["cells"]) < 4


def test_mosaic_aspect_widens_the_grid(projection):
    wide = mosaic_view(projection["rel"], aspect=4.0)
    tall = mosaic_view(projection["rel"], aspect=0.25)
    assert wide["columns"] > tall["columns"]


def test_mosaic_keeps_similar_images_close_together(projection):
    """Locality, not axis-alignment, is what a mosaic has to preserve.

    The two groups sit at opposite corners of the projection, so neither a row
    band nor a column band separates them; what must hold is that an image lands
    nearer its own group than the other one.
    """
    result = mosaic_view(projection["rel"], columns=5)
    by_path = {cell["relative_path"]: cell for cell in result["cells"]}
    first = [by_path[f"img/pic_{i:02d}.jpg"] for i in range(projection["half"])]
    second = [by_path[f"img/pic_{i:02d}.jpg"] for i in range(projection["half"], projection["count"])]

    def mean_distance(a, b):
        pairs = [
            abs(x["col"] - y["col"]) + abs(x["row"] - y["row"])
            for x in a for y in b if x is not y
        ]
        return sum(pairs) / len(pairs)

    within = (mean_distance(first, first) + mean_distance(second, second)) / 2
    across = mean_distance(first, second)
    assert within < across


def test_mosaic_carries_cluster_labels_through(projection):
    cells = mosaic_view(projection["rel"])["cells"]
    assert {cell["cluster"] for cell in cells} == {1, 2}


def test_neighbor_graph_uses_the_embedding_file(projection):
    graph = neighbor_graph_view(projection["rel"], k=3)
    assert graph["node_count"] == projection["count"]
    assert graph["edge_count"] > 0
    assert graph["model_label"] == "Pytest model"
    # Two well-separated groups in embedding space give two components.
    assert graph["component_count"] == 2


def test_neighbor_graph_nodes_carry_position_and_degree(projection):
    graph = neighbor_graph_view(projection["rel"], k=2)
    for index, node in enumerate(graph["nodes"]):
        assert node["index"] == index
        assert isinstance(node["degree"], int)
        assert isinstance(node["component"], int)
        assert isinstance(node["x"], float)


def test_neighbor_graph_threshold_and_mutual_flags_apply(projection):
    loose = neighbor_graph_view(projection["rel"], k=4)
    strict = neighbor_graph_view(projection["rel"], k=4, min_similarity=0.999, mutual_only=True)
    assert strict["edge_count"] <= loose["edge_count"]
    assert strict["mutual_only"] is True


def test_neighbor_graph_reports_a_missing_embedding_file(projection):
    (OUTPUT_DIR / "embeddings" / EMBEDDING_NAME).unlink()
    with pytest.raises(ValueError, match="embedding file .* is missing"):
        neighbor_graph_view(projection["rel"])


def test_neighbor_graph_reports_a_row_count_mismatch(projection):
    np.save(OUTPUT_DIR / "embeddings" / EMBEDDING_NAME, np.zeros((3, 6), dtype="float32"))
    with pytest.raises(ValueError, match="Re-run the projection"):
        neighbor_graph_view(projection["rel"])


def test_neighbor_graph_requires_a_sidecar(projection):
    projection["tsv"].with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="no embedding sidecar"):
        neighbor_graph_view(projection["rel"])


def test_availability_probe_explains_a_missing_sidecar(projection):
    projection["tsv"].with_suffix(".json").unlink()
    info = projection_supports_graph(projection["rel"])
    assert info["ok"] is False
    assert "sidecar" in info["reason"]


def test_availability_probe_reports_a_usable_projection(projection):
    info = projection_supports_graph(projection["rel"])
    assert info["ok"] is True
    assert info["image_count"] == projection["count"]


def test_availability_probe_rejects_a_path_outside_projections():
    info = projection_supports_graph("config.json")
    assert info["ok"] is False


def test_sidecar_reader_tolerates_a_corrupt_file(projection):
    projection["tsv"].with_suffix(".json").write_text("{oops", encoding="utf-8")
    assert projection_sidecar(projection["tsv"]) is None
