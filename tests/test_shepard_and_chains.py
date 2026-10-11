"""Map honesty and the chain between two images.

Both read a planted projection whose answer is known in advance, because a
measurement that cannot recognise a distortion it was handed is not a
measurement.

Two layouts are planted over the same embeddings:

*faithful*
    the map coordinates are the first two dimensions of the vectors themselves,
    so the map is as close to the space as a 2D map can be. Stress must come out
    low and the rank correlation high.

*shuffled*
    the same coordinates, randomly reassigned to other images. The map then says
    nothing about the space, and stress must come out far worse. A metric that
    cannot separate these two has no business being shown to a user.

The chain is planted on a gradient rather than on clusters: sixty images whose
centre slides steadily along one direction, the way a collection spanning a
period or a palette actually looks. Neighbours are then the works either side,
the graph is connected, and a route from one end to the other has to cross the
middle - which is the claim the strip makes on screen.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend.chains import similarity_chain
from backend.runtime_dirs import OUTPUT_DIR
from backend.shepard import shepard_diagram

NAME = "_pytest_shepard"
REL = f"output/projections/{NAME}.tsv"
DIM = 32
PER_CLUSTER = 20
CLUSTERS = 3


def _write(meta, matrix, coords):
    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "projections").mkdir(parents=True, exist_ok=True)
    embedding = OUTPUT_DIR / "embeddings" / f"{NAME}.npy"
    np.save(embedding, matrix)
    tsv = OUTPUT_DIR / "projections" / f"{NAME}.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "relative_path", "x", "y", "cluster"], delimiter="\t")
        writer.writeheader()
        for index, (filename, cluster) in enumerate(meta):
            writer.writerow({
                "filename": filename,
                "relative_path": f"img/{filename}",
                "x": float(coords[index, 0]),
                "y": float(coords[index, 1]),
                "cluster": cluster,
            })
    tsv.with_suffix(".json").write_text(json.dumps({
        "model_label": "Pytest model",
        "reducer": "planted",
        "embedding_path": embedding.relative_to(OUTPUT_DIR.parent).as_posix(),
    }), encoding="utf-8")
    return tsv, tsv.with_suffix(".json"), embedding


@pytest.fixture
def planted(request):
    """A gradient of images, laid out faithfully and then shuffled."""
    rng = np.random.default_rng(11)
    vectors: list[np.ndarray] = []
    meta: list[tuple[str, int]] = []

    base = rng.normal(0, 1, DIM)
    base /= np.linalg.norm(base)
    step = rng.normal(0, 1, DIM)
    step -= step @ base * base
    step /= np.linalg.norm(step)

    total = PER_CLUSTER * CLUSTERS
    for index in range(total):
        position = index / (total - 1) * 1.8
        centre = base + step * position
        centre /= np.linalg.norm(centre)
        vectors.append(centre + rng.normal(0, 0.03, DIM))
        # Thirds of the gradient, named so a route can be checked for crossing
        # the middle one.
        third = min(CLUSTERS - 1, index * CLUSTERS // total)
        meta.append((f"c{third}_{index:02d}.jpg", third + 1))

    matrix = np.vstack(vectors).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    faithful = matrix[:, :2].astype("float64") * 10.0
    shuffled = faithful[rng.permutation(len(matrix))]

    state: dict = {}

    def build(coords):
        for path in state.get("paths", ()):
            path.unlink(missing_ok=True)
        state["paths"] = _write(meta, matrix, coords)
        return REL

    request.addfinalizer(lambda: [path.unlink(missing_ok=True) for path in state.get("paths", ())])
    return {
        "build": build,
        "faithful": faithful,
        "shuffled": shuffled,
        "total": len(meta),
        "first": f"img/{meta[0][0]}",
        "last": f"img/{meta[-1][0]}",
    }


# --------------------------------------------------------------------------
# The Shepard diagram


def test_a_faithful_layout_scores_better_than_a_shuffled_one(planted):
    faithful = shepard_diagram(planted["build"](planted["faithful"]))
    shuffled = shepard_diagram(planted["build"](planted["shuffled"]))

    assert faithful["stress"] < shuffled["stress"]
    assert faithful["rank_correlation"] > shuffled["rank_correlation"]
    # The shuffled map carries no information about the space at all, so its
    # rank correlation must sit near zero rather than merely lower.
    assert abs(shuffled["rank_correlation"]) < 0.25


def test_the_grid_holds_every_measured_pair(planted):
    result = shepard_diagram(planted["build"](planted["faithful"]))
    grid = np.array(result["grid"])
    assert grid.shape == (result["grid_size"], result["grid_size"])
    assert int(grid.sum()) == result["pairs"]
    # Small collection: every pair is compared, nothing is sampled away.
    assert result["exhaustive"] is True
    assert result["pairs"] == planted["total"] * (planted["total"] - 1) // 2


def test_the_named_pairs_point_in_the_directions_they_claim(planted):
    result = shepard_diagram(planted["build"](planted["shuffled"]))
    # "Compressed" means far apart in the embedding but close on the map, so its
    # residual must be positive, and "stretched" the mirror image.
    assert all(pair["residual"] > 0 for pair in result["compressed"])
    assert all(pair["residual"] < 0 for pair in result["stretched"])
    assert result["compressed"][0]["residual"] > result["stretched"][0]["residual"]


def test_sampling_kicks_in_without_changing_the_shape(planted):
    rel = planted["build"](planted["faithful"])
    full = shepard_diagram(rel)
    sampled = shepard_diagram(rel, sample_pairs=1_000)
    assert sampled["exhaustive"] is False
    assert sampled["pairs"] < full["pairs"]
    # A sample that changed the verdict would be a sample not worth taking.
    assert abs(sampled["stress"] - full["stress"]) < 0.05


def test_a_missing_projection_is_refused(planted):
    planted["build"](planted["faithful"])
    with pytest.raises(ValueError):
        shepard_diagram("output/projections/_pytest_missing.tsv")


# --------------------------------------------------------------------------
# The chain


def test_the_route_passes_through_the_middle_cluster(planted):
    rel = planted["build"](planted["faithful"])
    result = similarity_chain(rel, from_path=planted["first"], to_path=planted["last"], k=4)
    assert result["connected"] is True
    names = [step["filename"] for step in result["steps"]]
    assert f"img/{names[0]}" == planted["first"] and f"img/{names[-1]}" == planted["last"]
    # The gradient is a line, so there is no honest shortcut past its middle.
    assert any(name.startswith("c1_") for name in names), names


def test_every_link_is_a_real_neighbour_similarity(planted):
    rel = planted["build"](planted["faithful"])
    result = similarity_chain(rel, from_path=planted["first"], to_path=planted["last"], k=4)
    assert len(result["links"]) == len(result["steps"]) - 1 == result["hops"]
    assert all(-1.0 <= value <= 1.0 for value in result["links"])
    assert result["weakest_link"] == min(result["links"])
    # Each step is a neighbour hop, so every link must beat the similarity of the
    # endpoints themselves - otherwise the chain is not buying anything.
    assert result["weakest_link"] > result["direct_similarity"]


def test_a_bigger_k_cannot_make_the_route_longer(planted):
    rel = planted["build"](planted["faithful"])
    tight = similarity_chain(rel, from_path=planted["first"], to_path=planted["last"], k=3)
    loose = similarity_chain(rel, from_path=planted["first"], to_path=planted["last"], k=12)
    assert loose["hops"] <= tight["hops"]


def test_an_unreachable_pair_explains_itself(planted):
    """At k=1 the graph falls apart, and the answer has to say so rather than
    return an empty list that reads like 'these images are unrelated'."""
    rel = planted["build"](planted["faithful"])
    result = similarity_chain(rel, from_path=planted["first"], to_path=planted["last"], k=1)
    if result["connected"]:
        pytest.skip("k=1 still connected this fixture; nothing to assert")
    assert result["steps"] == []
    assert "k=1" in result["reason"]
    assert len(result["component_sizes"]) == 2


def test_an_image_cannot_be_chained_to_itself(planted):
    rel = planted["build"](planted["faithful"])
    with pytest.raises(ValueError, match="two different images"):
        similarity_chain(rel, from_path=planted["first"], to_path=planted["first"])


def test_an_unknown_image_is_named_in_the_error(planted):
    rel = planted["build"](planted["faithful"])
    with pytest.raises(ValueError, match="not part of this projection"):
        similarity_chain(rel, from_path="img/nope.jpg", to_path=planted["last"])
