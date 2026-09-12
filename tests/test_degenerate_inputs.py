"""Degenerate numeric input across the view pipeline.

Every one of these came from an audit pass rather than from a bug report, and
each corresponds to a defect that was present: an infinite coordinate collapsed
the whole mosaic onto one axis value (and warned on stderr), and a single NaN
embedding vector silently reduced *other* images' neighbour lists because NaN
compares false and could be selected into anyone's top-k.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from backend.layouts import _normalize, grid_mosaic
from backend.neighbors import _sanitize, build_neighbor_graph

# --------------------------------------------------------------------------
# Coordinates


def test_normalize_is_silent_on_non_finite_input():
    coords = np.array([[0.0, 0.0], [np.inf, 1.0], [np.nan, 2.0], [1.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _normalize(coords)
    assert np.isfinite(out).all()


def test_normalize_does_not_let_an_infinity_collapse_the_finite_values():
    # Without finite-only bounds, span becomes inf and every real coordinate maps to 0.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [np.inf, 0.0]])
    out = _normalize(coords)
    finite_axis = out[:3, 0]
    assert len(set(np.round(finite_axis, 6))) == 3, "finite coordinates must stay distinct"
    assert out[3, 0] == pytest.approx(0.5), "the infinity belongs in the middle, not at an edge"


def test_normalize_centres_an_axis_with_no_finite_values():
    coords = np.array([[np.nan, 0.0], [np.inf, 1.0]])
    out = _normalize(coords)
    assert out[:, 0].tolist() == [0.5, 0.5]


def test_normalize_centres_a_constant_axis():
    coords = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
    out = _normalize(coords)
    assert out[:, 0].tolist() == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_mosaic_still_places_every_image_with_a_bad_coordinate(bad):
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [bad, 2.0], [3.0, bad], [4.0, 4.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = grid_mosaic(coords, algorithm="split")
    assert len(set(result["cells"])) == 5


# --------------------------------------------------------------------------
# Embedding vectors


def test_sanitize_zeroes_only_the_bad_rows():
    vectors = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, np.inf], [4.0, 5.0]], dtype="float32")
    clean, bad = _sanitize(vectors)
    assert bad == [1, 2]
    assert np.isfinite(clean).all()
    assert clean[0].tolist() == [1.0, 2.0]
    assert clean[3].tolist() == [4.0, 5.0]
    assert clean[1].tolist() == [0.0, 0.0]


def test_sanitize_does_not_mutate_the_caller_s_array():
    vectors = np.array([[np.nan, 1.0]], dtype="float32")
    _sanitize(vectors)
    assert np.isnan(vectors[0, 0])


def test_sanitize_reports_nothing_for_clean_input():
    clean, bad = _sanitize(np.ones((4, 3), dtype="float32"))
    assert bad == []
    assert clean.tolist() == np.ones((4, 3)).tolist()


def _two_groups(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros(8)
    a[0] = 1.0
    b = np.zeros(8)
    b[1] = 1.0
    return np.vstack([
        a + rng.normal(0, 0.01, (8, 8)),
        b + rng.normal(0, 0.01, (8, 8)),
    ]).astype("float32")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_corrupt_vector_is_reported_and_never_yields_a_bad_similarity(bad):
    vectors = _two_groups()
    vectors[5, :] = bad
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = build_neighbor_graph(vectors, k=2)
    assert graph["unusable_vectors"] == [5]
    assert all(np.isfinite(edge["similarity"]) for edge in graph["edges"])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_corrupt_vector_does_not_cost_other_images_their_neighbours(bad):
    """The original defect: NaN entered other nodes' top-k and was then dropped.

    Each node contributes k directed selections, so no node may be left without
    an edge, and the total must not fall below the clean graph's count.
    """
    clean_vectors = _two_groups()
    clean_graph = build_neighbor_graph(clean_vectors, k=2)

    poisoned = clean_vectors.copy()
    poisoned[5, :] = bad
    graph = build_neighbor_graph(poisoned, k=2)

    assert min(graph["degrees"]) >= 1
    assert graph["edge_count"] >= clean_graph["edge_count"] - 2


def test_a_corrupt_vector_is_not_treated_as_similar_to_anything():
    vectors = _two_groups()
    vectors[5, :] = np.nan
    graph = build_neighbor_graph(vectors, k=2)
    for edge in graph["edges"]:
        if 5 in (edge["source"], edge["target"]):
            assert edge["similarity"] == pytest.approx(0.0, abs=1e-6)


def test_an_all_zero_vector_behaves_like_a_sanitized_one():
    vectors = _two_groups()
    vectors[5, :] = 0.0
    graph = build_neighbor_graph(vectors, k=2)
    assert graph["unusable_vectors"] == []
    assert all(np.isfinite(edge["similarity"]) for edge in graph["edges"])


def test_every_vector_corrupt_still_produces_a_usable_graph():
    vectors = np.full((6, 4), np.nan, dtype="float32")
    graph = build_neighbor_graph(vectors, k=2)
    assert graph["unusable_vectors"] == [0, 1, 2, 3, 4, 5]
    assert graph["node_count"] == 6
    assert all(np.isfinite(edge["similarity"]) for edge in graph["edges"])
