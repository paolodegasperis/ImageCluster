"""k-NN graph construction over embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from backend.neighbors import MAX_GRAPH_NODES, build_neighbor_graph


def _two_clusters(per_cluster: int = 10, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros(dim)
    a[0] = 1.0
    b = np.zeros(dim)
    b[1] = 1.0
    first = a + rng.normal(0, 0.01, (per_cluster, dim))
    second = b + rng.normal(0, 0.01, (per_cluster, dim))
    return np.vstack([first, second]).astype("float32")


def test_graph_needs_at_least_two_images():
    with pytest.raises(ValueError, match="At least two"):
        build_neighbor_graph(np.zeros((1, 4), dtype="float32"))


def test_graph_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="2D embedding matrix"):
        build_neighbor_graph(np.zeros(8, dtype="float32"))


def test_graph_refuses_oversized_collections():
    oversized = np.zeros((MAX_GRAPH_NODES + 1, 2), dtype="float32")
    with pytest.raises(ValueError, match="limited to"):
        build_neighbor_graph(oversized)


def test_k_is_capped_below_the_node_count():
    vectors = _two_clusters(per_cluster=2)
    graph = build_neighbor_graph(vectors, k=50)
    assert graph["k"] == vectors.shape[0] - 1


def test_no_self_edges_and_no_duplicate_pairs():
    graph = build_neighbor_graph(_two_clusters(), k=3)
    seen = set()
    for edge in graph["edges"]:
        assert edge["source"] != edge["target"]
        assert edge["source"] < edge["target"]
        key = (edge["source"], edge["target"])
        assert key not in seen
        seen.add(key)


def test_edges_connect_images_inside_the_same_cluster():
    per_cluster = 10
    graph = build_neighbor_graph(_two_clusters(per_cluster), k=3)
    crossing = [e for e in graph["edges"] if (e["source"] < per_cluster) != (e["target"] < per_cluster)]
    assert crossing == []


def test_two_separated_clusters_give_two_components():
    graph = build_neighbor_graph(_two_clusters(), k=3)
    assert graph["component_count"] == 2
    assert len(set(graph["components"])) == 2


def test_edges_are_sorted_by_descending_similarity():
    graph = build_neighbor_graph(_two_clusters(), k=4)
    similarities = [edge["similarity"] for edge in graph["edges"]]
    assert similarities == sorted(similarities, reverse=True)


def test_similarity_threshold_removes_weak_edges():
    vectors = _two_clusters()
    loose = build_neighbor_graph(vectors, k=4)
    strict = build_neighbor_graph(vectors, k=4, min_similarity=0.999)
    assert strict["edge_count"] <= loose["edge_count"]
    assert all(edge["similarity"] >= 0.999 for edge in strict["edges"])


def test_mutual_only_keeps_a_subset_of_the_edges():
    vectors = _two_clusters(per_cluster=12, seed=2)
    everything = build_neighbor_graph(vectors, k=2)
    mutual = build_neighbor_graph(vectors, k=2, mutual_only=True)
    assert mutual["mutual_only"] is True
    assert mutual["edge_count"] <= everything["edge_count"]
    assert all(edge["mutual"] for edge in mutual["edges"])


def test_degrees_match_the_kept_edges():
    graph = build_neighbor_graph(_two_clusters(), k=3)
    counted = [0] * graph["node_count"]
    for edge in graph["edges"]:
        counted[edge["source"]] += 1
        counted[edge["target"]] += 1
    assert counted == graph["degrees"]


def test_identical_vectors_do_not_break_ranking():
    vectors = np.ones((6, 4), dtype="float32")
    graph = build_neighbor_graph(vectors, k=2)
    assert graph["node_count"] == 6
    assert graph["edge_count"] > 0
    assert graph["component_count"] == 1


def test_chunked_similarity_matches_a_small_direct_computation():
    vectors = _two_clusters(per_cluster=25, seed=9)
    graph = build_neighbor_graph(vectors, k=1)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    expected_top = np.argmax(sims, axis=1)
    # Every node's single nearest neighbour must appear as one of its edges.
    incident = {index: set() for index in range(vectors.shape[0])}
    for edge in graph["edges"]:
        incident[edge["source"]].add(edge["target"])
        incident[edge["target"]].add(edge["source"])
    for index, best in enumerate(expected_top):
        assert int(best) in incident[index]


# --------------------------------------------------------------------------
# Structural reading: communities, bridges, articulation points


def _two_linked_groups(seed: int = 3):
    """Two tight groups joined through one image, so the graph is connected."""
    rng = np.random.default_rng(seed)
    dim = 32
    first = rng.normal(0, 1, dim)
    first /= np.linalg.norm(first)
    second = rng.normal(0, 1, dim)
    second /= np.linalg.norm(second)
    rows = [first + rng.normal(0, 0.05, dim) for _ in range(14)]
    rows += [second + rng.normal(0, 0.05, dim) for _ in range(14)]
    rows.append((first + second) / 2 + rng.normal(0, 0.01, dim))  # the linking work
    matrix = np.vstack(rows).astype("float32")
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_communities_divide_a_graph_that_components_cannot():
    """The point of the feature: one component, several communities."""
    graph = build_neighbor_graph(_two_linked_groups(), k=5)
    if not graph["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    assert graph["component_count"] == 1, "the planted graph should be connected"
    assert graph["community_count"] >= 2, "communities must still separate the two groups"
    assert graph["modularity"] > 0.2


def test_communities_do_not_mix_the_planted_groups():
    graph = build_neighbor_graph(_two_linked_groups(), k=4)
    if not graph["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    # Members 0-13 are group one, 14-27 group two, 28 is the link.
    for community in set(graph["communities"][:28]):
        members = [i for i in range(28) if graph["communities"][i] == community]
        groups = {i // 14 for i in members}
        assert len(groups) == 1, f"community {community} mixes both planted groups"


def test_every_node_gets_a_community_label():
    graph = build_neighbor_graph(_two_linked_groups(), k=4)
    if not graph["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    assert len(graph["communities"]) == graph["node_count"]
    assert min(graph["communities"]) == 0
    assert max(graph["communities"]) == graph["community_count"] - 1


def test_the_largest_community_is_numbered_first():
    """Stable numbering keeps the palette from reshuffling between builds."""
    graph = build_neighbor_graph(_two_linked_groups(), k=4)
    if not graph["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    sizes = [graph["communities"].count(label) for label in range(graph["community_count"])]
    assert sizes == sorted(sizes, reverse=True)


def test_communities_are_reproducible():
    vectors = _two_linked_groups()
    first = build_neighbor_graph(vectors, k=4)
    second = build_neighbor_graph(vectors, k=4)
    if not first["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    assert first["communities"] == second["communities"]
    assert first["modularity"] == second["modularity"]


def test_bridges_are_reported_and_flagged_on_their_edge():
    graph = build_neighbor_graph(_two_linked_groups(), k=3)
    if not graph["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    flagged = {tuple(sorted((e["source"], e["target"]))) for e in graph["edges"] if e["bridge"]}
    declared = {tuple(sorted(pair)) for pair in graph["bridges"]}
    assert flagged == declared


def test_every_edge_carries_the_bridge_flag():
    graph = build_neighbor_graph(_two_linked_groups(), k=4)
    assert all("bridge" in edge for edge in graph["edges"])
    assert all(isinstance(edge["bridge"], bool) for edge in graph["edges"])


def test_articulation_points_are_valid_node_indices():
    graph = build_neighbor_graph(_two_linked_groups(), k=4)
    if not graph["structure_available"]:
        pytest.skip("networkx is not installed in this environment")
    for index in graph["articulation_points"]:
        assert 0 <= index < graph["node_count"]
    assert graph["articulation_points"] == sorted(graph["articulation_points"])


def test_a_missing_networkx_degrades_without_breaking_the_graph(monkeypatch):
    """The structural reading is an enrichment, not the point of the view."""
    from backend import neighbors

    monkeypatch.setattr(neighbors, "_networkx", lambda: None)
    graph = build_neighbor_graph(_two_linked_groups(), k=4)
    assert graph["structure_available"] is False
    assert graph["communities"] is None
    assert graph["modularity"] is None
    assert graph["bridges"] == []
    assert graph["articulation_points"] == []
    # The graph itself is unaffected.
    assert graph["edge_count"] > 0
    assert all(edge["bridge"] is False for edge in graph["edges"])
