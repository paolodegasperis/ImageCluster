"""Nearest-neighbour graph over image embeddings.

The scatter plot answers "where does this image sit in the overall space"; it
cannot answer "which images is this one actually closest to", because a 2D
projection distorts distances and two images can land near each other by
accident. The k-NN graph is built in the full embedding space instead, so an
edge means genuine similarity.

Edges are undirected and de-duplicated: if A lists B among its neighbours and B
lists A, that is one edge, flagged as mutual. Mutual edges are the reliable
ones — a one-directional edge often means A is an outlier that had to pick
someone.

On top of the edges, three structural readings are computed:

*Communities* (Louvain) are groups of images that link to each other far more
than to the rest. Unlike connected components, which only separate images with
no path between them at all, communities divide a graph that is fully connected —
which is the usual case. This is the graph's own answer to "how does this
collection divide up", arrived at without the 2D projection and without K-Means.

*Bridges* are edges whose removal would split the graph. A bridge is the single
link holding two groups together: in a collection it is the work that connects
two otherwise separate visual worlds.

*Articulation points* are the images whose removal would split the graph — the
same idea applied to nodes rather than links.
"""

from __future__ import annotations

import numpy as np

MAX_GRAPH_NODES = 5000
MAX_K = 20

# Louvain is randomised; a fixed seed keeps the same graph giving the same
# communities across reloads, which matters because the colours would otherwise
# shuffle every time the user changed an unrelated setting.
LOUVAIN_SEED = 42


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _sanitize(vectors: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Zero out rows holding NaN or infinity, and report which ones.

    A single corrupt vector does more damage than losing its own edges: NaN
    compares false against everything, so ``argpartition`` can select it into any
    other node's top-k and push a genuine neighbour out of consideration
    entirely. Zeroing the row makes its similarity 0 to everything, which never
    outranks a real neighbour, and keeps the row indices aligned with the
    projection.
    """
    clean = np.asarray(vectors, dtype="float32").copy()
    bad_mask = ~np.isfinite(clean).all(axis=1)
    bad = [int(index) for index in np.flatnonzero(bad_mask)]
    if bad:
        clean[bad_mask] = 0.0
    return clean, bad


def _cosine_neighbors(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) of the k nearest neighbours of each row.

    Cosine similarity on L2-normalised vectors is a plain dot product, so the
    whole thing is one matrix multiply. It is chunked because the full n x n
    similarity matrix is the one allocation here big enough to matter.
    """
    unit = _l2_normalize(vectors)
    count = unit.shape[0]
    neighbor_idx = np.zeros((count, k), dtype=np.int32)
    neighbor_sim = np.zeros((count, k), dtype=np.float32)

    # ~64 MB per chunk of similarities at float32.
    chunk = max(1, min(count, 16_000_000 // max(1, count)))
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        sims = unit[start:stop] @ unit.T
        # Exclude self-matches before ranking.
        for offset in range(stop - start):
            sims[offset, start + offset] = -np.inf
        # argpartition gives the top k without a full sort of every row.
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        top_sims = sims[rows, top]
        order = np.argsort(-top_sims, axis=1)
        neighbor_idx[start:stop] = top[rows, order]
        neighbor_sim[start:stop] = top_sims[rows, order]

    return neighbor_idx, neighbor_sim


def _connected_components(count: int, edges: list[tuple[int, int]]) -> list[int]:
    """Union-find component id per node, so the UI can colour disjoint groups."""
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for source, target in edges:
        a, b = find(source), find(target)
        if a != b:
            parent[a] = b

    labels: dict[int, int] = {}
    result = []
    for node in range(count):
        root = find(node)
        if root not in labels:
            labels[root] = len(labels)
        result.append(labels[root])
    return result


def _networkx():
    """Import networkx, or return None when it is unavailable.

    It arrives as a dependency of scikit-image and is declared explicitly in
    requirements-core.txt, but the structural readings are an enrichment rather
    than the point of the view: without it the graph still draws, and the UI says
    which extras are missing instead of failing the request.
    """
    try:
        import networkx
    except ImportError:
        return None
    return networkx


def _graph_structure(count: int, pairs: list[tuple[int, int]], weights: list[float]) -> dict:
    """Communities, bridges and articulation points for the kept edge set."""
    empty = {
        "communities": None,
        "community_count": 0,
        "modularity": None,
        "bridges": [],
        "articulation_points": [],
        "available": False,
        "reason": None,
    }
    nx = _networkx()
    # Two unrelated causes, and the caller must be able to tell them apart: one
    # is fixed by installing a package, the other by lowering a threshold. Saying
    # the wrong one sends the reader to solve a problem they do not have.
    if nx is None:
        return {**empty, "reason": "missing_networkx"}
    if not pairs:
        return {**empty, "reason": "no_edges"}

    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    for (source, target), weight in zip(pairs, weights, strict=True):
        # Similarity is the edge weight: Louvain should treat a strong link as
        # harder to cut than a weak one.
        graph.add_edge(source, target, weight=max(0.0, weight))

    try:
        groups = nx.community.louvain_communities(graph, weight="weight", seed=LOUVAIN_SEED)
        modularity = float(nx.community.modularity(graph, groups, weight="weight"))
    except Exception:
        return {**empty, "reason": "analysis_failed"}

    # Largest community first, so community 0 is the dominant one and the palette
    # assigns its most distinct colour to the group the user notices first.
    groups = sorted(groups, key=len, reverse=True)
    communities = [0] * count
    for label, members in enumerate(groups):
        for node in members:
            communities[node] = label

    try:
        bridges = [[int(a), int(b)] for a, b in nx.bridges(graph)]
    except Exception:
        bridges = []
    try:
        articulation = sorted(int(node) for node in nx.articulation_points(graph))
    except Exception:
        articulation = []

    return {
        "communities": communities,
        "community_count": len(groups),
        "modularity": round(modularity, 6),
        "bridges": bridges,
        "articulation_points": articulation,
        "available": True,
        "reason": None,
    }


def build_neighbor_graph(
    embeddings: np.ndarray,
    k: int = 4,
    *,
    min_similarity: float | None = None,
    mutual_only: bool = False,
) -> dict:
    """Build the undirected k-NN graph of an embedding matrix.

    ``min_similarity`` drops weak edges, which is the quickest way to make a
    dense graph readable. ``mutual_only`` keeps just the reciprocal edges.
    """
    vectors = np.asarray(embeddings)
    if vectors.ndim != 2:
        raise ValueError("Neighbor graph input must be a 2D embedding matrix.")

    count = vectors.shape[0]
    if count < 2:
        raise ValueError("At least two images are required to build a neighbor graph.")
    if count > MAX_GRAPH_NODES:
        raise ValueError(
            f"This projection has {count} images; the neighbor graph is limited to {MAX_GRAPH_NODES}. "
            "Build the graph from a subfolder, or use the mosaic view for the whole set."
        )

    k = int(max(1, min(MAX_K, k, count - 1)))
    clean, unusable = _sanitize(vectors)
    neighbor_idx, neighbor_sim = _cosine_neighbors(clean, k)

    # Collapse the directed k-NN relation into undirected edges, remembering
    # which ones were reciprocal.
    directed: dict[tuple[int, int], float] = {}
    reciprocal: set[tuple[int, int]] = set()
    for source in range(count):
        for slot in range(k):
            target = int(neighbor_idx[source, slot])
            similarity = float(neighbor_sim[source, slot])
            if target == source or not np.isfinite(similarity):
                continue
            key = (source, target) if source < target else (target, source)
            if key in directed:
                reciprocal.add(key)
                directed[key] = max(directed[key], similarity)
            else:
                directed[key] = similarity

    edges = []
    kept_pairs: list[tuple[int, int]] = []
    kept_weights: list[float] = []
    for (source, target), similarity in directed.items():
        is_mutual = (source, target) in reciprocal
        if mutual_only and not is_mutual:
            continue
        if min_similarity is not None and similarity < min_similarity:
            continue
        edges.append({
            "source": source,
            "target": target,
            "similarity": round(similarity, 6),
            "mutual": is_mutual,
        })
        kept_pairs.append((source, target))
        kept_weights.append(similarity)

    edges.sort(key=lambda edge: edge["similarity"], reverse=True)

    degrees = [0] * count
    for source, target in kept_pairs:
        degrees[source] += 1
        degrees[target] += 1

    components = _connected_components(count, kept_pairs)
    structure = _graph_structure(count, kept_pairs, kept_weights)

    # Flag the edges that hold the graph together, so the renderer can draw them
    # differently without having to re-derive anything.
    bridge_set = {tuple(sorted(pair)) for pair in structure["bridges"]}
    for edge in edges:
        edge["bridge"] = tuple(sorted((edge["source"], edge["target"]))) in bridge_set

    return {
        "k": k,
        "node_count": count,
        "edge_count": len(edges),
        "component_count": (max(components) + 1) if components else 0,
        "mutual_only": bool(mutual_only),
        "min_similarity": min_similarity,
        "unusable_vectors": unusable,
        "edges": edges,
        "degrees": degrees,
        "components": components,
        # Structural reading (see the module docstring). `structure_available`
        # is False when networkx is missing; everything else still works.
        "communities": structure["communities"],
        "community_count": structure["community_count"],
        "modularity": structure["modularity"],
        "bridges": structure["bridges"],
        "articulation_points": structure["articulation_points"],
        "structure_available": structure["available"],
        "structure_reason": structure.get("reason"),
    }
