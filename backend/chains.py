"""The shortest chain of neighbours linking two images.

*Find similar* answers "what else looks like this one?" and *Compare* answers
"how related are these two?". Neither says what lies between them. This does:
the shortest route from one image to another through the nearest-neighbour
graph, so a jump the numbers describe as 0.41 becomes six pictures that turn one
into the other.

For a collection of paintings that is often the answer worth having. Two works
can be linked through their subject, through a shared palette or through a
compositional habit, and which of those the model used is invisible in a
similarity score but obvious in the chain.

Two things the result is careful about.

The route is the shortest path in the graph, not a straight line in the
embedding space, so it depends on *k*: a sparser graph forces longer detours,
and at a low enough *k* two images may not be connected at all. The response
says which component each endpoint is in rather than silently returning
nothing.

Edges are weighted by ``1 - similarity``, so the chain chosen among the shortest
ones is the one whose individual steps are tightest. Counting hops alone would
happily route through a weak link.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import _find_row, _row_payload
from .neighbors import MAX_K, _cosine_neighbors, _sanitize
from .runtime_dirs import ROOT
from .views import load_projection, projection_sidecar

# Above this the graph build is too slow to run inside a request; it matches the
# neighbour graph's own ceiling, since it is the same construction.
MAX_CHAIN_IMAGES = 20_000

DEFAULT_K = 6


def _dijkstra(adjacency: list[list[tuple[int, float]]], source: int, target: int) -> tuple[list[int], list[float]]:
    """Cheapest route from source to target, by summed ``1 - similarity``.

    Written out rather than delegated to networkx: that dependency is optional
    here (the neighbour graph degrades gracefully without it), and this is a
    textbook loop over a graph with at most ``MAX_K`` edges per node.
    """
    import heapq

    count = len(adjacency)
    best = [float("inf")] * count
    previous = [-1] * count
    best[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    seen = [False] * count

    while queue:
        cost, node = heapq.heappop(queue)
        if seen[node]:
            continue
        seen[node] = True
        if node == target:
            break
        for neighbour, weight in adjacency[node]:
            if seen[neighbour]:
                continue
            candidate = cost + weight
            if candidate < best[neighbour]:
                best[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))

    if best[target] == float("inf"):
        return [], []

    route = [target]
    while route[-1] != source:
        route.append(previous[route[-1]])
    route.reverse()
    return route, [best[node] for node in route]


def _components(adjacency: list[list[tuple[int, float]]]) -> list[int]:
    """Connected-component label per node, so a failure can say *why*."""
    label = [-1] * len(adjacency)
    current = 0
    for start in range(len(adjacency)):
        if label[start] != -1:
            continue
        stack = [start]
        label[start] = current
        while stack:
            node = stack.pop()
            for neighbour, _ in adjacency[node]:
                if label[neighbour] == -1:
                    label[neighbour] = current
                    stack.append(neighbour)
        current += 1
    return label


def similarity_chain(
    projection_path: str,
    *,
    from_path: str | None = None,
    from_index: int | None = None,
    to_path: str | None = None,
    to_index: int | None = None,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """The shortest neighbour-to-neighbour route between two images."""
    path, rows, vectors = load_projection(projection_path)
    if len(rows) < 2:
        raise ValueError("A chain needs at least two images.")
    if len(rows) > MAX_CHAIN_IMAGES:
        raise ValueError(
            f"This projection has {len(rows)} images; chains are limited to {MAX_CHAIN_IMAGES}. "
            "Build the projection from a subfolder to use this view."
        )

    start = _find_row(rows, from_path, from_index)
    end = _find_row(rows, to_path, to_index)
    if start == end:
        raise ValueError("Pick two different images: an image is already where it is.")

    k = int(max(1, min(MAX_K, k, len(rows) - 1)))
    clean, unusable = _sanitize(vectors)
    if start in unusable or end in unusable:
        broken = rows[start if start in unusable else end].get("filename", "")
        raise ValueError(
            f"'{broken}' has an unusable embedding vector, so it cannot be linked to anything. "
            "Re-run the projection for this folder."
        )

    neighbour_idx, neighbour_sim = _cosine_neighbors(clean, k)

    # Undirected: an image reachable from another must be reachable back, or the
    # route would depend on which end was asked first.
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(rows))]
    similarity: dict[tuple[int, int], float] = {}
    for node in range(len(rows)):
        if node in unusable:
            continue
        for slot in range(k):
            other = int(neighbour_idx[node, slot])
            score = float(neighbour_sim[node, slot])
            if other == node or other in unusable or not np.isfinite(score):
                continue
            key = (node, other) if node < other else (other, node)
            if key in similarity and similarity[key] >= score:
                continue
            similarity[key] = score
    for (left, right), score in similarity.items():
        weight = max(0.0, 1.0 - score)
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))

    route, _ = _dijkstra(adjacency, start, end)
    sidecar = projection_sidecar(path) or {}
    base = {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "model_label": sidecar.get("model_label", ""),
        "k": k,
        "from": _row_payload(rows[start], start),
        "to": _row_payload(rows[end], end),
    }

    if not route:
        label = _components(adjacency)
        return {
            **base,
            "connected": False,
            "steps": [],
            "links": [],
            "reason": (
                "These two images are in separate parts of the neighbour graph at k="
                f"{k}, so no chain of neighbours links them. Raise k to connect more of "
                "the collection, or check whether one of them is an outlier."
            ),
            "component_sizes": [int(label.count(label[start])), int(label.count(label[end]))],
        }

    steps = [_row_payload(rows[node], node) for node in route]
    links = []
    for position in range(len(route) - 1):
        left, right = route[position], route[position + 1]
        key = (left, right) if left < right else (right, left)
        links.append(round(float(similarity.get(key, 0.0)), 6))

    direct = float(np.dot(clean[start], clean[end]) /
                   max(float(np.linalg.norm(clean[start]) * np.linalg.norm(clean[end])), 1e-12))
    return {
        **base,
        "connected": True,
        "steps": steps,
        "links": links,
        "hops": len(route) - 1,
        # The endpoints' own similarity, for scale: a long chain between two
        # images that are already close means the graph, not the space, is
        # keeping them apart.
        "direct_similarity": round(direct, 6),
        "weakest_link": round(min(links), 6) if links else None,
    }
