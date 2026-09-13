"""Read-only analyses over a saved projection's embeddings.

Everything here works on vectors already on disk, so nothing is re-encoded and
no model is loaded — which is what makes these answers instant, and what makes
them available for image-only models (DINOv2, Visual Features) that cannot do
text search at all.

Four questions, one shared similarity pass:

``similar_images``
    "What else looks like this one?" — the query is an image already in the
    projection, so it needs no encoder, unlike text search.

``duplicate_groups``
    "Which images are effectively the same?" — near-duplicates above a
    similarity threshold, grouped transitively.

``outlier_ranking``
    "What is unlike everything else?" — ranked by distance to the k-th nearest
    neighbour, the standard local-density measure.

``cluster_summaries``
    "What is each cluster actually about?" — size, cohesion, and the medoid: the
    image closest to the cluster's centre, i.e. its most representative member.

``compare_images``
    "How related are these two, exactly?" — the raw similarity is hard to read on
    its own (is 0.72 close?), so it is reported alongside where each image ranks
    in the other's neighbour list, where the pair sits in the distribution of all
    pairs, and which neighbours the two have in common.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .neighbors import MAX_K, _sanitize
from .runtime_dirs import ROOT
from .views import load_projection, projection_sidecar

# The full similarity matrix is the only allocation here that can get large, so
# it is computed in row blocks of roughly this many float32 entries (~64 MB).
_SIMILARITY_BLOCK = 16_000_000

# Above this, a pairwise duplicate scan is too slow to run synchronously in a
# request. It is far above the neighbor graph's cap because no edge payload is
# produced — only the pairs that exceed the threshold.
MAX_ANALYSIS_IMAGES = 20_000

# Cosine similarity at or above this counts as "the same image" by default.
# Re-encodes, re-saves and light crops of one work land in the high 0.9s.
DEFAULT_DUPLICATE_THRESHOLD = 0.98


def _unit_vectors(vectors: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """L2-normalised copy of the matrix, with unusable rows zeroed and reported."""
    clean, unusable = _sanitize(vectors)
    norms = np.linalg.norm(clean, axis=1, keepdims=True)
    return clean / np.clip(norms, 1e-12, None), unusable


def _guard_size(count: int) -> None:
    if count > MAX_ANALYSIS_IMAGES:
        raise ValueError(
            f"This projection has {count} images; the analysis tools are limited to "
            f"{MAX_ANALYSIS_IMAGES}. Run them on a subfolder instead."
        )


def _row_payload(row: dict[str, str], index: int) -> dict[str, Any]:
    """The shape every result list uses, so one card component renders them all."""
    raw_cluster = row.get("cluster")
    cluster: int | None = None
    if raw_cluster not in (None, "", "null"):
        try:
            cluster = int(float(raw_cluster))
        except (TypeError, ValueError):
            cluster = None

    def coordinate(key: str) -> float | None:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    return {
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "x": coordinate("x"),
        "y": coordinate("y"),
        "cluster": cluster,
    }


def _find_row(rows: list[dict[str, str]], relative_path: str | None, index: int | None) -> int:
    """Resolve a query image to its row index, by path or by position."""
    if index is not None:
        if not 0 <= index < len(rows):
            raise ValueError(f"Image index {index} is outside this projection (0-{len(rows) - 1}).")
        return index
    if not relative_path:
        raise ValueError("Provide either relative_path or index to identify the query image.")
    for position, row in enumerate(rows):
        if row.get("relative_path") == relative_path:
            return position
    raise ValueError(f"'{relative_path}' is not part of this projection.")


# --------------------------------------------------------------------------
# 1. Similar images


def similar_images(
    projection_path: str,
    *,
    relative_path: str | None = None,
    index: int | None = None,
    top_k: int = 24,
    min_similarity: float | None = None,
) -> dict[str, Any]:
    """Rank every other image by similarity to one image in the projection.

    This is text search's counterpart for a query that is itself an image, and
    it is cheaper: no encoder runs, because the query vector is already stored.
    """
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))

    query_index = _find_row(rows, relative_path, index)
    unit, unusable = _unit_vectors(vectors)

    if query_index in unusable:
        raise ValueError(
            f"'{rows[query_index].get('filename', '')}' has an unusable embedding vector, "
            "so it cannot be compared with anything. Re-run the projection for this folder."
        )

    scores = unit @ unit[query_index]
    scores[query_index] = -np.inf  # never return the query as its own match

    order = np.argsort(-scores)
    results = []
    # Rank comes from the kept list, not the scan position: rows can be skipped.
    for position in order:
        score = float(scores[int(position)])
        if not np.isfinite(score):
            continue
        if min_similarity is not None and score < min_similarity:
            break  # sorted descending: everything after this is weaker too
        payload = _row_payload(rows[int(position)], int(position))
        payload.update({"rank": len(results) + 1, "score": round(score, 6)})
        results.append(payload)
        if len(results) >= top_k:
            break

    sidecar = projection_sidecar(path) or {}
    query = _row_payload(rows[query_index], query_index)
    query["score"] = 1.0
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "model_label": sidecar.get("model_label", ""),
        "query": query,
        "count": len(results),
        "searched": len(rows),
        "results": results,
    }


# --------------------------------------------------------------------------
# 2. Duplicates and near-duplicates


def _pairs_above(unit: np.ndarray, threshold: float, limit: int) -> list[tuple[int, int, float]]:
    """Every pair whose cosine similarity reaches ``threshold``, upper triangle only."""
    count = unit.shape[0]
    block = max(1, min(count, _SIMILARITY_BLOCK // max(1, count)))
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, count, block):
        stop = min(start + block, count)
        sims = unit[start:stop] @ unit.T
        # Keep only j > i so each pair is found once and self-matches drop out.
        rows_idx, cols_idx = np.nonzero(sims >= threshold)
        for local_row, col in zip(rows_idx, cols_idx, strict=True):
            i = start + int(local_row)
            j = int(col)
            if j <= i:
                continue
            pairs.append((i, j, float(sims[local_row, col])))
            if len(pairs) >= limit:
                return pairs
    return pairs


def duplicate_groups(
    projection_path: str,
    *,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    max_pairs: int = 20_000,
) -> dict[str, Any]:
    """Group images that are effectively the same picture.

    Grouping is transitive (union-find): if A matches B and B matches C, all
    three land in one group even when A and C fall just under the threshold.
    That is what an archivist wants for a chain of re-saves — one work, not two
    overlapping pairs — but it is also how a tight *series* can get swallowed
    whole when the threshold sits just below the series' internal similarity.

    Two things keep that honest. Groups are ordered by how alike their members
    are, not by size, so the confident groups come first and a long chain does
    not push them out of sight. And a group whose weakest internal pair falls
    below the threshold is flagged ``chained``: its members were joined through
    intermediaries rather than all matching each other directly, which is the
    signal to look before deleting anything.
    """
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))
    if not 0.0 < threshold <= 1.0:
        raise ValueError("The duplicate threshold must be between 0 and 1.")

    unit, unusable = _unit_vectors(vectors)
    # A zeroed row has similarity 0 with everything, so it can never be a duplicate.
    pairs = _pairs_above(unit, threshold, max_pairs)
    truncated = len(pairs) >= max_pairs

    parent = list(range(len(rows)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    # Every image that appears in at least one pair, collected once. Rebuilding
    # this set per row would be O(rows x pairs) — minutes on a large archive.
    involved: set[int] = set()
    for i, j, _ in pairs:
        involved.add(i)
        involved.add(j)
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    members: dict[int, list[int]] = {}
    for node in sorted(involved):
        members.setdefault(find(node), []).append(node)

    groups = []
    for indices in members.values():
        if len(indices) < 2:
            continue
        block = unit[indices]
        # Accumulated in row blocks rather than as one (g, g) matrix. A
        # collection of near-duplicates - video frames, a scan series, a
        # bracketed set - merges into a single group approaching the 20 000-image
        # ceiling, where the full matrix plus its boolean mask runs to gigabytes
        # and raises MemoryError, which no route catches.
        size = len(indices)
        centrality = np.empty(size, dtype="float64")
        total = 0.0
        weakest = 1.0
        step = max(1, min(size, 4_000_000 // max(1, size)))
        for start in range(0, size, step):
            stop = min(start + step, size)
            rows_block = block[start:stop] @ block.T
            centrality[start:stop] = rows_block.sum(axis=1)
            total += float(rows_block.sum())
            # The weakest pair decides whether the group was chained together
            # through intermediaries. The diagonal is 1.0 and would never be the
            # minimum, but it is masked anyway so a one-member block is right.
            masked = rows_block.copy()
            for offset in range(stop - start):
                masked[offset, start + offset] = np.inf
            if masked.size:
                block_min = float(masked.min())
                if np.isfinite(block_min):
                    weakest = min(weakest, block_min)
        # Mean off-diagonal similarity: how alike the group is overall. The
        # diagonal contributes exactly one per row, so it is subtracted rather
        # than masked out.
        pair_count = size * (size - 1)
        mean_similarity = (total - size) / pair_count if pair_count else 0.0
        order = list(np.argsort(-centrality))
        items = []
        for rank, local in enumerate(order, start=1):
            payload = _row_payload(rows[indices[local]], indices[local])
            payload.update({
                "rank": rank,
                # Mean similarity to the other members: how central this copy is.
                "score": round(float(centrality[local] - 1.0) / max(1, len(indices) - 1), 6),
                # The most central member, offered as the one to keep. It is a
                # suggestion, not a verdict - especially in a chained group.
                "most_central": rank == 1,
            })
            items.append(payload)
        groups.append({
            "size": len(indices),
            "mean_similarity": round(mean_similarity, 6) if pair_count else 1.0,
            "min_similarity": round(weakest, 6),
            # Not every member matched every other one directly; they were linked
            # through intermediaries. Review before treating these as duplicates.
            "chained": bool(weakest < threshold),
            "images": items,
        })

    # Confidence first, then size. Ordering by size would put the longest chain -
    # the least certain group - at the top, which is exactly backwards.
    groups.sort(key=lambda group: (-group["mean_similarity"], -group["size"]))

    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "threshold": threshold,
        "group_count": len(groups),
        "duplicate_image_count": sum(group["size"] for group in groups),
        "redundant_image_count": sum(group["size"] - 1 for group in groups),
        "chained_group_count": sum(1 for group in groups if group["chained"]),
        "searched": len(rows),
        "unusable_vectors": unusable,
        "truncated": truncated,
        "groups": groups,
    }


# --------------------------------------------------------------------------
# 3. Outliers


def outlier_ranking(
    projection_path: str,
    *,
    k: int = 5,
    top_k: int = 40,
) -> dict[str, Any]:
    """Rank images by how unlike the rest of the collection they are.

    The measure is the similarity to the k-th nearest neighbour: an image that
    has a few close matches is typical even if it sits far from the centre,
    while an image whose k-th neighbour is already distant has no company at
    all. Using the k-th rather than the single nearest avoids calling a pair of
    duplicates "typical" when nothing else resembles either of them.
    """
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))
    count = len(rows)
    if count < 3:
        raise ValueError("At least three images are needed to tell an outlier from the rest.")

    k = int(max(1, min(MAX_K, k, count - 1)))
    unit, unusable = _unit_vectors(vectors)

    kth = np.zeros(count, dtype="float32")
    mean_top = np.zeros(count, dtype="float32")
    block = max(1, min(count, _SIMILARITY_BLOCK // max(1, count)))
    for start in range(0, count, block):
        stop = min(start + block, count)
        sims = unit[start:stop] @ unit.T
        for offset in range(stop - start):
            sims[offset, start + offset] = -np.inf
        top = np.partition(sims, -k, axis=1)[:, -k:]
        kth[start:stop] = top.min(axis=1)
        mean_top[start:stop] = top.mean(axis=1)

    order = np.argsort(kth)  # least similar to its k-th neighbour = most isolated
    results = []
    for rank, position in enumerate(order[:top_k], start=1):
        payload = _row_payload(rows[int(position)], int(position))
        payload.update({
            "rank": rank,
            # "score" keeps the shared card contract; here it is the k-th-neighbour
            # similarity, so a low value means isolated.
            "score": round(float(kth[int(position)]), 6),
            "mean_neighbour_similarity": round(float(mean_top[int(position)]), 6),
            "unusable": int(position) in unusable,
        })
        results.append(payload)

    finite = kth[np.isfinite(kth)]
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "k": k,
        "count": len(results),
        "searched": count,
        "median_similarity": round(float(np.median(finite)), 6) if finite.size else 0.0,
        "unusable_vectors": unusable,
        "results": results,
    }


# --------------------------------------------------------------------------
# 4. Cluster summaries


def cluster_summaries(projection_path: str) -> dict[str, Any]:
    """Describe each cluster: how big, how tight, and which image represents it.

    The medoid — the member closest to the cluster's centre — is a real image
    rather than a synthetic average, so it can be shown as the cluster's face.
    Ranking members by distance from that centre also gives the gallery a
    meaningful order: most typical first, marginal cases last. The projection's
    own ``cluster_score`` cannot do that, being one silhouette value for the
    whole run and therefore identical on every row.
    """
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))

    unit, unusable = _unit_vectors(vectors)

    buckets: dict[int, list[int]] = {}
    unlabelled = 0
    for index, row in enumerate(rows):
        raw = row.get("cluster")
        if raw in (None, "", "null"):
            unlabelled += 1
            continue
        try:
            label = int(float(raw))
        except (TypeError, ValueError):
            unlabelled += 1
            continue
        buckets.setdefault(label, []).append(index)

    if not buckets:
        raise ValueError(
            "This projection has no clusters. Re-run it with clustering enabled to get summaries."
        )

    clusters = []
    for label in sorted(buckets):
        indices = buckets[label]
        block = unit[indices]
        centroid = block.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid = centroid / norm if norm > 1e-12 else centroid
        # Similarity to the centroid is the typicality measure; the maximum is the medoid.
        typicality = block @ centroid
        order = list(np.argsort(-typicality))

        members = []
        for rank, local in enumerate(order, start=1):
            payload = _row_payload(rows[indices[local]], indices[local])
            payload.update({
                "rank": rank,
                "score": round(float(typicality[local]), 6),
                "is_medoid": rank == 1,
            })
            members.append(payload)

        clusters.append({
            "cluster": label,
            "size": len(indices),
            # Mean similarity to the centre: high means the cluster is one thing,
            # low means K-Means merged several.
            "cohesion": round(float(typicality.mean()), 6),
            "spread": round(float(typicality.min()), 6),
            "medoid": members[0],
            "members": members,
        })

    total_labelled = sum(cluster["size"] for cluster in clusters)
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "cluster_count": len(clusters),
        "labelled": total_labelled,
        "unlabelled": unlabelled,
        "unusable_vectors": unusable,
        "clusters": clusters,
    }


# --------------------------------------------------------------------------
# 5. Two-image comparison


def compare_images(
    projection_path: str,
    *,
    path_a: str,
    path_b: str,
    neighbours: int = 12,
) -> dict[str, Any]:
    """Compare two images of the projection in the model's own terms.

    A bare cosine value is almost unreadable on its own: whether 0.72 means "very
    close" depends entirely on the model and the collection. Three pieces of
    context make it legible.

    *Reciprocal rank* — where each image sits in the other's neighbour list. Rank
    1 both ways means they are each other's closest match; rank 1 one way and 400
    the other means one of them is an outlier clinging to a popular image.

    *Percentile* — how this pair compares with every other pair involving the same
    image, so "closer than 99% of the collection" replaces a bare number.

    *Shared neighbours* — the images both are close to. Two works can have a
    middling direct similarity and still belong together because they sit in the
    same neighbourhood; this is what shows that.
    """
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))

    index_a = _find_row(rows, path_a, None)
    index_b = _find_row(rows, path_b, None)
    if index_a == index_b:
        raise ValueError("Pick two different images to compare.")

    unit, unusable = _unit_vectors(vectors)
    for index in (index_a, index_b):
        if index in unusable:
            raise ValueError(
                f"'{rows[index].get('filename', '')}' has an unusable embedding vector, "
                "so it cannot be compared. Re-run the projection for this folder."
            )

    count = len(rows)
    neighbours = int(max(1, min(100, neighbours, count - 1)))

    scores_a = unit @ unit[index_a]
    scores_b = unit @ unit[index_b]
    similarity = float(scores_a[index_b])

    def ranking(scores: np.ndarray, self_index: int) -> np.ndarray:
        masked = scores.copy()
        masked[self_index] = -np.inf
        return np.argsort(-masked)

    order_a = ranking(scores_a, index_a)
    order_b = ranking(scores_b, index_b)
    rank_of_b = int(np.flatnonzero(order_a == index_b)[0]) + 1
    rank_of_a = int(np.flatnonzero(order_b == index_a)[0]) + 1

    def percentile(scores: np.ndarray, self_index: int) -> float:
        others = np.delete(scores, self_index)
        if others.size == 0:
            return 0.0
        return round(float((others < similarity).mean()) * 100.0, 2)

    top_a = [int(i) for i in order_a[:neighbours]]
    top_b = [int(i) for i in order_b[:neighbours]]
    set_a, set_b = set(top_a), set(top_b)
    shared_indices = [i for i in top_a if i in set_b and i not in (index_a, index_b)]
    union = len(set_a | set_b) or 1

    shared = []
    for rank, index in enumerate(shared_indices, start=1):
        payload = _row_payload(rows[index], index)
        payload.update({
            "rank": rank,
            # The card component reads "score"; here it is the weaker of the two
            # similarities, i.e. how firmly this image belongs to *both*.
            "score": round(float(min(scores_a[index], scores_b[index])), 6),
            "similarity_to_a": round(float(scores_a[index]), 6),
            "similarity_to_b": round(float(scores_b[index]), 6),
        })
        shared.append(payload)
    shared.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(shared, start=1):
        item["rank"] = rank

    sidecar = projection_sidecar(path) or {}
    image_a = _row_payload(rows[index_a], index_a)
    image_b = _row_payload(rows[index_b], index_b)
    image_a["score"] = similarity
    image_b["score"] = similarity

    # Median pair similarity gives the reader a baseline for "normal".
    sample = scores_a[np.arange(count) != index_a]
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "model_label": sidecar.get("model_label", ""),
        "a": image_a,
        "b": image_b,
        "similarity": round(similarity, 6),
        "rank_of_b_in_a": rank_of_b,
        "rank_of_a_in_b": rank_of_a,
        "percentile_in_a": percentile(scores_a, index_a),
        "percentile_in_b": percentile(scores_b, index_b),
        "median_similarity": round(float(np.median(sample)), 6) if sample.size else 0.0,
        "neighbours": neighbours,
        "shared_count": len(shared),
        "shared_fraction": round(len(shared_indices) / union, 4),
        "same_cluster": (
            image_a["cluster"] is not None and image_a["cluster"] == image_b["cluster"]
        ),
        "searched": count,
        "shared_neighbours": shared,
    }
