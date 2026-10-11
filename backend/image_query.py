"""Searching with images instead of words.

Text search needs the thing named. *Similar images* needs one example, which
makes the search as narrow as that one picture: ask with a Madonna and you get
Madonnas, including the ones that share only the gold ground.

This asks with several at once, and lets some of them be the wrong kind:

    like these three  →  the direction they have in common
    like these, not those  →  that direction with the unwanted part removed

Both are the same operation on vectors that `Lenses` already performs on
phrases. Doing it with pictures matters because the normal condition in front of
a visual collection is knowing what you are looking for and not being able to
name it - a palette, a way of placing a figure, a kind of light.

What the numbers mean. Positives are averaged after normalising, so one image
cannot dominate by having a longer vector; the average is the direction they
agree on, and the more they disagree the shorter it is - which is why a query
built from unrelated images returns weak, scattered results rather than a
confident wrong answer. Negatives are averaged the same way and subtracted. The
result is normalised again, so scores stay comparable with every other
similarity in the application.

Nothing is re-encoded: the query vectors are already on disk, which is what
makes this work for DINOv2 and Visual Features as well as for CLIP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import _find_row, _guard_size, _row_payload, _unit_vectors
from .runtime_dirs import ROOT
from .views import load_projection, projection_sidecar

# Enough to express a taste, few enough that the average still means something:
# past a couple of dozen images the direction they agree on is the collection's
# own centre, and the query stops saying anything.
MAX_QUERY_IMAGES = 24

# Below this the subtraction has cancelled the query out. Returning the nearest
# images to a near-zero vector would be returning noise with confident scores.
MIN_QUERY_NORM = 1e-3


def _resolve(rows: list[dict[str, str]], paths: list[str] | None, indexes: list[int] | None,
             label: str) -> list[int]:
    """Row indexes for one side of the query, in the order they were given."""
    found: list[int] = []
    for path in paths or []:
        found.append(_find_row(rows, path, None))
    for index in indexes or []:
        found.append(_find_row(rows, None, index))
    if len(found) > MAX_QUERY_IMAGES:
        raise ValueError(
            f"{len(found)} images given as {label}; the limit is {MAX_QUERY_IMAGES}. "
            "Past that the average is the collection's own centre and the query says nothing."
        )
    # Order is kept, duplicates are not: asking twice for the same picture would
    # silently weight it double.
    seen: set[int] = set()
    unique = []
    for index in found:
        if index not in seen:
            seen.add(index)
            unique.append(index)
    return unique


def similar_to_set(
    projection_path: str,
    *,
    positive_paths: list[str] | None = None,
    positive_indexes: list[int] | None = None,
    negative_paths: list[str] | None = None,
    negative_indexes: list[int] | None = None,
    top_k: int = 24,
    negative_weight: float = 1.0,
    min_similarity: float | None = None,
) -> dict[str, Any]:
    """Rank the collection by likeness to several images at once."""
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))

    positives = _resolve(rows, positive_paths, positive_indexes, "likes")
    negatives = _resolve(rows, negative_paths, negative_indexes, "unlikes")
    if not positives:
        raise ValueError("Add at least one image to the query: there is nothing to be like.")

    overlap = set(positives) & set(negatives)
    if overlap:
        name = rows[sorted(overlap)[0]].get("filename", "")
        raise ValueError(f"'{name}' is on both sides of the query. An image cannot be both wanted and unwanted.")

    unit, unusable = _unit_vectors(vectors)
    broken = [index for index in positives + negatives if index in unusable]
    if broken:
        name = rows[broken[0]].get("filename", "")
        raise ValueError(
            f"'{name}' has an unusable embedding vector, so it cannot take part in a query. "
            "Re-run the projection for this folder."
        )

    query = unit[positives].mean(axis=0)
    if negatives:
        query = query - float(negative_weight) * unit[negatives].mean(axis=0)

    norm = float(np.linalg.norm(query))
    if not np.isfinite(norm) or norm < MIN_QUERY_NORM:
        raise ValueError(
            "The images you added cancel each other out, so there is no direction left to search in. "
            "Remove an unwanted image, or lower its weight."
        )
    query = query / norm

    scores = unit @ query
    # The query images themselves are always the best answer to their own
    # question, and saying so helps nobody.
    for index in positives + negatives:
        scores[index] = -np.inf
    for index in unusable:
        scores[index] = -np.inf

    # Which of the chosen images each result is closest to: with a query built
    # from several, "why did this come back?" is a real question.
    closest = (unit[positives] @ unit.T).argmax(axis=0) if positives else None

    results = []
    for position in np.argsort(-scores):
        score = float(scores[int(position)])
        if not np.isfinite(score):
            continue
        if min_similarity is not None and score < min_similarity:
            break
        payload = _row_payload(rows[int(position)], int(position))
        payload["rank"] = len(results) + 1
        payload["score"] = round(score, 6)
        if closest is not None:
            source = positives[int(closest[int(position)])]
            payload["closest_to"] = rows[source].get("filename", "")
        results.append(payload)
        if len(results) >= top_k:
            break

    sidecar = projection_sidecar(path) or {}
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "model_label": sidecar.get("model_label", ""),
        "positives": [_row_payload(rows[index], index) for index in positives],
        "negatives": [_row_payload(rows[index], index) for index in negatives],
        "negative_weight": round(float(negative_weight), 4),
        # How much the chosen images actually agree: the length of their average
        # before it was normalised away. Near 1 they point the same way; near 0
        # they are pulling apart and the results will be weak whatever they are.
        "agreement": round(float(np.linalg.norm(unit[positives].mean(axis=0))), 4),
        "count": len(results),
        "searched": len(rows),
        "results": results,
    }
