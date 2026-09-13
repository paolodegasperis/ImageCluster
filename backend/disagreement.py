"""Where two models disagree about the same images.

CLIP-family models group by what a picture *shows*; DINOv2 and the visual-feature
provider group by how it *looks*. That is not one of them being wrong — they
answer different questions. The interesting part is the residue: the works the
two place differently.

An image whose neighbours are entirely different under the two models has a gap
between its subject and its form. A landscape painted like a portrait. A sacred
scene handled as genre. Neither model can point at those on its own, because
each one only sees its own answer; the disagreement is information that exists
only in the comparison.

The comparison is made on neighbourhoods rather than on coordinates. Two
embedding spaces have no common frame — their axes are unrelated and their
scales differ — so positions cannot be compared directly. "Which images does the
model consider closest to this one" is meaningful in both, and the overlap
between those two sets is a number that means the same thing either way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .neighbors import _sanitize
from .runtime_dirs import ROOT
from .views import load_projection, projection_sidecar

# Neighbourhood overlap is compared over this many neighbours unless the caller
# says otherwise. Too few and the measure is noisy; too many and every image
# overlaps with every other.
DEFAULT_NEIGHBOURS = 15
MAX_DISAGREEMENT_IMAGES = 20_000


def _neighbour_sets(vectors: np.ndarray, k: int) -> list[set[int]]:
    """The k nearest neighbours of every row, as sets."""
    clean, _ = _sanitize(vectors)
    norms = np.linalg.norm(clean, axis=1, keepdims=True)
    unit = clean / np.clip(norms, 1e-12, None)

    count = unit.shape[0]
    block = max(1, min(count, 16_000_000 // max(1, count)))
    result: list[set[int]] = []
    for start in range(0, count, block):
        stop = min(start + block, count)
        sims = unit[start:stop] @ unit.T
        for offset in range(stop - start):
            sims[offset, start + offset] = -np.inf
        top = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for row in top:
            result.append({int(i) for i in row})
    return result


def model_disagreement(
    projection_a: str,
    projection_b: str,
    *,
    neighbours: int = DEFAULT_NEIGHBOURS,
    top_k: int = 40,
) -> dict[str, Any]:
    """Rank the images the two projections place most differently.

    Both projections must cover the same images, in the same order — which they
    do when they were generated from the same folder, because the manifest fixes
    the order. The check is explicit rather than assumed: comparing two different
    image sets row by row would produce plausible numbers about nothing.
    """
    path_a, rows_a, vectors_a = load_projection(projection_a)
    path_b, rows_b, vectors_b = load_projection(projection_b)

    if path_a == path_b:
        raise ValueError("Pick two different projections to compare.")

    count = len(rows_a)
    if count > MAX_DISAGREEMENT_IMAGES:
        raise ValueError(
            f"These projections hold {count} images; the comparison is limited to "
            f"{MAX_DISAGREEMENT_IMAGES}. Run it on a subfolder instead."
        )

    paths_a = [row.get("relative_path", "") for row in rows_a]
    paths_b = [row.get("relative_path", "") for row in rows_b]
    if paths_a != paths_b:
        if set(paths_a) != set(paths_b):
            only_a = len(set(paths_a) - set(paths_b))
            only_b = len(set(paths_b) - set(paths_a))
            raise ValueError(
                "These two projections cover different images "
                f"({only_a} only in the first, {only_b} only in the second), so they cannot be "
                "compared. Generate both from the same folder."
            )
        raise ValueError(
            "These two projections hold the same images in a different order, so their rows "
            "do not correspond. Re-run the newer one to rebuild it from the current manifest."
        )

    if count < 3:
        raise ValueError("At least three images are needed to compare two models.")
    k = int(max(1, min(50, neighbours, count - 1)))

    sets_a = _neighbour_sets(vectors_a, k)
    sets_b = _neighbour_sets(vectors_b, k)

    # Overlap of the two neighbourhoods, 0 = the models share no neighbour for
    # this image, 1 = they agree completely.
    agreement = np.zeros(count, dtype="float32")
    shared_counts = np.zeros(count, dtype=np.int32)
    for index in range(count):
        shared = sets_a[index] & sets_b[index]
        shared_counts[index] = len(shared)
        agreement[index] = len(shared) / k

    sidecar_a = projection_sidecar(path_a) or {}
    sidecar_b = projection_sidecar(path_b) or {}

    order = np.argsort(agreement)  # least agreement first

    # Competition ranking: everything on the same score shares a rank.
    #
    # Overlap is a count out of k, so it takes only k+1 distinct values and ties
    # are the rule rather than the exception - on two genuinely different models
    # dozens of images sit at zero together. Numbering them 1, 2, 3 presented the
    # tie-break, which is index order and therefore alphabetical, as if it were a
    # measured ordering, and made the first few look singled out when nothing
    # distinguishes them from the rest of the tie.
    ranks: dict[int, int] = {}
    tie_sizes: dict[int, int] = {}
    position = 0
    while position < len(order):
        score = agreement[order[position]]
        run = position
        while run < len(order) and agreement[order[run]] == score:
            run += 1
        for offset in range(position, run):
            ranks[int(order[offset])] = position + 1
            tie_sizes[int(order[offset])] = run - position
        position = run

    results = []
    for index in order[:top_k]:
        index = int(index)
        row = rows_a[index]
        results.append({
            "index": index,
            "rank": ranks[index],
            # How many images hold this exact rank. 1 means the position is the
            # image's own; more means the order within the group is arbitrary.
            "tied_with": tie_sizes[index],
            "relative_path": row.get("relative_path", ""),
            "filename": row.get("filename", ""),
            # The shared card contract reads "score"; here a LOW score is the
            # interesting end, which the UI states explicitly.
            "score": round(float(agreement[index]), 6),
            "shared_neighbours": int(shared_counts[index]),
            "cluster": _optional_int(row.get("cluster")),
        })

    return {
        "projection_a": path_a.relative_to(ROOT).as_posix(),
        "projection_b": path_b.relative_to(ROOT).as_posix(),
        "model_a": sidecar_a.get("model_label", "") or sidecar_a.get("model_key", ""),
        "model_b": sidecar_b.get("model_label", "") or sidecar_b.get("model_key", ""),
        "neighbours": k,
        "count": count,
        "mean_agreement": round(float(agreement.mean()), 6),
        "median_agreement": round(float(np.median(agreement)), 6),
        # Images on which the two models share nothing at all.
        "total_disagreement": int((agreement == 0).sum()),
        # How many of the returned results share the best (lowest) rank, so the
        # caller can say "these N are equally far apart" instead of implying an
        # order that the measurement does not support.
        "tied_at_top": int(tie_sizes[int(order[0])]) if len(order) else 0,
        "full_agreement": int((agreement == 1).sum()),
        "results": results,
        "agreement": [round(float(value), 6) for value in agreement],
    }


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
