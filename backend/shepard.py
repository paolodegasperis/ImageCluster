"""How much the 2D map lies.

A projection takes a space of several hundred dimensions and flattens it onto a
page. Something always gets bent, and the map never says so: two images can sit
a centimetre apart because they are alike, or because UMAP had nowhere else to
put them. The neighbour graph already colours each edge by that discrepancy;
this reads the same thing globally.

A Shepard diagram plots, for a sample of image pairs, the distance in the real
embedding space against the distance on the map. A faithful projection draws a
narrow rising band. Everything that strays from it is a pair the map describes
wrongly, and the two directions mean different things:

*compressed*
    far apart in the embedding, close together on the map. The dangerous one:
    the map invites a comparison the model never made.

*stretched*
    close in the embedding, far apart on the map. A relationship the map hides,
    which is where the neighbour graph is worth opening.

Nothing is re-encoded here: the vectors and the coordinates are both already on
disk.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import _guard_size, _row_payload, _unit_vectors
from .runtime_dirs import ROOT
from .views import load_projection, projection_sidecar

# Pairs sampled when the collection is too large to compare exhaustively. 120k
# pairs is plenty for a stable density plot and for the rank correlation, and it
# keeps the request well under a second.
DEFAULT_SAMPLE_PAIRS = 120_000
MAX_SAMPLE_PAIRS = 400_000

# Resolution of the density grid sent to the browser. The raw pairs are never
# shipped: 120k points would be a megabyte of JSON drawn as one dark smudge.
GRID = 48

# How many of the worst-described pairs to name in each direction.
WORST_PAIRS = 8

# Fixed so the same projection always produces the same picture. A diagram that
# changed on every reload could not be compared with the one in a report.
_SEED = 20260411


def _pair_sample(count: int, limit: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs to measure: every pair when that is cheap, a sample when not."""
    total = count * (count - 1) // 2
    if total <= limit:
        return np.triu_indices(count, k=1)
    # Drawing with replacement and discarding the degenerate pairs is far cheaper
    # than enumerating a billion pairs to choose from, and the duplicate rate at
    # these sizes is negligible.
    draw = int(limit * 1.1) + 16
    first = rng.integers(0, count, size=draw)
    second = rng.integers(0, count, size=draw)
    keep = first != second
    first, second = first[keep], second[keep]
    return np.minimum(first, second)[:limit], np.maximum(first, second)[:limit]


def _coordinates(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[int]]:
    """The map positions, with unreadable ones reported rather than guessed."""
    coords = np.full((len(rows), 2), np.nan, dtype="float64")
    unusable: list[int] = []
    for index, row in enumerate(rows):
        try:
            x = float(row.get("x", ""))
            y = float(row.get("y", ""))
        except (TypeError, ValueError):
            unusable.append(index)
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            unusable.append(index)
            continue
        coords[index] = (x, y)
    return coords, unusable


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Spearman's rho, computed here so scipy stays an optional dependency."""
    if len(first) < 3:
        return None
    ranks_a = np.argsort(np.argsort(first)).astype("float64")
    ranks_b = np.argsort(np.argsort(second)).astype("float64")
    ranks_a -= ranks_a.mean()
    ranks_b -= ranks_b.mean()
    denom = float(np.sqrt((ranks_a * ranks_a).sum() * (ranks_b * ranks_b).sum()))
    if denom == 0:
        return None
    return float((ranks_a * ranks_b).sum() / denom)


def _stress(original: np.ndarray, mapped: np.ndarray) -> float | None:
    """Kruskal's stress-1, after the one rescaling that makes two different
    distance scales comparable at all.

    0 is a perfect map. Below about 0.1 the layout can be read as a picture of
    the embedding space; above roughly 0.2 it is also telling a story of its own.
    """
    denom = float((mapped * mapped).sum())
    if denom == 0:
        return None
    scale = float((original * mapped).sum() / denom)
    if not np.isfinite(scale) or scale == 0:
        return None
    residual = original - scale * mapped
    total = float((original * original).sum())
    if total == 0:
        return None
    return float(np.sqrt((residual * residual).sum() / total))


def shepard_diagram(
    projection_path: str,
    *,
    sample_pairs: int = DEFAULT_SAMPLE_PAIRS,
    worst: int = WORST_PAIRS,
) -> dict[str, Any]:
    """Embedding distance against map distance, over a sample of image pairs."""
    path, rows, vectors = load_projection(projection_path)
    _guard_size(len(rows))
    if len(rows) < 3:
        raise ValueError("A Shepard diagram needs at least three images.")

    limit = int(max(1_000, min(MAX_SAMPLE_PAIRS, sample_pairs)))
    worst = int(max(0, min(50, worst)))

    unit, bad_vectors = _unit_vectors(vectors)
    coords, bad_coords = _coordinates(rows)
    unusable = set(bad_vectors) | set(bad_coords)
    usable = np.array([i for i in range(len(rows)) if i not in unusable], dtype=np.int64)
    if len(usable) < 3:
        raise ValueError(
            "Fewer than three images have both a usable embedding and usable map coordinates, "
            "so the projection cannot be compared with the space it came from."
        )

    rng = np.random.default_rng(_SEED)
    left, right = _pair_sample(len(usable), limit, rng)
    first = usable[left]
    second = usable[right]

    # Cosine distance in the embedding space against Euclidean distance on the
    # map. Different units by nature, which is why the diagram is read as a
    # shape and not as a line of slope 1.
    embedding = 1.0 - np.einsum("ij,ij->i", unit[first], unit[second])
    delta = coords[first] - coords[second]
    mapped = np.sqrt(np.einsum("ij,ij->i", delta, delta))

    finite = np.isfinite(embedding) & np.isfinite(mapped)
    embedding, mapped = embedding[finite], mapped[finite]
    first, second = first[finite], second[finite]
    if len(embedding) < 3:
        raise ValueError("Too few comparable pairs remained to draw a diagram.")

    emb_max = float(embedding.max()) or 1.0
    map_max = float(mapped.max()) or 1.0
    emb_norm = embedding / emb_max
    map_norm = mapped / map_max

    # A density grid rather than raw points: the browser draws a picture, not a
    # list, and this keeps the response small whatever the collection's size.
    counts, _, _ = np.histogram2d(map_norm, emb_norm, bins=GRID, range=[[0, 1], [0, 1]])
    grid = counts.astype(np.int64).T  # rows: embedding distance, columns: map distance

    # Residual against the best single rescaling of the map distances: how far
    # each pair sits from the behaviour the projection claims overall.
    scale = float((emb_norm * map_norm).sum() / max(float((map_norm * map_norm).sum()), 1e-12))
    residual = emb_norm - scale * map_norm

    def pairs_for(order: np.ndarray) -> list[dict[str, Any]]:
        out = []
        for position in order[:worst]:
            at = int(position)
            out.append({
                "a": _row_payload(rows[int(first[at])], int(first[at])),
                "b": _row_payload(rows[int(second[at])], int(second[at])),
                "embedding_distance": round(float(embedding[at]), 6),
                "map_distance": round(float(mapped[at]), 6),
                "similarity": round(1.0 - float(embedding[at]), 6),
                "residual": round(float(residual[at]), 6),
            })
        return out

    sidecar = projection_sidecar(path) or {}
    rho = _rank_correlation(emb_norm, map_norm)
    stress = _stress(emb_norm, map_norm)
    return {
        "projection_path": path.relative_to(ROOT).as_posix(),
        "model_label": sidecar.get("model_label", ""),
        "reducer": sidecar.get("reducer", ""),
        "images": len(rows),
        "compared_images": int(len(usable)),
        "skipped_images": int(len(unusable)),
        "pairs": int(len(embedding)),
        "exhaustive": bool(len(usable) * (len(usable) - 1) // 2 <= limit),
        "grid": grid.tolist(),
        "grid_size": GRID,
        "max_embedding_distance": round(emb_max, 6),
        "max_map_distance": round(map_max, 6),
        "rank_correlation": None if rho is None else round(rho, 4),
        "stress": None if stress is None else round(stress, 4),
        # Positive residual: far apart in the embedding, close together on the map.
        "compressed": pairs_for(np.argsort(-residual)),
        "stretched": pairs_for(np.argsort(residual)),
    }
