"""Grid-mosaic layout: turn a 2D projection into a dense, gapless grid.

A scatter plot of thumbnails becomes unreadable exactly where it matters most —
in the dense regions, which are the ones a curator wants to look at. Images
overlap, small clusters hide behind large ones, and apparent size carries no
meaning. A mosaic assigns every image to its own cell of a rectangular grid
while keeping neighbours in the projection neighbours in the grid, so each work
occupies equal space and local structure stays legible.

Two assignment strategies are provided:

``optimal``
    A true linear-assignment solution (``scipy.optimize.linear_sum_assignment``,
    a Jonker-Volgenant style solver) minimising total squared distance between
    each image's projected position and its assigned cell centre. This is the
    best possible mosaic for a given grid, but it is roughly O(n^3) in time and
    O(n * cells) in memory, so it is only used for small collections.

``split``
    Recursive median splitting: the grid is cut in half along its longer side,
    the images are cut along the matching axis in the same proportion, and the
    halves recurse. It runs in O(n log n), needs no extra dependency, and
    produces a mosaic that preserves neighbourhood structure well even though it
    is not globally optimal. This is the default for larger collections.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np

# Above this many images the exact solver's cost matrix and runtime stop being
# reasonable on a laptop, so the recursive split takes over.
OPTIMAL_SOLVER_MAX_IMAGES = 2000


def scipy_available() -> bool:
    return importlib.util.find_spec("scipy") is not None


def grid_dimensions(count: int, columns: int | None = None, aspect: float = 1.0) -> tuple[int, int]:
    """Choose a grid that holds ``count`` images with as few empty cells as possible.

    ``aspect`` is width / height of the area the mosaic will be drawn into, so a
    wide canvas yields a wide grid instead of a square one floating in the middle.
    """
    if count < 1:
        raise ValueError("A mosaic needs at least one image.")
    if columns is not None:
        cols = max(1, int(columns))
    else:
        safe_aspect = aspect if aspect and aspect > 0 else 1.0
        cols = max(1, int(round(math.sqrt(count * safe_aspect))))
    cols = min(cols, count)
    rows = int(math.ceil(count / cols))
    return cols, rows


def _normalize(coords: np.ndarray) -> np.ndarray:
    """Map coordinates into the unit square, preserving relative positions.

    Non-finite values are placed at the middle of their axis rather than being
    allowed into the arithmetic: an infinity makes the span infinite, which
    collapses every real coordinate onto 0 and turns the infinity itself into
    NaN. Callers coming through the API filter these rows out beforehand, but the
    function is also used directly and must not depend on that.
    """
    values = np.asarray(coords, dtype="float64")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Mosaic input must be an array of 2D coordinates.")

    out = np.full(values.shape, 0.5, dtype="float64")
    for axis in (0, 1):
        column = values[:, axis]
        finite = np.isfinite(column)
        if not finite.any():
            continue
        low = float(column[finite].min())
        high = float(column[finite].max())
        span = high - low
        if span <= 0:
            continue
        out[finite, axis] = (column[finite] - low) / span
    return out


def _cell_centres(cols: int, rows: int) -> np.ndarray:
    """Centre of every grid cell in the same unit square as the coordinates."""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    grid_x, grid_y = np.meshgrid(xs, ys)
    # Row-major: cell index = row * cols + col.
    return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def _assign_optimal(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    # Squared euclidean cost; the square root is monotonic so omitting it does
    # not change the optimum and keeps the matrix cheaper to build.
    diff = points[:, None, :] - centres[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    _, cell_for_point = linear_sum_assignment(cost)
    return cell_for_point.astype(int)


def _assign_recursive_split(points: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Recursive median split of points against grid cells.

    The grid is halved along its longer side and the points are split along the
    matching coordinate in proportion to the cell counts, so both halves stay
    full and empty cells end up spread across the mosaic rather than pooled in
    one corner.
    """
    assignment = np.full(points.shape[0], -1, dtype=int)

    def recurse(indices: np.ndarray, col0: int, row0: int, width: int, height: int) -> None:
        if indices.size == 0 or width <= 0 or height <= 0:
            return
        if width == 1 and height == 1:
            assignment[indices[0]] = row0 * cols + col0
            return

        if width >= height:
            first = width // 2
            cells_first = first * height
            cells_second = (width - first) * height
            axis = 0
        else:
            first = height // 2
            cells_first = width * first
            cells_second = width * (height - first)
            axis = 1

        total = indices.size
        target = int(round(total * cells_first / float(cells_first + cells_second)))
        # Never hand a half more points than it has cells, and never leave the
        # other half over-subscribed.
        target = min(target, cells_first)
        target = max(target, total - cells_second)
        target = max(0, min(total, target))

        order = indices[np.argsort(points[indices, axis], kind="stable")]
        head, tail = order[:target], order[target:]

        if axis == 0:
            recurse(head, col0, row0, first, height)
            recurse(tail, col0 + first, row0, width - first, height)
        else:
            recurse(head, col0, row0, width, first)
            recurse(tail, col0, row0 + first, width, height - first)

    recurse(np.arange(points.shape[0]), 0, 0, cols, rows)
    return assignment


def grid_mosaic(
    coords: np.ndarray,
    *,
    columns: int | None = None,
    aspect: float = 1.0,
    algorithm: str = "auto",
) -> dict:
    """Assign each projected point to a distinct grid cell.

    Returns the grid shape, the algorithm actually used, and one ``(col, row)``
    per input point in input order.
    """
    points = _normalize(coords)
    count = points.shape[0]
    cols, rows = grid_dimensions(count, columns, aspect)

    requested = (algorithm or "auto").lower()
    if requested not in {"auto", "optimal", "split"}:
        raise ValueError(f"Unknown mosaic algorithm '{algorithm}'. Use auto, optimal or split.")

    use_optimal = requested == "optimal" or (requested == "auto" and count <= OPTIMAL_SOLVER_MAX_IMAGES)
    note = ""
    if use_optimal and not scipy_available():
        if requested == "optimal":
            raise RuntimeError(
                "The optimal mosaic assignment needs SciPy, which is not installed. "
                "Choose the 'split' algorithm, or install scipy."
            )
        use_optimal = False
        note = "SciPy is not installed, so the recursive-split assignment was used."

    if use_optimal:
        cell_for_point = _assign_optimal(points, _cell_centres(cols, rows))
        used = "optimal"
    else:
        cell_for_point = _assign_recursive_split(points, cols, rows)
        used = "split"
        if requested == "auto" and count > OPTIMAL_SOLVER_MAX_IMAGES and not note:
            note = (
                f"{count} images exceed the {OPTIMAL_SOLVER_MAX_IMAGES}-image limit for the exact solver, "
                "so the recursive-split assignment was used."
            )

    if int((cell_for_point >= 0).sum()) != count:
        raise RuntimeError("Mosaic assignment failed to place every image.")
    if len(set(cell_for_point.tolist())) != count:
        raise RuntimeError("Mosaic assignment produced duplicate cells.")

    return {
        "columns": cols,
        "rows": rows,
        "algorithm": used,
        "note": note,
        "cells": [(int(cell % cols), int(cell // cols)) for cell in cell_for_point],
    }
