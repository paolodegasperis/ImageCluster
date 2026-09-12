"""Grid-mosaic assignment.

The two invariants that matter for a mosaic are that every image gets a cell and
no two images share one: a mosaic that silently drops or stacks images is worse
than the scatter plot it replaces. Locality is checked separately — a mosaic
that scrambles neighbours is technically valid and useless.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.layouts import (
    OPTIMAL_SOLVER_MAX_IMAGES,
    grid_dimensions,
    grid_mosaic,
    scipy_available,
)

ALGORITHMS = ["split"] + (["optimal"] if scipy_available() else [])


def test_grid_dimensions_holds_every_image():
    for count in [1, 2, 3, 7, 10, 50, 101, 1000]:
        cols, rows = grid_dimensions(count)
        assert cols * rows >= count
        # Never waste a whole row: the grid must be tight.
        assert (cols * rows) - count < cols


def test_grid_dimensions_follows_the_canvas_aspect():
    wide_cols, wide_rows = grid_dimensions(100, aspect=4.0)
    tall_cols, tall_rows = grid_dimensions(100, aspect=0.25)
    assert wide_cols > wide_rows
    assert tall_cols < tall_rows


def test_grid_dimensions_respects_an_explicit_column_count():
    cols, rows = grid_dimensions(30, columns=5)
    assert (cols, rows) == (5, 6)


def test_grid_dimensions_never_exceeds_the_image_count():
    cols, rows = grid_dimensions(3, columns=99)
    assert cols == 3 and rows == 1


def test_grid_dimensions_rejects_an_empty_collection():
    with pytest.raises(ValueError):
        grid_dimensions(0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("count", [1, 2, 5, 16, 37, 100])
def test_every_image_gets_exactly_one_distinct_cell(algorithm, count):
    rng = np.random.default_rng(count)
    coords = rng.normal(0, 1, (count, 2))
    result = grid_mosaic(coords, algorithm=algorithm)
    cells = result["cells"]
    assert len(cells) == count
    assert len(set(cells)) == count
    for col, row in cells:
        assert 0 <= col < result["columns"]
        assert 0 <= row < result["rows"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_mosaic_preserves_locality(algorithm):
    """Two well-separated blobs must not be interleaved in the grid."""
    rng = np.random.default_rng(7)
    left = rng.normal((-10, 0), 0.3, (32, 2))
    right = rng.normal((10, 0), 0.3, (32, 2))
    coords = np.vstack([left, right])
    result = grid_mosaic(coords, columns=8, algorithm=algorithm)
    cells = result["cells"]
    left_cols = [c for c, _ in cells[:32]]
    right_cols = [c for c, _ in cells[32:]]
    # The blobs may occupy either side, but they must not overlap.
    assert max(left_cols) < min(right_cols) or max(right_cols) < min(left_cols)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_identical_coordinates_are_still_placed_uniquely(algorithm):
    coords = np.zeros((12, 2))
    result = grid_mosaic(coords, algorithm=algorithm)
    assert len(set(result["cells"])) == 12


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_degenerate_axis_does_not_collapse_the_mosaic(algorithm):
    # All points on a horizontal line: the y range is zero.
    coords = np.column_stack([np.linspace(0, 1, 20), np.zeros(20)])
    result = grid_mosaic(coords, algorithm=algorithm)
    assert len(set(result["cells"])) == 20


def test_auto_uses_the_split_solver_above_the_exact_limit(monkeypatch):
    monkeypatch.setattr("backend.layouts.OPTIMAL_SOLVER_MAX_IMAGES", 10)
    coords = np.random.default_rng(3).normal(0, 1, (40, 2))
    result = grid_mosaic(coords, algorithm="auto")
    assert result["algorithm"] == "split"
    assert "exceed" in result["note"]


def test_auto_uses_the_exact_solver_for_small_sets():
    if not scipy_available():
        pytest.skip("SciPy is not installed")
    coords = np.random.default_rng(4).normal(0, 1, (12, 2))
    result = grid_mosaic(coords, algorithm="auto")
    assert result["algorithm"] == "optimal"
    assert result["note"] == ""


def test_split_scales_past_the_exact_solver_limit():
    count = OPTIMAL_SOLVER_MAX_IMAGES + 500
    coords = np.random.default_rng(5).normal(0, 1, (count, 2))
    result = grid_mosaic(coords, algorithm="split")
    assert len(set(result["cells"])) == count


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="Unknown mosaic algorithm"):
        grid_mosaic(np.zeros((4, 2)), algorithm="som")


def test_non_2d_input_is_rejected():
    with pytest.raises(ValueError, match="2D coordinates"):
        grid_mosaic(np.zeros((4, 3)))
