"""Guard rails around projection and clustering.

The reducers are the part of the pipeline most likely to be handed degenerate
input (one image, two identical images, a feature axis that does not exist), and
each of those used to reach scikit-learn as an opaque exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.encoders.visual_features_encoder import FEATURE_NAMES, feature_column_names
from backend.reducers import (
    cluster_feature_matrix,
    cluster_projection,
    project_visual_features,
    reduce_embeddings,
)


def test_reduce_embeddings_rejects_single_row():
    with pytest.raises(ValueError, match="At least two embeddings"):
        reduce_embeddings(np.zeros((1, 8), dtype="float32"), "umap")


def test_reduce_embeddings_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="At least two embeddings"):
        reduce_embeddings(np.zeros(8, dtype="float32"), "umap")


def test_reduce_embeddings_rejects_unknown_reducer():
    with pytest.raises(ValueError, match="Unsupported reducer"):
        reduce_embeddings(np.random.rand(10, 8).astype("float32"), "pca")


def test_cluster_projection_rejects_non_2d_coordinates():
    with pytest.raises(ValueError, match="two-dimensional projection"):
        cluster_projection(np.random.rand(10, 3).astype("float32"))


def test_cluster_projection_rejects_too_few_points():
    with pytest.raises(ValueError, match="At least three"):
        cluster_projection(np.random.rand(2, 2).astype("float32"))


def test_cluster_projection_labels_are_one_based_and_complete():
    rng = np.random.default_rng(0)
    coords = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))]).astype("float32")
    result = cluster_projection(coords, auto=False, fixed_k=2)
    assert result["k"] == 2
    assert len(result["labels"]) == 40
    assert min(result["labels"]) == 1
    assert set(result["labels"]) == {1, 2}


def test_cluster_projection_auto_k_stays_inside_the_requested_range():
    rng = np.random.default_rng(1)
    coords = rng.normal(0, 1, (30, 2)).astype("float32")
    result = cluster_projection(coords, auto=True, min_k=2, max_k=5)
    assert 2 <= result["k"] <= 5
    assert result["auto"] is True


def test_cluster_projection_caps_k_below_the_sample_count():
    coords = np.random.rand(4, 2).astype("float32")
    result = cluster_projection(coords, auto=False, fixed_k=50)
    assert result["k"] < 4


def test_cluster_feature_matrix_rejects_too_few_rows():
    with pytest.raises(ValueError, match="At least three"):
        cluster_feature_matrix(np.random.rand(2, 12).astype("float32"))


def test_cluster_feature_matrix_runs_on_wide_feature_vectors():
    rng = np.random.default_rng(2)
    matrix = np.vstack([rng.normal(0, 0.1, (15, 12)), rng.normal(4, 0.1, (15, 12))]).astype("float32")
    result = cluster_feature_matrix(matrix, auto=False, fixed_k=2)
    assert result["method"] == "kmeans_features"
    assert set(result["labels"]) == {1, 2}


def test_project_visual_features_rejects_unknown_axis():
    names = feature_column_names()
    matrix = np.random.rand(5, len(names)).astype("float32")
    with pytest.raises(ValueError, match="Unknown visual-feature axis"):
        project_visual_features(matrix, names, "not_a_feature", "brightness")


def test_project_visual_features_is_deterministic_and_bounded():
    names = feature_column_names()
    matrix = np.zeros((3, len(names)), dtype="float32")
    matrix[:, names.index("brightness")] = [0.0, 127.5, 255.0]
    matrix[:, names.index("hue")] = [0.0, 90.0, 180.0]
    coords = project_visual_features(matrix, names, "hue", "brightness")
    assert coords.shape == (3, 2)
    # Fixed ranges mean the same image always lands on the same coordinates.
    assert coords[0].tolist() == [0.0, 0.0]
    assert coords[2].tolist() == [100.0, 100.0]
    assert coords.min() >= 0.0 and coords.max() <= 100.0


def test_project_visual_features_clips_out_of_range_values():
    names = feature_column_names()
    matrix = np.zeros((2, len(names)), dtype="float32")
    matrix[:, names.index("brightness")] = [-50.0, 1000.0]
    coords = project_visual_features(matrix, names, "brightness", "brightness")
    assert coords[0][0] == 0.0
    assert coords[1][0] == 100.0


def test_feature_column_names_start_with_the_named_scalars():
    names = feature_column_names()
    assert names[: len(FEATURE_NAMES)] == FEATURE_NAMES
    assert len(names) > len(FEATURE_NAMES)
