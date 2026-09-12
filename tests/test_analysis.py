"""Similar images, duplicates, outliers and cluster summaries.

The fixture plants exactly what each analysis is supposed to find: a chain of
near-identical re-saves, two images unrelated to everything, and four clusters
with realistic internal similarity (~0.9, as a coherent CLIP group actually is).
An analysis that cannot find a planted answer is broken regardless of what it
returns for real data.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend.analysis import (
    MAX_ANALYSIS_IMAGES,
    cluster_summaries,
    duplicate_groups,
    outlier_ranking,
    similar_images,
)
from backend.runtime_dirs import OUTPUT_DIR

NAME = "_pytest_analysis"
REL = f"output/projections/{NAME}.tsv"
CLUSTERS = 4
PER_CLUSTER = 20
DUPLICATES = 3
OUTLIERS = 2


@pytest.fixture
def planted(request):
    rng = np.random.default_rng(5)
    dim = 64
    meta: list[tuple[str, int]] = []
    vectors: list[np.ndarray] = []

    for cluster in range(CLUSTERS):
        base = rng.normal(0, 1, dim)
        base /= np.linalg.norm(base)
        for index in range(PER_CLUSTER):
            vectors.append(base + rng.normal(0, 0.06, dim))
            meta.append((f"c{cluster}_{index:02d}.jpg", cluster + 1))

    # A re-save chain of the very first image.
    anchor = vectors[0].copy()
    for index in range(DUPLICATES):
        vectors.append(anchor + rng.normal(0, 0.004, dim))
        meta.append((f"dup_{index}.jpg", 1))

    # Two images unrelated to every cluster centre, filed under the last cluster.
    for index in range(OUTLIERS):
        stray = rng.normal(0, 1, dim)
        stray /= np.linalg.norm(stray)
        vectors.append(stray)
        meta.append((f"outlier_{index}.jpg", CLUSTERS))

    matrix = np.vstack(vectors).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "projections").mkdir(parents=True, exist_ok=True)
    embedding = OUTPUT_DIR / "embeddings" / f"{NAME}.npy"
    np.save(embedding, matrix)

    tsv = OUTPUT_DIR / "projections" / f"{NAME}.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "relative_path", "x", "y", "cluster"], delimiter="\t")
        writer.writeheader()
        for index, (filename, cluster) in enumerate(meta):
            writer.writerow({
                "filename": filename,
                "relative_path": f"img/{filename}",
                "x": float(matrix[index, 0] * 10),
                "y": float(matrix[index, 1] * 10),
                "cluster": cluster,
            })

    sidecar = tsv.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "model_label": "Pytest model",
        "embedding_path": embedding.relative_to(OUTPUT_DIR.parent).as_posix(),
    }), encoding="utf-8")

    def cleanup():
        for path in (tsv, sidecar, embedding):
            path.unlink(missing_ok=True)

    request.addfinalizer(cleanup)
    return {"rel": REL, "total": len(meta)}


# --------------------------------------------------------------------------
# 1.1 Similar images


def test_similar_finds_the_planted_duplicates_first(planted):
    result = similar_images(planted["rel"], relative_path="img/c0_00.jpg", top_k=5)
    top = [item["filename"] for item in result["results"][:DUPLICATES]]
    assert all(name.startswith("dup_") for name in top), top
    assert result["results"][0]["score"] > 0.99


def test_similar_never_returns_the_query_itself(planted):
    result = similar_images(planted["rel"], relative_path="img/c0_00.jpg", top_k=100)
    assert all(item["relative_path"] != "img/c0_00.jpg" for item in result["results"])
    assert result["query"]["filename"] == "c0_00.jpg"


def test_similar_results_are_ranked_and_numbered_consecutively(planted):
    result = similar_images(planted["rel"], relative_path="img/c1_00.jpg", top_k=10)
    scores = [item["score"] for item in result["results"]]
    assert scores == sorted(scores, reverse=True)
    assert [item["rank"] for item in result["results"]] == list(range(1, len(scores) + 1))


def test_similar_honours_top_k(planted):
    assert len(similar_images(planted["rel"], relative_path="img/c0_00.jpg", top_k=3)["results"]) == 3


def test_similar_honours_a_minimum_similarity(planted):
    result = similar_images(planted["rel"], relative_path="img/c0_00.jpg", top_k=100, min_similarity=0.95)
    assert result["results"]
    assert all(item["score"] >= 0.95 for item in result["results"])
    # Only the re-saves clear 0.95; ordinary cluster members sit near 0.9.
    assert len(result["results"]) == DUPLICATES


def test_similar_accepts_a_row_index_as_the_query(planted):
    by_index = similar_images(planted["rel"], index=0, top_k=3)
    by_path = similar_images(planted["rel"], relative_path="img/c0_00.jpg", top_k=3)
    assert by_index["query"]["filename"] == by_path["query"]["filename"]


def test_similar_rejects_an_unknown_query(planted):
    with pytest.raises(ValueError, match="not part of this projection"):
        similar_images(planted["rel"], relative_path="img/nope.jpg")
    with pytest.raises(ValueError, match="outside this projection"):
        similar_images(planted["rel"], index=99999)
    with pytest.raises(ValueError, match="either relative_path or index"):
        similar_images(planted["rel"])


# --------------------------------------------------------------------------
# 1.2 Duplicates


def test_duplicates_find_exactly_the_planted_chain(planted):
    result = duplicate_groups(planted["rel"], threshold=0.98)
    assert result["group_count"] == 1
    group = result["groups"][0]
    assert group["size"] == DUPLICATES + 1  # the chain plus the image it copies
    assert {item["filename"] for item in group["images"]} == {"c0_00.jpg", "dup_0.jpg", "dup_1.jpg", "dup_2.jpg"}
    assert result["redundant_image_count"] == DUPLICATES


def test_a_directly_matching_group_is_not_flagged_as_chained(planted):
    group = duplicate_groups(planted["rel"], threshold=0.98)["groups"][0]
    assert group["chained"] is False
    assert group["min_similarity"] >= 0.98


def test_exactly_one_member_is_offered_as_the_one_to_keep(planted):
    group = duplicate_groups(planted["rel"], threshold=0.98)["groups"][0]
    central = [item for item in group["images"] if item["most_central"]]
    assert len(central) == 1
    assert central[0]["rank"] == 1


def test_a_low_threshold_flags_its_groups_as_chained(planted):
    """The safeguard: at 0.80 whole clusters link up through intermediaries."""
    result = duplicate_groups(planted["rel"], threshold=0.80)
    assert result["group_count"] > 1
    assert result["chained_group_count"] == result["group_count"]
    assert all(group["min_similarity"] < 0.80 for group in result["groups"])


def test_groups_are_ordered_by_confidence_not_by_size(planted):
    """Ordering by size would put the longest, least certain chain first."""
    result = duplicate_groups(planted["rel"], threshold=0.80)
    means = [group["mean_similarity"] for group in result["groups"]]
    assert means == sorted(means, reverse=True)


def test_a_high_threshold_finds_nothing(planted):
    result = duplicate_groups(planted["rel"], threshold=0.99999)
    assert result["group_count"] == 0
    assert result["redundant_image_count"] == 0


def test_duplicate_threshold_is_validated(planted):
    for bad in [0.0, -0.5, 1.5]:
        with pytest.raises(ValueError, match="between 0 and 1"):
            duplicate_groups(planted["rel"], threshold=bad)


def test_the_pair_budget_is_reported_when_reached(planted):
    result = duplicate_groups(planted["rel"], threshold=0.5, max_pairs=100)
    assert result["truncated"] is True


# --------------------------------------------------------------------------
# 1.3 Outliers


def test_outliers_rank_the_planted_strays_first(planted):
    result = outlier_ranking(planted["rel"], k=5, top_k=5)
    assert {item["filename"] for item in result["results"][:OUTLIERS]} == {"outlier_0.jpg", "outlier_1.jpg"}


def test_outlier_scores_ascend_because_low_means_isolated(planted):
    result = outlier_ranking(planted["rel"], k=5, top_k=20)
    scores = [item["score"] for item in result["results"]]
    assert scores == sorted(scores)
    assert [item["rank"] for item in result["results"]] == list(range(1, len(scores) + 1))


def test_the_planted_strays_are_far_below_the_median(planted):
    result = outlier_ranking(planted["rel"], k=5, top_k=5)
    assert result["results"][0]["score"] < result["median_similarity"] / 2


def test_outliers_honour_top_k_and_cap_k(planted):
    assert len(outlier_ranking(planted["rel"], k=3, top_k=7)["results"]) == 7
    assert outlier_ranking(planted["rel"], k=999, top_k=3)["k"] <= 20


def test_outliers_need_a_few_images(tmp_path, planted):
    tiny = OUTPUT_DIR / "projections" / "_pytest_tiny.tsv"
    embedding = OUTPUT_DIR / "embeddings" / "_pytest_tiny.npy"
    np.save(embedding, np.ones((2, 4), dtype="float32"))
    with tiny.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "relative_path", "x", "y"], delimiter="\t")
        writer.writeheader()
        for index in range(2):
            writer.writerow({"filename": f"a{index}.jpg", "relative_path": f"img/a{index}.jpg", "x": 0, "y": 0})
    tiny.with_suffix(".json").write_text(
        json.dumps({"embedding_path": "output/embeddings/_pytest_tiny.npy"}), encoding="utf-8"
    )
    try:
        with pytest.raises(ValueError, match="At least three"):
            outlier_ranking("output/projections/_pytest_tiny.tsv")
    finally:
        tiny.unlink(missing_ok=True)
        tiny.with_suffix(".json").unlink(missing_ok=True)
        embedding.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 1.4 Cluster summaries


def test_every_cluster_is_summarised(planted):
    result = cluster_summaries(planted["rel"])
    assert result["cluster_count"] == CLUSTERS
    assert result["labelled"] == planted["total"]
    assert result["unlabelled"] == 0
    assert [cluster["cluster"] for cluster in result["clusters"]] == [1, 2, 3, 4]


def test_the_medoid_is_the_most_typical_member(planted):
    for cluster in cluster_summaries(planted["rel"])["clusters"]:
        assert cluster["medoid"]["is_medoid"] is True
        assert cluster["medoid"]["score"] == max(member["score"] for member in cluster["members"])
        assert cluster["members"][0] == cluster["medoid"]


def test_members_are_ordered_most_typical_first(planted):
    for cluster in cluster_summaries(planted["rel"])["clusters"]:
        scores = [member["score"] for member in cluster["members"]]
        assert scores == sorted(scores, reverse=True)
        assert sum(1 for member in cluster["members"] if member["is_medoid"]) == 1


def test_sizes_add_up_to_the_projection(planted):
    result = cluster_summaries(planted["rel"])
    assert sum(cluster["size"] for cluster in result["clusters"]) == planted["total"]


def test_the_cluster_holding_the_strays_is_the_least_cohesive(planted):
    cohesion = {cluster["cluster"]: cluster["cohesion"] for cluster in cluster_summaries(planted["rel"])["clusters"]}
    assert cohesion[CLUSTERS] < min(cohesion[label] for label in cohesion if label != CLUSTERS)


def test_spread_exposes_the_worst_member(planted):
    for cluster in cluster_summaries(planted["rel"])["clusters"]:
        assert cluster["spread"] == min(member["score"] for member in cluster["members"])


def test_a_projection_without_clusters_says_so():
    tsv = OUTPUT_DIR / "projections" / "_pytest_noclusters.tsv"
    embedding = OUTPUT_DIR / "embeddings" / "_pytest_noclusters.npy"
    np.save(embedding, np.eye(3, 4, dtype="float32"))
    with tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "relative_path", "x", "y"], delimiter="\t")
        writer.writeheader()
        for index in range(3):
            writer.writerow({"filename": f"n{index}.jpg", "relative_path": f"img/n{index}.jpg", "x": index, "y": index})
    tsv.with_suffix(".json").write_text(
        json.dumps({"embedding_path": "output/embeddings/_pytest_noclusters.npy"}), encoding="utf-8"
    )
    try:
        with pytest.raises(ValueError, match="no clusters"):
            cluster_summaries("output/projections/_pytest_noclusters.tsv")
    finally:
        tsv.unlink(missing_ok=True)
        tsv.with_suffix(".json").unlink(missing_ok=True)
        embedding.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Shared guarantees


def test_every_analysis_refuses_a_path_outside_projections():
    for call in (
        lambda: similar_images("config.json", index=0),
        lambda: duplicate_groups("config.json"),
        lambda: outlier_ranking("config.json"),
        lambda: cluster_summaries("config.json"),
    ):
        with pytest.raises(ValueError, match="inside output/projections"):
            call()


def test_every_result_item_shares_one_card_shape(planted):
    """One payload shape means one card component renders all four panels."""
    required = {"index", "filename", "relative_path", "x", "y", "cluster", "rank", "score"}
    samples = [
        similar_images(planted["rel"], index=0, top_k=1)["results"][0],
        duplicate_groups(planted["rel"], threshold=0.98)["groups"][0]["images"][0],
        outlier_ranking(planted["rel"], top_k=1)["results"][0],
        cluster_summaries(planted["rel"])["clusters"][0]["members"][0],
    ]
    for sample in samples:
        assert required <= set(sample), required - set(sample)


def test_the_size_guard_is_declared_well_above_the_graph_cap():
    from backend.neighbors import MAX_GRAPH_NODES

    assert MAX_ANALYSIS_IMAGES > MAX_GRAPH_NODES


# --------------------------------------------------------------------------
# Two-image comparison


def test_comparison_reports_a_near_identical_pair_as_mutual_first(planted):
    from backend.analysis import compare_images

    result = compare_images(planted["rel"], path_a="img/c0_00.jpg", path_b="img/dup_0.jpg")
    assert result["similarity"] > 0.99
    assert result["rank_of_b_in_a"] == 1
    assert result["rank_of_a_in_b"] == 1


def test_comparison_is_symmetric_in_its_similarity(planted):
    from backend.analysis import compare_images

    forward = compare_images(planted["rel"], path_a="img/c0_01.jpg", path_b="img/c1_02.jpg")
    backward = compare_images(planted["rel"], path_a="img/c1_02.jpg", path_b="img/c0_01.jpg")
    assert forward["similarity"] == backward["similarity"]
    # The ranks swap with the arguments.
    assert forward["rank_of_b_in_a"] == backward["rank_of_a_in_b"]
    assert forward["rank_of_a_in_b"] == backward["rank_of_b_in_a"]


def test_same_group_shares_more_neighbours_than_a_cross_group_pair(planted):
    from backend.analysis import compare_images

    same = compare_images(planted["rel"], path_a="img/c0_01.jpg", path_b="img/c0_05.jpg")
    cross = compare_images(planted["rel"], path_a="img/c0_01.jpg", path_b="img/c2_07.jpg")
    assert same["similarity"] > cross["similarity"]
    assert same["shared_count"] >= cross["shared_count"]
    assert same["same_cluster"] is True
    assert cross["same_cluster"] is False


def test_shared_neighbours_are_close_to_both_and_ranked(planted):
    from backend.analysis import compare_images

    result = compare_images(planted["rel"], path_a="img/c0_01.jpg", path_b="img/c0_05.jpg")
    assert result["shared_count"] > 0
    scores = [item["score"] for item in result["shared_neighbours"]]
    assert scores == sorted(scores, reverse=True)
    for item in result["shared_neighbours"]:
        # "score" is the weaker of the two similarities: it belongs to both.
        assert item["score"] == pytest.approx(min(item["similarity_to_a"], item["similarity_to_b"]))
        assert item["relative_path"] not in {"img/c0_01.jpg", "img/c0_05.jpg"}
    assert [item["rank"] for item in result["shared_neighbours"]] == list(range(1, len(scores) + 1))


def test_comparison_refuses_an_image_against_itself(planted):
    from backend.analysis import compare_images

    with pytest.raises(ValueError, match="two different images"):
        compare_images(planted["rel"], path_a="img/c0_00.jpg", path_b="img/c0_00.jpg")


def test_comparison_refuses_an_unknown_image(planted):
    from backend.analysis import compare_images

    with pytest.raises(ValueError, match="not part of this projection"):
        compare_images(planted["rel"], path_a="img/c0_00.jpg", path_b="img/ghost.jpg")


def test_percentile_places_a_strong_pair_near_the_top(planted):
    from backend.analysis import compare_images

    twins = compare_images(planted["rel"], path_a="img/c0_00.jpg", path_b="img/dup_0.jpg")
    cross = compare_images(planted["rel"], path_a="img/c0_00.jpg", path_b="img/c2_07.jpg")
    assert twins["percentile_in_a"] > cross["percentile_in_a"]
    assert twins["similarity"] > twins["median_similarity"]
