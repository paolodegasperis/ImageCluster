"""Searching with several images, some of them unwanted.

The fixture plants three groups that are each coherent and mutually unrelated,
so every claim this feature makes has a knowable answer: a query built from one
group must return that group, a query that subtracts a group must push it down,
and a query built from images that agree about nothing must say so rather than
return a confident list.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend.image_query import MAX_QUERY_IMAGES, similar_to_set
from backend.runtime_dirs import OUTPUT_DIR

NAME = "_pytest_image_query"
REL = f"output/projections/{NAME}.tsv"
GROUPS = 3
PER_GROUP = 15
DIM = 48


@pytest.fixture
def planted(request):
    rng = np.random.default_rng(7)
    vectors: list[np.ndarray] = []
    meta: list[tuple[str, int]] = []
    for group in range(GROUPS):
        centre = rng.normal(0, 1, DIM)
        centre /= np.linalg.norm(centre)
        for index in range(PER_GROUP):
            vectors.append(centre + rng.normal(0, 0.05, DIM))
            meta.append((f"g{group}_{index:02d}.jpg", group + 1))

    # An exact pair: the only way to make a query cancel itself out, since an
    # image may not sit on both sides of one.
    twin = rng.normal(0, 1, DIM)
    twin /= np.linalg.norm(twin)
    for name in ("twin_a.jpg", "twin_b.jpg"):
        vectors.append(twin.copy())
        meta.append((name, GROUPS + 1))

    matrix = np.vstack(vectors).astype("float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "projections").mkdir(parents=True, exist_ok=True)
    embedding = OUTPUT_DIR / "embeddings" / f"{NAME}.npy"
    np.save(embedding, matrix)

    tsv = OUTPUT_DIR / "projections" / f"{NAME}.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "relative_path", "x", "y", "cluster"], delimiter="\t")
        writer.writeheader()
        for index, (filename, group) in enumerate(meta):
            writer.writerow({
                "filename": filename,
                "relative_path": f"img/{filename}",
                "x": float(matrix[index, 0] * 10),
                "y": float(matrix[index, 1] * 10),
                "cluster": group,
            })
    tsv.with_suffix(".json").write_text(json.dumps({
        "model_label": "Pytest model",
        "embedding_path": embedding.relative_to(OUTPUT_DIR.parent).as_posix(),
    }), encoding="utf-8")

    request.addfinalizer(lambda: [p.unlink(missing_ok=True) for p in (tsv, tsv.with_suffix(".json"), embedding)])
    return REL


def group_of(filename: str) -> str:
    return filename.split("_")[0]


def test_three_images_from_one_group_return_that_group(planted):
    result = similar_to_set(planted, positive_paths=[f"img/g0_{i:02d}.jpg" for i in range(3)], top_k=10)
    assert result["count"] == 10
    assert all(group_of(item["filename"]) == "g0" for item in result["results"])


def test_the_query_images_are_never_returned(planted):
    asked = [f"img/g1_{i:02d}.jpg" for i in range(3)]
    result = similar_to_set(planted, positive_paths=asked, top_k=PER_GROUP)
    returned = {f"img/{item['filename']}" for item in result["results"]}
    assert not returned & set(asked)


def test_agreement_separates_a_coherent_query_from_a_scattered_one(planted):
    coherent = similar_to_set(planted, positive_paths=[f"img/g0_{i:02d}.jpg" for i in range(3)], top_k=5)
    scattered = similar_to_set(planted, positive_paths=["img/g0_00.jpg", "img/g1_00.jpg", "img/g2_00.jpg"], top_k=5)
    # One group points one way; three unrelated directions average to roughly
    # 1/sqrt(3) of a unit vector, which is the shape of "these disagree".
    assert coherent["agreement"] > 0.9
    assert scattered["agreement"] < 0.7


def test_an_unwanted_group_is_pushed_down(planted):
    """The point of the minus sign: the same wanted images, with one group
    removed, must stop returning that group."""
    wanted = ["img/g0_00.jpg", "img/g1_00.jpg"]
    without = similar_to_set(planted, positive_paths=wanted, top_k=12)
    with_minus = similar_to_set(planted, positive_paths=wanted, negative_paths=["img/g1_01.jpg"], top_k=12)
    before = sum(group_of(i["filename"]) == "g1" for i in without["results"])
    after = sum(group_of(i["filename"]) == "g1" for i in with_minus["results"])
    assert after < before


def test_each_result_says_which_image_it_is_closest_to(planted):
    """With a query built from several, "why did this come back?" is a real
    question, and the answer costs one more matrix row."""
    result = similar_to_set(planted, positive_paths=["img/g0_00.jpg", "img/g2_00.jpg"], top_k=8)
    for item in result["results"]:
        assert item["closest_to"] in {"g0_00.jpg", "g2_00.jpg"}
        # The nearest chosen image must be in the result's own group.
        assert group_of(item["closest_to"]) == group_of(item["filename"])


def test_a_query_with_nothing_wanted_is_refused(planted):
    with pytest.raises(ValueError, match="at least one image"):
        similar_to_set(planted, negative_paths=["img/g0_00.jpg"])


def test_an_image_cannot_be_both_wanted_and_unwanted(planted):
    with pytest.raises(ValueError, match="both sides"):
        similar_to_set(planted, positive_paths=["img/g0_00.jpg"], negative_paths=["img/g0_00.jpg"])


def test_a_query_that_cancels_itself_out_says_so(planted):
    """Subtracting a group from itself leaves no direction to search in.
    Returning the nearest images to a near-zero vector would be returning noise
    with confident scores attached."""
    with pytest.raises(ValueError, match="cancel each other out"):
        similar_to_set(planted, positive_paths=["img/twin_a.jpg"], negative_paths=["img/twin_b.jpg"])


def test_too_many_images_are_refused_with_the_reason(planted):
    with pytest.raises(ValueError, match="limit is"):
        too_many = ([f"img/g0_{i:02d}.jpg" for i in range(PER_GROUP)]
                    + [f"img/g1_{i:02d}.jpg" for i in range(PER_GROUP)])
        assert len(too_many) > MAX_QUERY_IMAGES
        similar_to_set(planted, positive_paths=too_many)


def test_an_unknown_image_is_named_in_the_error(planted):
    with pytest.raises(ValueError, match="not part of this projection"):
        similar_to_set(planted, positive_paths=["img/nope.jpg"])
