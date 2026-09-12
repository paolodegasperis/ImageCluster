"""Comparing two models over the same images.

No model is loaded: the fixture writes two projections whose embeddings are
constructed so the expected neighbourhoods are known. That is the only way to
assert what "disagreement" should be — with real models the answer is whatever
they happen to think.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend import disagreement
from backend.runtime_dirs import OUTPUT_DIR

COUNT = 12


def _write(name: str, vectors: np.ndarray, *, paths: list[str], model: str) -> str:
    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "projections").mkdir(parents=True, exist_ok=True)
    embedding = OUTPUT_DIR / "embeddings" / f"{name}.npy"
    np.save(embedding, vectors.astype("float32"))

    tsv = OUTPUT_DIR / "projections" / f"{name}.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "relative_path", "x", "y", "cluster"], delimiter="\t")
        writer.writeheader()
        for index, rel in enumerate(paths):
            writer.writerow({
                "filename": rel.rsplit("/", 1)[-1],
                "relative_path": rel,
                "x": index,
                "y": -index,
                "cluster": index % 3,
            })
    tsv.with_suffix(".json").write_text(json.dumps({
        "model_key": model,
        "model_label": model.upper(),
        "embedding_path": embedding.relative_to(OUTPUT_DIR.parent).as_posix(),
    }), encoding="utf-8")
    return f"output/projections/{name}.tsv"


@pytest.fixture
def two_models(request):
    rng = np.random.default_rng(7)
    paths = [f"img/{index:02d}.jpg" for index in range(COUNT)]
    a = rng.normal(size=(COUNT, 16))
    b = rng.normal(size=(COUNT, 16))

    names = ["_pytest_dis_a", "_pytest_dis_b", "_pytest_dis_same", "_pytest_dis_other"]

    def cleanup():
        for name in names:
            (OUTPUT_DIR / "projections" / f"{name}.tsv").unlink(missing_ok=True)
            (OUTPUT_DIR / "projections" / f"{name}.json").unlink(missing_ok=True)
            (OUTPUT_DIR / "embeddings" / f"{name}.npy").unlink(missing_ok=True)

    request.addfinalizer(cleanup)
    return {
        "paths": paths,
        "a": _write(names[0], a, paths=paths, model="openclip_vit_b_32"),
        "b": _write(names[1], b, paths=paths, model="dinov2_base"),
        # The same embeddings under a second name: the two must agree completely.
        "same": _write(names[2], a, paths=paths, model="openclip_vit_b_32"),
        "write": _write,
        "names": names,
        "vectors_a": a,
    }


def test_identical_embeddings_agree_completely(two_models):
    result = disagreement.model_disagreement(two_models["a"], two_models["same"], neighbours=5)
    assert result["mean_agreement"] == 1.0
    assert result["full_agreement"] == COUNT
    assert result["total_disagreement"] == 0


def test_unrelated_embeddings_agree_far_less(two_models):
    result = disagreement.model_disagreement(two_models["a"], two_models["b"], neighbours=5)
    assert 0.0 <= result["mean_agreement"] < 0.8


def test_every_image_is_scored_and_ranked_least_agreement_first(two_models):
    result = disagreement.model_disagreement(two_models["a"], two_models["b"], neighbours=5, top_k=COUNT)
    assert result["count"] == COUNT
    assert len(result["agreement"]) == COUNT
    scores = [item["score"] for item in result["results"]]
    assert scores == sorted(scores)
    assert [item["rank"] for item in result["results"]] == list(range(1, COUNT + 1))


def test_shared_neighbours_is_consistent_with_the_score(two_models):
    result = disagreement.model_disagreement(two_models["a"], two_models["b"], neighbours=5, top_k=COUNT)
    for item in result["results"]:
        assert item["score"] == pytest.approx(item["shared_neighbours"] / result["neighbours"], abs=1e-5)


def test_the_two_model_labels_are_reported(two_models):
    result = disagreement.model_disagreement(two_models["a"], two_models["b"], neighbours=5)
    assert result["model_a"] == "OPENCLIP_VIT_B_32"
    assert result["model_b"] == "DINOV2_BASE"


def test_comparing_a_projection_with_itself_is_refused(two_models):
    with pytest.raises(ValueError, match="two different projections"):
        disagreement.model_disagreement(two_models["a"], two_models["a"])


def test_different_image_sets_are_refused(two_models):
    other_paths = [f"img/x{index:02d}.jpg" for index in range(COUNT)]
    other = two_models["write"](
        two_models["names"][3], two_models["vectors_a"], paths=other_paths, model="dinov2_base",
    )
    with pytest.raises(ValueError, match="different images"):
        disagreement.model_disagreement(two_models["a"], other)


def test_the_same_images_in_a_different_order_are_refused(two_models):
    """Row i of one file would be compared with a different image in the other."""
    shuffled = list(reversed(two_models["paths"]))
    other = two_models["write"](
        two_models["names"][3], two_models["vectors_a"], paths=shuffled, model="dinov2_base",
    )
    with pytest.raises(ValueError, match="different order"):
        disagreement.model_disagreement(two_models["a"], other)


def test_the_neighbour_count_is_clamped_to_what_the_collection_allows(two_models):
    result = disagreement.model_disagreement(two_models["a"], two_models["b"], neighbours=999)
    assert result["neighbours"] <= COUNT - 1


def test_an_image_is_never_its_own_neighbour(two_models):
    sets = disagreement._neighbour_sets(two_models["vectors_a"], 4)
    assert all(index not in neighbours for index, neighbours in enumerate(sets))
    assert all(len(neighbours) == 4 for neighbours in sets)


def test_a_non_finite_row_does_not_poison_the_comparison(two_models):
    """One broken vector must not silently change everyone else's neighbours."""
    broken = two_models["vectors_a"].copy()
    broken[3] = np.nan
    sets = disagreement._neighbour_sets(broken, 4)
    assert len(sets) == COUNT
    assert all(len(neighbours) == 4 for neighbours in sets)
