"""The shared text-image space: axes, concept scores, distinguish, arithmetic.

The text encoder is stubbed. A real one would make these tests slow, require
torch, and — worse — make them assertions about what CLIP happens to think rather
than about whether this code does the arithmetic correctly. The stub maps each
phrase to a known direction, so every expected answer is exact.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from backend import semantic
from backend.runtime_dirs import OUTPUT_DIR

NAME = "_pytest_semantic"
REL = f"output/projections/{NAME}.tsv"
DIM = 8

# Four orthogonal directions standing in for concepts. An image built from them
# has an exactly predictable score against each.
AXES = {
    "portrait": 0,
    "landscape": 1,
    "night": 2,
    "day": 3,
}


def _vector(weights: dict[str, float]) -> np.ndarray:
    vector = np.zeros(DIM, dtype="float32")
    for name, weight in weights.items():
        vector[AXES[name]] = weight
    return vector


@pytest.fixture
def stub_space(request, monkeypatch):
    """A projection whose images are known mixtures of four named directions."""
    recipes = [
        ("portrait_day", {"portrait": 1.0, "day": 0.3}),
        ("portrait_night", {"portrait": 1.0, "night": 0.3}),
        ("portrait_plain", {"portrait": 1.0}),
        ("landscape_day", {"landscape": 1.0, "day": 0.3}),
        ("landscape_night", {"landscape": 1.0, "night": 0.3}),
        ("landscape_plain", {"landscape": 1.0}),
    ]
    vectors = np.vstack([_vector(w) for _, w in recipes]).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    (OUTPUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "projections").mkdir(parents=True, exist_ok=True)
    embedding = OUTPUT_DIR / "embeddings" / f"{NAME}.npy"
    np.save(embedding, vectors)

    tsv = OUTPUT_DIR / "projections" / f"{NAME}.tsv"
    with tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "relative_path", "x", "y", "cluster"], delimiter="\t")
        writer.writeheader()
        for index, (name, _) in enumerate(recipes):
            writer.writerow({
                "filename": f"{name}.jpg",
                "relative_path": f"img/{name}.jpg",
                "x": index,
                "y": index,
                "cluster": 1 if index < 3 else 2,
            })

    sidecar = tsv.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "model_key": "openclip_vit_b_32",
        "model_label": "Stub",
        "embedding_path": embedding.relative_to(OUTPUT_DIR.parent).as_posix(),
    }), encoding="utf-8")

    def fake_encode_texts(texts, spec, batch_size=32):
        """Each known word maps to its axis; an unknown phrase is orthogonal noise."""
        out = []
        for text in texts:
            lowered = text.lower()
            weights = {name: 1.0 for name in AXES if name in lowered}
            vector = _vector(weights) if weights else np.full(DIM, 0.01, dtype="float32")
            out.append(vector)
        matrix = np.vstack(out).astype("float32")
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    monkeypatch.setattr(semantic, "encode_texts", fake_encode_texts)

    def cleanup():
        for path in (tsv, sidecar, embedding):
            path.unlink(missing_ok=True)

    request.addfinalizer(cleanup)
    return {"rel": REL, "names": [name for name, _ in recipes]}


# --------------------------------------------------------------------------
# Guards


def test_a_projection_without_a_model_key_is_refused(stub_space):
    tsv = OUTPUT_DIR / "projections" / f"{NAME}.tsv"
    tsv.with_suffix(".json").write_text(json.dumps({
        "embedding_path": f"output/embeddings/{NAME}.npy",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not record which model"):
        semantic.concept_scores(stub_space["rel"], phrases=["portrait"])


def test_an_image_only_model_is_refused(stub_space):
    tsv = OUTPUT_DIR / "projections" / f"{NAME}.tsv"
    tsv.with_suffix(".json").write_text(json.dumps({
        "model_key": "dinov2_base",
        "embedding_path": f"output/embeddings/{NAME}.npy",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="encodes images but not text"):
        semantic.concept_scores(stub_space["rel"], phrases=["portrait"])


# --------------------------------------------------------------------------
# 1.2 Concept intensity


def test_concept_scores_rank_the_matching_images_highest(stub_space):
    result = semantic.concept_scores(stub_space["rel"], phrases=["landscape"])
    assert result["count"] == 6
    top = [item["filename"] for item in result["top"][:3]]
    assert all(name.startswith("landscape") for name in top), top


def test_concept_scores_cover_every_image(stub_space):
    result = semantic.concept_scores(stub_space["rel"], phrases=["portrait"])
    assert len(result["scores"]) == 6
    assert all(item["score"] is not None for item in result["scores"])


def test_the_colour_ramp_is_narrower_than_the_full_range(stub_space):
    """Percentile bounds keep one outlier from flattening the ramp."""
    result = semantic.concept_scores(stub_space["rel"], phrases=["portrait"])
    assert result["min"] <= result["ramp_low"]
    assert result["ramp_high"] <= result["max"]


def test_concept_scores_need_a_phrase(stub_space):
    with pytest.raises(ValueError, match="Enter a concept"):
        semantic.concept_scores(stub_space["rel"], phrases=["   "])


# --------------------------------------------------------------------------
# 1.1 Semantic axes


def test_an_axis_points_from_one_pole_to_the_other(stub_space):
    spec, _ = semantic.text_model_for(stub_space["rel"])
    axis = semantic.semantic_axis(spec, ["portrait"], ["landscape"])
    assert axis[AXES["portrait"]] > 0
    assert axis[AXES["landscape"]] < 0
    assert float(np.linalg.norm(axis)) == pytest.approx(1.0, abs=1e-5)


def test_an_axis_between_identical_poles_is_refused(stub_space):
    spec, _ = semantic.text_model_for(stub_space["rel"])
    with pytest.raises(ValueError, match="same thing"):
        semantic.semantic_axis(spec, ["portrait"], ["portrait"])


def test_an_axis_needs_both_ends(stub_space):
    spec, _ = semantic.text_model_for(stub_space["rel"])
    with pytest.raises(ValueError, match="phrase at each end"):
        semantic.semantic_axis(spec, ["portrait"], [])


def test_the_preview_orders_images_along_the_named_axis(stub_space):
    result = semantic.semantic_axis_preview(
        stub_space["rel"],
        x_positive=["portrait"], x_negative=["landscape"],
        y_positive=["night"], y_negative=["day"],
    )
    assert result["count"] == 6
    by_x = sorted(result["points"], key=lambda p: p["x"])
    assert by_x[0]["filename"].startswith("landscape")
    assert by_x[-1]["filename"].startswith("portrait")
    by_y = sorted(result["points"], key=lambda p: p["y"])
    assert "day" in by_y[0]["filename"]
    assert "night" in by_y[-1]["filename"]


def test_orthogonal_axes_are_reported_as_independent(stub_space):
    result = semantic.semantic_axis_preview(
        stub_space["rel"],
        x_positive=["portrait"], x_negative=["landscape"],
        y_positive=["night"], y_negative=["day"],
    )
    assert result["independent"] is True
    assert abs(result["axis_overlap"]) < 0.1


def test_overlapping_axes_are_flagged(stub_space):
    """Two axes measuring the same thing collapse the plot onto a diagonal."""
    result = semantic.semantic_axis_preview(
        stub_space["rel"],
        x_positive=["portrait"], x_negative=["landscape"],
        y_positive=["portrait"], y_negative=["night"],
    )
    assert result["independent"] is False or result["axis_overlap"] > 0.3


# --------------------------------------------------------------------------
# 1.4 Distinguish


def test_distinguish_names_what_one_group_has_and_the_other_lacks(stub_space):
    portraits = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("portrait")]
    landscapes = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("landscape")]
    result = semantic.distinguish(
        stub_space["rel"],
        paths_a=portraits, paths_b=landscapes,
        vocabulary=["portrait", "landscape", "night", "day"],
        top_k=4,
    )
    assert result["towards_a"][0]["term"] == "portrait"
    assert result["towards_b"][0]["term"] == "landscape"
    assert result["size_a"] == 3
    assert result["size_b"] == 3


def test_distinguish_reports_how_far_apart_the_groups_are(stub_space):
    portraits = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("portrait")]
    landscapes = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("landscape")]
    result = semantic.distinguish(stub_space["rel"], paths_a=portraits, paths_b=landscapes)
    # Orthogonal groups: their centroids should be nearly unrelated.
    assert result["centroid_similarity"] < 0.5


def test_distinguish_refuses_two_identical_groups(stub_space):
    same = [f"img/{stub_space['names'][0]}.jpg"]
    with pytest.raises(ValueError, match="identical"):
        semantic.distinguish(stub_space["rel"], paths_a=same, paths_b=same)


def test_distinguish_refuses_a_group_with_no_known_image(stub_space):
    with pytest.raises(ValueError, match="Group A contains no image"):
        semantic.distinguish(
            stub_space["rel"],
            paths_a=["img/ghost.jpg"],
            paths_b=[f"img/{stub_space['names'][0]}.jpg"],
        )


def test_distinguish_uses_the_supplied_vocabulary_only(stub_space):
    portraits = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("portrait")]
    landscapes = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("landscape")]
    result = semantic.distinguish(
        stub_space["rel"], paths_a=portraits, paths_b=landscapes,
        vocabulary=["night", "day"], top_k=2,
    )
    assert result["vocabulary_size"] == 2
    reported = {t["term"] for t in result["towards_a"]} | {t["term"] for t in result["towards_b"]}
    assert reported <= {"night", "day"}


def test_the_default_vocabulary_is_offered_when_none_is_given(stub_space):
    portraits = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("portrait")]
    landscapes = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("landscape")]
    result = semantic.distinguish(stub_space["rel"], paths_a=portraits, paths_b=landscapes)
    assert result["vocabulary_size"] == len(semantic.DEFAULT_VOCABULARY)


# --------------------------------------------------------------------------
# 1.6 Concept arithmetic


def test_arithmetic_moves_the_query_towards_the_added_term(stub_space):
    """portrait_day - day + night should land on portrait_night."""
    result = semantic.concept_arithmetic(
        stub_space["rel"],
        relative_path="img/portrait_day.jpg",
        add=["night"], subtract=["day"], top_k=3,
    )
    assert result["results"][0]["filename"] == "portrait_night.jpg"


def test_arithmetic_never_returns_the_starting_image(stub_space):
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["night"], top_k=10,
    )
    assert all(item["relative_path"] != "img/portrait_day.jpg" for item in result["results"])


def test_arithmetic_can_start_from_a_phrase_alone(stub_space):
    result = semantic.concept_arithmetic(stub_space["rel"], base_phrases=["landscape"], top_k=3)
    assert result["results"][0]["filename"].startswith("landscape")
    assert result["base_image"] is None


def test_arithmetic_needs_a_starting_point(stub_space):
    with pytest.raises(ValueError, match="Start from an image or a phrase"):
        semantic.concept_arithmetic(stub_space["rel"], add=[])


def test_arithmetic_rejects_an_unknown_image(stub_space):
    with pytest.raises(ValueError, match="not part of this projection"):
        semantic.concept_arithmetic(stub_space["rel"], relative_path="img/ghost.jpg", add=["night"])


def test_arithmetic_reports_the_expression_it_evaluated(stub_space):
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["night"], subtract=["day"],
    )
    assert "night" in result["expression"]
    assert "day" in result["expression"]


# --------------------------------------------------------------------------
# Shared helpers


def test_phrases_are_trimmed_deduplicated_and_capped():
    cleaned = semantic._clean_terms(["  a  ", "a", "A", "", "b"], limit=10)
    assert cleaned == ["a", "b"]
    assert len(semantic._clean_terms([str(i) for i in range(100)], limit=5)) == 5


def test_several_phrasings_average_into_one_direction(stub_space):
    """Averaging before normalising is what makes a pole robust to wording."""
    spec, _ = semantic.text_model_for(stub_space["rel"])
    single = semantic._pole(spec, ["portrait"])
    doubled = semantic._pole(spec, ["portrait", "a portrait"])
    assert float(np.dot(single, doubled)) > 0.99


# --------------------------------------------------------------------------
# 1.6 regression: the modality gap
#
# The first implementation added the phrase vector to the image vector. CLIP puts
# images and text in two separate cones - an image sits ~0.2 from every phrase
# whatever either depicts, while image-to-image similarity reaches 0.8 - so the
# phrase never moved the ranking and every query returned the base image's own
# neighbours. These tests fail against that implementation.


def test_different_added_concepts_give_different_answers(stub_space):
    night = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", add=["night"], top_k=5,
    )
    day = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", add=["day"], top_k=5,
    )
    assert [r["filename"] for r in night["results"]] != [r["filename"] for r in day["results"]]


def test_adding_a_concept_changes_the_plain_neighbour_order(stub_space):
    plain = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", base_phrases=None,
        add=[], subtract=[], strength=1.0, top_k=5,
    )
    with_concept = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", add=["landscape"], top_k=5,
    )
    assert [r["filename"] for r in plain["results"]] != [r["filename"] for r in with_concept["results"]]


def test_strength_zero_ignores_the_concepts_entirely(stub_space):
    """At 0 the phrases must not move the ranking at all - that is what 0 means."""
    neutral = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", add=[], top_k=6,
    )
    silenced = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg",
        add=["landscape"], strength=0.0, top_k=6,
    )
    assert [r["filename"] for r in neutral["results"]] == [r["filename"] for r in silenced["results"]]


def test_raising_the_strength_pulls_towards_the_concept(stub_space):
    weak = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg",
        add=["landscape"], strength=0.25, top_k=6,
    )
    strong = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg",
        add=["landscape"], strength=5.0, top_k=6,
    )
    # At high strength the landscapes must lead; at low strength the portraits do.
    assert strong["results"][0]["filename"].startswith("landscape")
    assert weak["results"][0]["filename"].startswith("portrait")


def test_subtracting_is_the_opposite_of_adding(stub_space):
    added = semantic.concept_arithmetic(
        stub_space["rel"], base_phrases=["portrait"], add=["night"], strength=2.0, top_k=6,
    )
    removed = semantic.concept_arithmetic(
        stub_space["rel"], base_phrases=["portrait"], subtract=["night"], strength=2.0, top_k=6,
    )
    assert added["results"][0]["filename"] != removed["results"][0]["filename"]
    assert "night" in added["results"][0]["filename"]


def test_the_strength_is_clamped_to_a_sane_range(stub_space):
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", add=["night"], strength=999,
    )
    assert result["strength"] <= 5.0


def test_each_term_is_reported_with_its_sign_and_weight(stub_space):
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg",
        add=["night"], subtract=["day"], strength=2.0,
    )
    signs = {c["term"]: c["sign"] for c in result["components"]}
    assert signs["night"] == 1
    assert signs["day"] == -1
    assert any(c["term"] == "portrait_day.jpg" for c in result["components"])


def test_a_flat_score_column_contributes_nothing_instead_of_dividing_by_zero():
    flat = np.full(8, 0.5, dtype="float32")
    assert np.allclose(semantic._standardise(flat), 0.0)


# --------------------------------------------------------------------------
# Regression: the arithmetic used to ignore the terms entirely.
#
# Images and text occupy separate regions of a CLIP space, and image-image
# cosines are far more concentrated than image-text ones. Summing the raw
# vectors therefore let the starting image outvote every phrase, and the answer
# was its plain neighbour list whatever you typed. Measured on a real 165-image
# collection: only 24% of results differed from those neighbours. Combining
# standardised score columns instead removes the asymmetry - 71% now differ.


def test_different_terms_give_different_answers(stub_space):
    """The defect this guards: every query returned the same images."""
    night = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["night"], top_k=4)
    landscape = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["landscape"], top_k=4)
    assert night["results"][0]["filename"] != landscape["results"][0]["filename"]


def test_the_added_term_outranks_the_starting_image_s_own_neighbours(stub_space):
    """Asking for landscape from a portrait must actually reach the landscapes."""
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg",
        add=["landscape"], subtract=["portrait"], top_k=3)
    assert all(item["filename"].startswith("landscape") for item in result["results"])


def test_strength_zero_returns_the_plain_neighbours(stub_space):
    """The 'off' position has to be reachable, and has to mean something."""
    off = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg",
        add=["landscape"], strength=0, top_k=3)
    # With the term switched off, the portraits - the base image's own kind -
    # must come back, not the landscapes it asked for.
    assert off["results"][0]["filename"].startswith("portrait")


def test_raising_strength_moves_further_from_the_starting_image(stub_space):
    weak = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["landscape"],
        strength=0, top_k=5)
    strong = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["landscape"],
        strength=5, top_k=5)
    weak_names = [i["filename"] for i in weak["results"]]
    strong_names = [i["filename"] for i in strong["results"]]
    assert weak_names != strong_names


def test_strength_is_clamped_and_reported(stub_space):
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg", add=["night"], strength=999)
    assert result["strength"] <= 5.0


def test_every_term_is_reported_as_a_component(stub_space):
    """The UI shows what was combined; the caller must be able to see it."""
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_day.jpg",
        add=["night"], subtract=["day"], top_k=3)
    terms = {c["term"]: c["sign"] for c in result["components"]}
    assert terms.get("night") == 1
    assert terms.get("day") == -1


def test_standardise_puts_a_column_in_units_of_its_own_spread():
    values = np.array([0.20, 0.21, 0.22, 0.23, 0.24], dtype="float32")
    out = semantic._standardise(values)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.std()) == pytest.approx(1.0, abs=1e-3)


def test_standardise_survives_a_column_with_no_spread():
    """A term every image scores identically must not divide by zero."""
    out = semantic._standardise(np.full(6, 0.3, dtype="float32"))
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# One corrupt embedding must not poison the answers
#
# Every guard in this module compares a magnitude against a threshold, and any
# comparison against NaN is false. So a single non-finite row used to slip past
# `if length < 1e-9`, and the caller received a confidently ordered list
# computed from nothing at all.


def _corrupt_one_vector(index: int = 0):
    embedding = OUTPUT_DIR / "embeddings" / f"{NAME}.npy"
    vectors = np.load(embedding)
    vectors[index] = np.nan
    np.save(embedding, vectors)


def test_a_nan_row_does_not_wreck_the_concept_scores(stub_space):
    _corrupt_one_vector()
    result = semantic.concept_scores(stub_space["rel"], phrases=["landscape"])
    assert result["count"] == 6
    finite = [item["score"] for item in result["scores"] if item["score"] is not None]
    assert len(finite) == 6
    # The landscapes must still lead: the corrupt row scores 0, not NaN.
    assert result["top"][0]["filename"].startswith("landscape")


def test_a_nan_row_does_not_silently_empty_the_arithmetic(stub_space):
    """It used to return count 0 with no explanation - 'nothing matched'."""
    _corrupt_one_vector()
    result = semantic.concept_arithmetic(
        stub_space["rel"], relative_path="img/portrait_plain.jpg", add=["landscape"], top_k=5)
    assert result["count"] > 0
    assert all(np.isfinite(item["score"]) for item in result["results"])


def test_a_nan_row_does_not_produce_a_ranking_from_nothing(stub_space):
    _corrupt_one_vector()
    portraits = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("portrait")]
    landscapes = [f"img/{n}.jpg" for n in stub_space["names"] if n.startswith("landscape")]
    result = semantic.distinguish(
        stub_space["rel"], paths_a=portraits, paths_b=landscapes,
        vocabulary=["portrait", "landscape", "night", "day"], top_k=4)
    assert np.isfinite(result["centroid_similarity"])
    assert all(np.isfinite(term["score"]) for term in result["towards_a"])
    assert all(np.isfinite(term["score"]) for term in result["towards_b"])


def test_a_nan_row_keeps_the_axis_preview_plottable(stub_space):
    _corrupt_one_vector()
    result = semantic.semantic_axis_preview(
        stub_space["rel"],
        x_positive=["portrait"], x_negative=["landscape"],
        y_positive=["night"], y_negative=["day"])
    assert result["count"] == 6
    assert all(np.isfinite(p["x"]) and np.isfinite(p["y"]) for p in result["points"])
