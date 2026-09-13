"""Uses of the shared text-image space that go beyond search.

A CLIP-like model is not a feature extractor: images and words live in the same
space with the same coordinates. Search uses one operation from that dictionary —
"which images are nearest this phrase". The functions here use the rest of it.

``concept_scores``
    Not a ranking but a *distribution*: every image scored against one phrase, so
    the plot can be shaded by how much of a concept each image carries. "Where do
    ruins occur in this collection" is a different question from "show me the top
    thirty", and the answer is a gradient rather than a list.

``semantic_axis``
    Two phrases define a direction. Projecting the collection onto it gives an
    axis with a name you chose, instead of "the direction the algorithm happened
    to flatten along".

``distinguish``
    The difference between two groups' centroids, read back out as words. Not
    "these groups differ" — visible already — but *what* the difference is, in
    language.

``concept_arithmetic``
    An image, minus a phrase, plus another. The most direct way to show what an
    embedding space actually is.

All four need a text-capable model, and all four load it: unlike the analyses in
analysis.py, a phrase has to be encoded before anything can be compared with it.
The model cache means that cost is paid once per session.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .encoders.base import encode_texts
from .encoders.registry import ModelSpec, get_model_spec
from .neighbors import _sanitize
from .runtime_dirs import ROOT
from .views import load_projection, projection_sidecar

# Terms used when the caller supplies no vocabulary of its own. Chosen to span
# subject, setting, technique and mood rather than to be exhaustive: the point is
# to give `distinguish` enough axes to find a real difference along.
DEFAULT_VOCABULARY: tuple[str, ...] = (
    # subject
    "a portrait", "a group of people", "a single figure", "a crowd", "a nude",
    "a landscape", "a seascape", "a cityscape", "an architectural view",
    "a still life", "flowers", "an animal", "a horse", "a religious scene",
    "a mythological scene", "a battle", "a domestic interior", "a street scene",
    "a ruin", "a machine", "an abstract composition", "a map or diagram",
    # setting and light
    "an outdoor scene", "an indoor scene", "daylight", "night", "candlelight",
    "a storm", "snow", "fog", "a sunset",
    # technique and medium
    "an oil painting", "a watercolour", "a drawing", "an engraving", "an etching",
    "a photograph", "a sculpture", "a fresco", "a sketch",
    # form and mood
    "a detailed rendering", "a loose sketchy style", "strong contrast",
    "muted colours", "vivid colours", "a symmetrical composition",
    "a calm scene", "a dramatic scene", "a melancholy mood", "a festive scene",
    "an empty scene", "a busy composition",
)

MAX_CONCEPT_TERMS = 12
MAX_VOCABULARY_TERMS = 300


def _unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise, with non-finite rows zeroed first.

    One corrupt embedding otherwise poisons everything downstream in silence:
    NaN propagates through a centroid, and every guard in this module tests a
    magnitude against a threshold, which is false for NaN. The caller then gets
    an ordering computed from nothing rather than an error.
    """
    values = np.asarray(matrix, dtype="float32")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    values, _ = _sanitize(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _clean_terms(terms: list[str] | tuple[str, ...] | None, limit: int) -> list[str]:
    """Trim, drop blanks and duplicates, and cap the count."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for term in terms or []:
        text = str(term).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def text_model_for(projection_path: str) -> tuple[ModelSpec, Any]:
    """The projection's own model, refusing the ones that cannot read text.

    Using a different model than the projection was built with would compare
    vectors from two unrelated spaces, which produces numbers that look fine and
    mean nothing.
    """
    path, rows, vectors = load_projection(projection_path)
    sidecar = projection_sidecar(path) or {}
    model_key = sidecar.get("model_key")
    if not model_key:
        raise ValueError(
            "This projection does not record which model produced it, so its text "
            "space cannot be used. Re-run the projection to regenerate the sidecar."
        )
    spec = get_model_spec(model_key)
    if not spec.supports_text_search:
        raise ValueError(
            f"'{spec.label}' encodes images but not text, so it has no shared space to "
            "question. Re-run this folder with a CLIP-family model to use the semantic tools."
        )
    public = spec.to_public_dict()
    if not public.get("available"):
        missing = ", ".join(public.get("missing_requirements", []))
        raise ValueError(
            f"'{spec.label}' is not available in this environment"
            + (f" (missing: {missing})." if missing else ".")
        )
    return spec, {"path": path, "rows": rows, "vectors": vectors, "sidecar": sidecar}


def encode_phrases(spec: ModelSpec, phrases: list[str]) -> np.ndarray:
    """L2-normalised text vectors, one per phrase."""
    if not phrases:
        raise ValueError("No phrase was given.")
    return _unit(encode_texts(phrases, spec, batch_size=min(32, len(phrases))))


def _pole(spec: ModelSpec, phrases: list[str]) -> np.ndarray:
    """One direction from possibly several phrases.

    Averaging before normalising is deliberate: several phrasings of the same
    idea ("a portrait", "a painted portrait", "a head and shoulders") describe one
    direction more reliably than any single wording, which carries the quirks of
    how that phrase happened to appear in the training captions.
    """
    vectors = encode_phrases(spec, phrases)
    return _unit(vectors.mean(axis=0))[0]


# --------------------------------------------------------------------------
# 1.2 Concept intensity


def concept_scores(
    projection_path: str,
    *,
    phrases: list[str],
    normalize: bool = True,
) -> dict[str, Any]:
    """Score every image in the projection against one concept.

    Search answers "which are the closest"; this answers "how is the concept
    distributed". The caller gets one number per image, plus the quantiles needed
    to shade a plot sensibly: raw cosine values sit in a narrow band, so a colour
    ramp stretched over the theoretical -1..1 range would look uniform.
    """
    terms = _clean_terms(phrases, MAX_CONCEPT_TERMS)
    if not terms:
        raise ValueError("Enter a concept to score the collection against.")

    spec, data = text_model_for(projection_path)
    rows, vectors = data["rows"], data["vectors"]

    images = _unit(vectors) if normalize else np.asarray(vectors, dtype="float32")
    direction = _pole(spec, terms)
    scores = images @ direction

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("The concept could not be compared with these embeddings.")

    # Percentile bounds rather than min/max: one outlier should not flatten the
    # ramp for everything else.
    low = float(np.percentile(finite, 2))
    high = float(np.percentile(finite, 98))

    items = []
    for index, row in enumerate(rows):
        score = float(scores[index])
        items.append({
            "index": index,
            "relative_path": row.get("relative_path", ""),
            "filename": row.get("filename", ""),
            "score": round(score, 6) if np.isfinite(score) else None,
        })

    order = np.argsort(-scores)
    return {
        "projection_path": data["path"].relative_to(ROOT).as_posix(),
        "model_label": spec.label,
        "phrases": terms,
        "count": len(items),
        "min": round(float(finite.min()), 6),
        "max": round(float(finite.max()), 6),
        "median": round(float(np.median(finite)), 6),
        # The window a colour ramp should span.
        "ramp_low": round(low, 6),
        "ramp_high": round(high, 6),
        "scores": items,
        "top": [items[int(i)] for i in order[:12]],
        "bottom": [items[int(i)] for i in order[-12:][::-1]],
    }


# --------------------------------------------------------------------------
# 1.1 Semantic axes


def semantic_axis(spec: ModelSpec, positive: list[str], negative: list[str]) -> np.ndarray:
    """The direction from one pole to the other, as a unit vector.

    Subtracting the poles rather than using the positive one alone is what makes
    the axis an axis: "portrait" on its own also points at "person", "face" and
    "painting", and those components are shared with the other pole, so the
    difference cancels them and leaves what actually separates the two.
    """
    pos_terms = _clean_terms(positive, MAX_CONCEPT_TERMS)
    neg_terms = _clean_terms(negative, MAX_CONCEPT_TERMS)
    if not pos_terms or not neg_terms:
        raise ValueError("A semantic axis needs a phrase at each end.")

    direction = _pole(spec, pos_terms) - _pole(spec, neg_terms)
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError(
            "The two ends of this axis describe the same thing, so there is no direction "
            "between them. Try poles that genuinely oppose each other."
        )
    return direction / length


def semantic_axis_preview(
    projection_path: str,
    *,
    x_positive: list[str],
    x_negative: list[str],
    y_positive: list[str],
    y_negative: list[str],
) -> dict[str, Any]:
    """Coordinates on two named axes, without re-running the projection.

    Lets the axes be tried and adjusted against the collection that is already
    loaded; the job pipeline uses the same maths to write them into a projection.
    """
    spec, data = text_model_for(projection_path)
    rows, vectors = data["rows"], data["vectors"]
    images = _unit(vectors)

    axis_x = semantic_axis(spec, x_positive, x_negative)
    axis_y = semantic_axis(spec, y_positive, y_negative)
    # How independent the two axes are. Near ±1 means they measure the same
    # thing and the plot collapses onto a diagonal.
    overlap = float(np.dot(axis_x, axis_y))

    xs = images @ axis_x
    ys = images @ axis_y

    points = []
    for index, row in enumerate(rows):
        points.append({
            "index": index,
            "relative_path": row.get("relative_path", ""),
            "filename": row.get("filename", ""),
            "x": round(float(xs[index]), 6),
            "y": round(float(ys[index]), 6),
        })

    return {
        "projection_path": data["path"].relative_to(ROOT).as_posix(),
        "model_label": spec.label,
        "axis_overlap": round(overlap, 4),
        "independent": bool(abs(overlap) < 0.6),
        "count": len(points),
        "points": points,
    }


# --------------------------------------------------------------------------
# 1.4 What distinguishes A from B, in words


def distinguish(
    projection_path: str,
    *,
    paths_a: list[str],
    paths_b: list[str],
    vocabulary: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Read the difference between two groups back out as language.

    The centroid difference is a direction in the shared space; scoring the
    vocabulary along it says which words lie towards A and which towards B.
    Anything both groups share cancels in the subtraction, which is exactly what
    makes the answer specific: it returns what separates them, not what they are.
    """
    spec, data = text_model_for(projection_path)
    rows, vectors = data["rows"], data["vectors"]
    images = _unit(vectors)

    by_path = {row.get("relative_path", ""): index for index, row in enumerate(rows)}

    def indices_for(paths: list[str], label: str) -> list[int]:
        found = [by_path[p] for p in paths if p in by_path]
        if not found:
            raise ValueError(f"Group {label} contains no image from this projection.")
        return found

    index_a = indices_for(paths_a, "A")
    index_b = indices_for(paths_b, "B")
    if set(index_a) == set(index_b):
        raise ValueError("The two groups are identical, so nothing distinguishes them.")

    centroid_a = _unit(images[index_a].mean(axis=0))[0]
    centroid_b = _unit(images[index_b].mean(axis=0))[0]
    direction = centroid_a - centroid_b
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        raise ValueError("These two groups sit at the same place in the space.")
    direction = direction / length

    terms = _clean_terms(vocabulary or list(DEFAULT_VOCABULARY), MAX_VOCABULARY_TERMS)
    if len(terms) < 2:
        raise ValueError("The vocabulary needs at least two terms.")

    term_vectors = encode_phrases(spec, terms)
    alignment = term_vectors @ direction

    order = np.argsort(-alignment)
    towards_a = [
        {"term": terms[int(i)], "score": round(float(alignment[int(i)]), 6)}
        for i in order[:top_k]
    ]
    towards_b = [
        {"term": terms[int(i)], "score": round(float(-alignment[int(i)]), 6)}
        for i in order[::-1][:top_k]
    ]

    # How far apart the groups are overall, for context: a long list of
    # distinguishing words means little if the two groups nearly coincide.
    separation = float(np.dot(centroid_a, centroid_b))
    return {
        "projection_path": data["path"].relative_to(ROOT).as_posix(),
        "model_label": spec.label,
        "size_a": len(index_a),
        "size_b": len(index_b),
        "centroid_similarity": round(separation, 6),
        "vocabulary_size": len(terms),
        "towards_a": towards_a,
        "towards_b": towards_b,
    }


# --------------------------------------------------------------------------
# 1.6 Concept arithmetic


def _standardise(scores: np.ndarray) -> np.ndarray:
    """Put a score column in units of its own spread across this collection.

    Image-to-image cosine spans roughly 0.2..0.8 here; image-to-text spans
    0.13..0.27. Those two scales are not comparable, and summing them raw lets
    the image term outvote every phrase by a factor of four - which is precisely
    why the arithmetic used to return the base image's plain neighbours whatever
    you typed. Standardising first makes each term contribute in units of how
    much it actually separates *this* collection, the only scale that means
    anything here.
    """
    values = np.asarray(scores, dtype="float32")
    spread = float(values.std())
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def concept_arithmetic(
    projection_path: str,
    *,
    relative_path: str | None = None,
    base_phrases: list[str] | None = None,
    add: list[str] | None = None,
    subtract: list[str] | None = None,
    strength: float = 1.0,
    top_k: int = 24,
) -> dict[str, Any]:
    """Start from an image (or a phrase), add and subtract ideas, see what lands.

    Read it as "images like this one, but more X and less Y". The combination is
    made in score space rather than by adding the vectors themselves, because
    CLIP does not put images and words in the same region of the sphere: it puts
    them in two separate cones, and an image vector sits about 0.2 from *every*
    phrase regardless of what either depicts. Adding a phrase vector to an image
    vector therefore barely rotates it, and the ranking stays the base image's
    own neighbour list.

    Scoring each term separately and standardising before the sum sidesteps that
    gap entirely - each term is measured against the collection on its own terms,
    and only then are they weighed against each other.

    It is still not a clean operation. The space encodes web captions, not a
    grammar, so treat it as an instrument for thinking and look at what comes
    back.
    """
    spec, data = text_model_for(projection_path)
    rows, vectors = data["rows"], data["vectors"]
    images = _unit(vectors)

    base_index: int | None = None
    if relative_path:
        for index, row in enumerate(rows):
            if row.get("relative_path") == relative_path:
                base_index = index
                break
        if base_index is None:
            raise ValueError(f"'{relative_path}' is not part of this projection.")

    base_terms = _clean_terms(base_phrases, MAX_CONCEPT_TERMS)
    add_terms = _clean_terms(add, MAX_CONCEPT_TERMS)
    subtract_terms = _clean_terms(subtract, MAX_CONCEPT_TERMS)

    if base_index is None and not base_terms:
        raise ValueError("Start from an image or a phrase.")

    # How hard the phrases pull against the starting image. At 0 the result is
    # simply the image's neighbours; raise it and the concepts take over.
    weight = float(max(0.0, min(5.0, strength)))

    total = np.zeros(len(rows), dtype="float32")
    components: list[dict[str, Any]] = []

    def contribute(label: str, direction: np.ndarray, sign: float, scale: float) -> None:
        column = _standardise(images @ direction)
        nonlocal total
        total = total + sign * scale * column
        components.append({"term": label, "sign": int(sign), "weight": round(scale, 4)})

    if base_index is not None:
        contribute(rows[base_index].get("filename", "the image"), images[base_index], 1.0, 1.0)
    if base_terms:
        contribute(" + ".join(base_terms), _pole(spec, base_terms), 1.0, 1.0)
    for term in add_terms:
        contribute(term, _pole(spec, [term]), 1.0, weight)
    for term in subtract_terms:
        contribute(term, _pole(spec, [term]), -1.0, weight)

    if base_index is not None:
        total[base_index] = -np.inf  # the starting image is not an answer

    order = np.argsort(-total)
    results = []
    for position in order:
        score = float(total[int(position)])
        if not np.isfinite(score):
            continue
        row = rows[int(position)]
        results.append({
            "index": int(position),
            "rank": len(results) + 1,
            "relative_path": row.get("relative_path", ""),
            "filename": row.get("filename", ""),
            # Standard deviations above the collection average, not a cosine:
            # the UI says so rather than showing a number that looks like one.
            "score": round(score, 6),
        })
        if len(results) >= top_k:
            break

    described = []
    if base_index is not None:
        described.append(rows[base_index].get("filename", "the image"))
    if base_terms:
        described.append(" + ".join(base_terms))
    expression = " ".join(filter(None, [
        " + ".join(described),
        " ".join(f"+ {term}" for term in add_terms),
        " ".join(f"- {term}" for term in subtract_terms),
    ])).strip()

    return {
        "projection_path": data["path"].relative_to(ROOT).as_posix(),
        "model_label": spec.label,
        "expression": expression,
        "base_image": rows[base_index].get("relative_path") if base_index is not None else None,
        "strength": round(weight, 3),
        "components": components,
        "count": len(results),
        "searched": len(rows),
        "results": results,
    }
