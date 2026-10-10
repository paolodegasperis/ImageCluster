"""The README's model table must list every model the registry offers.

It drifted for three releases without anyone noticing: HQ-CLIP and PE-Core
(v6.7.0), CultureCLIP (v6.8.0) and EmbeddingGemma 2 (v6.9.0) were all wired,
documented in docs/ and shipped while the README still described the list as it
stood in v6.6. The README is the first thing a visitor reads and the only
description most of them will read, so a model missing from it is a model that,
for practical purposes, was not announced.

This test does not check the prose. It checks that nothing is absent.
"""

from __future__ import annotations

from pathlib import Path

from backend.encoders.registry import MODEL_REGISTRY

README = Path(__file__).resolve().parents[1] / "README.md"


def _table() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Available embedding models")
    return text[start:text.index("Models marked *optional*", start)]


def test_every_offered_model_is_listed():
    table = _table()
    missing = [
        spec.label
        for spec in MODEL_REGISTRY.values()
        if spec.status != "planned" and spec.label not in table
    ]
    assert not missing, (
        "These models are offered by the registry but absent from the README table: "
        + ", ".join(missing)
    )


def test_planned_models_are_not_advertised():
    """A roadmap entry has no checkpoint; listing it would promise what the app
    cannot do."""
    table = _table()
    for spec in MODEL_REGISTRY.values():
        if spec.status == "planned":
            assert spec.label not in table, f"{spec.label} is planned and must not be listed"


def test_text_search_column_matches_the_registry():
    table = _table()
    for spec in MODEL_REGISTRY.values():
        if spec.status == "planned":
            continue
        row = next(line for line in table.splitlines() if line.startswith(f"| {spec.label}"))
        mark = row.split("|")[2].strip()
        expected = "\u2713" if spec.supports_text_search else "\u2014"
        assert mark == expected, f"{spec.label}: README says {mark!r}, registry says {expected!r}"
