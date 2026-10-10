"""Two settings that Transformers 5 changed under us, pinned in place.

Both failures are quiet rather than loud, which is why they are tested at all:

* ``from_pretrained`` now honours the dtype a checkpoint was saved with instead
  of always loading float32. A bfloat16 checkpoint on CPU then returns vectors
  whose norms miss unity by a fraction of a percent - measured at 0.998 to 1.004
  on EmbeddingGemma - and every analysis here assumes a unit-norm vector.
* Models trained with a task prefix answer a bare query anyway: ranked,
  plausible, worse. This project has already shipped that failure once, in the
  four models whose text towers were collapsing silently before v6.7.0.

Neither test touches the network: both decisions are made before a model loads.
"""

from __future__ import annotations

import pytest

from backend.encoders import sentence_transformers_encoder as ste
from backend.encoders import transformers_encoder as te
from backend.encoders.registry import MODEL_REGISTRY, ModelSpec


def _spec(**kwargs) -> ModelSpec:
    base = dict(key="probe", family="Probe", label="Probe", provider="transformers_clip", model_id="probe/model")
    base.update(kwargs)
    return ModelSpec(**base)


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "float32"), ("cuda", ""), ("mps", "")],
)
def test_cpu_loads_float32_and_an_accelerator_keeps_the_saved_dtype(monkeypatch, device, expected):
    seen: dict = {}

    def fake_from_pretrained(factory, model_id, *, trust_remote_code, dtype=""):
        seen["dtype"] = dtype

        class _Model:
            def to(self, _device):
                return self

            def eval(self):
                return self

        return _Model()

    monkeypatch.setattr(te, "_from_pretrained", fake_from_pretrained)
    # A distinct model id per case: loaded models are cached per key, and a cache
    # hit would return another case's model and test nothing.
    te._load_model(_spec(model_id=f"probe/{device}"), device)
    assert seen["dtype"] == expected


class _RecordingModel:
    """Stands in for a SentenceTransformer and remembers how it was called."""

    def __init__(self, prompts: dict | None = None) -> None:
        self.prompts = prompts or {}
        self.calls: list[dict] = []

    def encode(self, items, **kwargs):
        self.calls.append(kwargs)
        return [[0.0, 1.0] for _ in items]


def test_a_declared_prompt_is_applied_to_a_query():
    model = _RecordingModel({"query": "task: search result | query: "})
    ste._encode(model, ["a woman in a red dress"], 1, prompt_name="query")
    assert model.calls[0]["prompt_name"] == "query"


def test_an_undeclared_prompt_is_not_passed():
    """Sentence-Transformers raises on an unknown prompt name, and the models
    that take a bare query must keep working."""
    model = _RecordingModel({})
    ste._encode(model, ["a woman in a red dress"], 1, prompt_name="query")
    assert "prompt_name" not in model.calls[0]


def test_embeddinggemma_declares_the_prompt_it_was_trained_with():
    assert MODEL_REGISTRY["embeddinggemma2"].text_prompt_name == "query"


def test_models_without_a_task_prefix_declare_none():
    for key in ("openai_clip_vit_b_32", "cultureclip_b32", "openclip_vit_b_32"):
        assert MODEL_REGISTRY[key].text_prompt_name == ""
