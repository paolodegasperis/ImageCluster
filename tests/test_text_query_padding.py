"""How a query is padded decides whether text search works at all.

Four models shipped with search that ranked at chance while their images
encoded perfectly, so nothing looked broken: SigLIP and SigLIP 2 through the
Transformers path, MobileCLIP2 S2 and S4 through OpenCLIP. Both causes were
padding, and in both the failure is silent - the text tower returns a
near-identical vector for every query instead of raising.

These tests pin the two settings in place. They need no network: the padding
decision is made before any model is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.encoders import transformers_encoder as te
from backend.encoders.openclip_encoder import _zero_pad_after_eot
from backend.encoders.registry import MODEL_REGISTRY


class _RecordingProcessor:
    """Stands in for a Hugging Face processor and remembers its keyword args."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs["text"])
        return {"input_ids": torch.ones((n, 8), dtype=torch.long)}


def test_transformers_queries_are_padded_to_the_full_context(monkeypatch):
    """padding=True pads to the longest line of the batch, which breaks SigLIP."""
    processor = _RecordingProcessor()
    monkeypatch.setattr(te, "_select_device", lambda: "cpu")
    monkeypatch.setattr(te, "_load_text_processor", lambda spec: processor)
    monkeypatch.setattr(te, "_load_model", lambda spec, device: object())
    monkeypatch.setattr(te, "_move_to_device", lambda inputs, device: inputs)
    monkeypatch.setattr(
        te, "_extract_text_embedding", lambda model, inputs, provider: torch.ones((2, 4))
    )

    te.encode_texts_with_transformers(["a short one", "a considerably longer query"],
                                      MODEL_REGISTRY["siglip_base_patch16_224"])

    assert processor.calls, "the processor was never called"
    assert processor.calls[0]["padding"] == "max_length"


def test_openclip_padding_after_the_end_of_text_token_is_blanked():
    """MobileCLIP2 S2/S4 collapse when the padding repeats the end-of-text id."""
    eot = 49407
    tokens = torch.tensor([
        [49406, 320, 1125, eot, eot, eot, eot],   # HF tokenizer: pads with EOT
        [49406, 15, 16, 17, 18, 19, eot],         # fills the context exactly
    ])
    cleaned = _zero_pad_after_eot(tokens)

    assert cleaned[0].tolist() == [49406, 320, 1125, eot, 0, 0, 0]
    # A row that ends on its end-of-text token has no padding to blank.
    assert cleaned[1].tolist() == [49406, 15, 16, 17, 18, 19, eot]
    # The caller's tensor is never modified in place.
    assert tokens[0][-1].item() == eot


def test_zero_padded_rows_are_left_alone():
    """OpenCLIP's own tokenizer already pads with zeros; the fix is a no-op there."""
    tokens = torch.tensor([[49406, 320, 49407, 0, 0]])
    assert np.array_equal(_zero_pad_after_eot(tokens).numpy(), tokens.numpy())


def test_non_tensor_input_passes_through_untouched():
    """Some tokenizers return a dict rather than a tensor; do not corrupt it."""
    payload = {"input_ids": [[1, 2, 3]]}
    assert _zero_pad_after_eot(payload) is payload
