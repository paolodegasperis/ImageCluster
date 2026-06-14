import numpy as np
import pytest

from backend.encoders.base import encode_texts
from backend.encoders.registry import MODEL_REGISTRY
from backend.search.semantic_search import _l2_normalize


def test_cosine_ranking_on_synthetic_vectors():
    image_vectors = _l2_normalize(np.array([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]], dtype="float32"))
    text_vector = _l2_normalize(np.array([[1.0, 0.0]], dtype="float32"))[0]
    scores = image_vectors @ text_vector
    assert list(np.argsort(-scores)) == [0, 1, 2]
    assert scores[0] == pytest.approx(1.0)


def test_encode_texts_rejects_image_only_model():
    with pytest.raises(RuntimeError, match="does not support text search"):
        encode_texts(["Madonna con bambino"], MODEL_REGISTRY["dinov2_base"])
