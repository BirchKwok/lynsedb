"""Tests for the lazy document embedding adapter."""

import numpy as np
import pytest


def test_embed_documents_uses_configurable_fastembed_adapter(monkeypatch, tmp_path):
    from lynse.api import _embedding

    calls = []

    class FakeTextEmbedding:
        def __init__(self, *, model_name, cache_dir):
            calls.append((model_name, cache_dir))

        def embed(self, documents):
            for idx, _ in enumerate(documents):
                yield np.array([float(idx), 1.0], dtype=np.float32)

    monkeypatch.setattr(_embedding, "_TEXT_MODELS", {})
    monkeypatch.setattr(
        _embedding,
        "_import_fastembed_text_embedding",
        lambda *, auto_install: FakeTextEmbedding,
    )

    vectors = _embedding.embed_documents(
        ["alpha", "beta"],
        model_name="test-model",
        cache_dir=tmp_path,
    )

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    assert calls == [("test-model", str(tmp_path))]


def test_embed_documents_rejects_unknown_adapter():
    from lynse.api import _embedding

    with pytest.raises(ValueError, match="unsupported text embedding adapter"):
        _embedding.embed_documents(["alpha"], adapter="unknown")


def test_embed_documents_uses_custom_function_without_loading_default_model(monkeypatch):
    from lynse.api import _embedding

    monkeypatch.setattr(
        _embedding, "_get_text_model", lambda **kwargs: pytest.fail("default model loaded")
    )
    vectors = _embedding.embed_documents(
        ["alpha", "beta"],
        embed_func=lambda docs: [[len(text), 1] for text in docs],
    )

    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors, [[5, 1], [4, 1]])


def test_embed_documents_validates_custom_function_output_count():
    from lynse.api import _embedding

    with pytest.raises(ValueError, match="output count"):
        _embedding.embed_documents(["alpha", "beta"], embed_func=lambda docs: [[1, 2]])
