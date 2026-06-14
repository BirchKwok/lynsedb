"""Lazy default embedding support for document-first inserts and search."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_TEXT_EMBEDDING_MODEL = "Qdrant/clip-ViT-B-32-text"
DEFAULT_MODEL_CACHE_DIR = Path(
    os.environ.get("LYNSE_MODEL_CACHE", Path.home() / ".cache" / "lynse" / "models")
)

_TEXT_MODELS: dict[tuple[str, str], object] = {}


def embed_documents(
    documents: Iterable[str],
    *,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    cache_dir: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Embed documents with LynseDB's lazy local default model."""
    docs = list(documents)
    if not docs:
        raise ValueError("documents cannot be empty")
    model = _get_text_model(model_name=model_name, cache_dir=cache_dir)
    vectors = list(model.embed(docs))
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _get_text_model(*, model_name: str, cache_dir: str | os.PathLike[str] | None):
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_MODEL_CACHE_DIR
    key = (model_name, str(cache))
    if key not in _TEXT_MODELS:
        TextEmbedding = _import_text_embedding()
        cache.mkdir(parents=True, exist_ok=True)
        _TEXT_MODELS[key] = TextEmbedding(model_name=model_name, cache_dir=str(cache))
    return _TEXT_MODELS[key]


def _import_text_embedding():
    try:
        from fastembed import TextEmbedding
        return TextEmbedding
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastembed"])
        from fastembed import TextEmbedding
        return TextEmbedding

