"""Lazy default embedding support for document-first inserts and search."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np


DEFAULT_TEXT_EMBEDDING_ADAPTER = os.environ.get("LYNSE_TEXT_EMBEDDING_ADAPTER", "fastembed")
DEFAULT_TEXT_EMBEDDING_MODEL = os.environ.get(
    "LYNSE_TEXT_EMBEDDING_MODEL",
    "Qdrant/clip-ViT-B-32-text",
)
DEFAULT_MODEL_CACHE_DIR = Path(
    os.environ.get("LYNSE_MODEL_CACHE", Path.home() / ".cache" / "lynse" / "models")
)
DEFAULT_AUTO_INSTALL_EMBEDDINGS = os.environ.get(
    "LYNSE_AUTO_INSTALL_EMBEDDINGS", "1"
).strip().lower() not in {"0", "false", "no", "off"}

_TEXT_MODELS: dict[tuple[str, str, str], object] = {}


def embed_documents(
    documents: Iterable[str],
    *,
    embed_func: Optional[Callable[[list[str]], Any]] = None,
    adapter: str = DEFAULT_TEXT_EMBEDDING_ADAPTER,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    cache_dir: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Embed documents with a user callable or LynseDB's lazy default model."""
    docs = list(documents)
    if not docs:
        raise ValueError("documents cannot be empty")
    if embed_func is not None:
        if not callable(embed_func):
            raise TypeError("embed_func must be callable")
        vectors = np.asarray(embed_func(docs), dtype=np.float32)
        if vectors.ndim == 1 and len(docs) == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError("embed_func must return a 2D array-like value")
        if vectors.shape[0] != len(docs):
            raise ValueError("embedding output count must match documents length")
        if vectors.shape[1] == 0:
            raise ValueError("embed_func must return non-empty vectors")
        return np.ascontiguousarray(vectors, dtype=np.float32)
    model = _get_text_model(adapter=adapter, model_name=model_name, cache_dir=cache_dir)
    vectors = list(model.embed(docs))
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _get_text_model(
    *,
    adapter: str,
    model_name: str,
    cache_dir: str | os.PathLike[str] | None,
):
    adapter_name = str(adapter or "fastembed").strip().lower()
    if adapter_name in {"default", "local"}:
        adapter_name = "fastembed"
    if adapter_name != "fastembed":
        raise ValueError(
            "unsupported text embedding adapter "
            f"{adapter!r}; supported adapters: 'fastembed'"
        )

    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_MODEL_CACHE_DIR
    key = (adapter_name, model_name, str(cache))
    if key not in _TEXT_MODELS:
        TextEmbedding = _import_fastembed_text_embedding(
            auto_install=DEFAULT_AUTO_INSTALL_EMBEDDINGS
        )
        cache.mkdir(parents=True, exist_ok=True)
        _TEXT_MODELS[key] = TextEmbedding(model_name=model_name, cache_dir=str(cache))
    return _TEXT_MODELS[key]


def _import_fastembed_text_embedding(*, auto_install: bool):
    try:
        from fastembed import TextEmbedding
        return TextEmbedding
    except ImportError as import_error:
        if not auto_install:
            raise ImportError(
                "document embedding requires fastembed. Install it with "
                "`pip install 'lynsedb[embeddings]'` or `pip install fastembed`, "
                "or pass vectors explicitly to add()/search()."
            ) from import_error
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fastembed"])
        except Exception as install_error:
            raise ImportError(
                "failed to install fastembed automatically. Install it with "
                "`pip install 'lynsedb[embeddings]'` or `pip install fastembed`, "
                "or set LYNSE_AUTO_INSTALL_EMBEDDINGS=0 and pass vectors explicitly."
            ) from install_error
        from fastembed import TextEmbedding
        return TextEmbedding
