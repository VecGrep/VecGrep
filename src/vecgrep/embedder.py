"""Local embedding using sentence-transformers."""

from __future__ import annotations

import numpy as np

MODEL_NAME = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning a float32 array of shape (N, 384)."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)
