"""Unit tests for the embed() function."""

from __future__ import annotations

import numpy as np

from vecgrep.embedder import embed


class TestShape:
    def test_returns_correct_shape(self):
        vecs = embed(["hello world", "foo bar"])
        assert vecs.shape == (2, 384)

    def test_dtype_is_float32(self):
        vecs = embed(["test"])
        assert vecs.dtype == np.float32

    def test_empty_input_returns_zero_rows(self):
        vecs = embed([])
        assert vecs.shape == (0, 384)
        assert vecs.dtype == np.float32

    def test_single_text(self):
        vecs = embed(["def foo(): pass"])
        assert vecs.shape == (1, 384)


class TestNormalization:
    def test_vectors_are_unit_norm(self):
        vecs = embed(["alpha", "beta", "gamma"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestDeterminism:
    def test_same_input_produces_same_output(self):
        texts = ["class Foo:\n    pass", "def bar(): ..."]
        v1 = embed(texts)
        v2 = embed(texts)
        np.testing.assert_array_equal(v1, v2)
