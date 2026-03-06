"""Unit tests for sirchmunk.vision.vector_search.

This module only depends on numpy (+ optional usearch), so it loads
the module file directly to avoid triggering the heavy vision deps check.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "src", "sirchmunk", "vision", "vector_search.py",
)
_spec = importlib.util.spec_from_file_location(
    "sirchmunk.vision.vector_search", os.path.abspath(_MOD_PATH),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

Metric = _mod.Metric
batch_cosine = _mod.batch_cosine
is_accelerated = _mod.is_accelerated
multi_query_max_pool = _mod.multi_query_max_pool
preferred_dtype = _mod.preferred_dtype
top_k_similar = _mod.top_k_similar


def _random_unit(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Random L2-normalised float32 matrix (n, d)."""
    rng = np.random.RandomState(seed)
    m = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-8)


class TestBatchCosine(unittest.TestCase):
    def test_perfect_match(self):
        mat = _random_unit(5, 64)
        scores = batch_cosine(mat, mat[2])
        self.assertAlmostEqual(float(scores[2]), 1.0, places=5)

    def test_empty_matrix(self):
        empty = np.empty((0, 64), dtype=np.float32)
        q = _random_unit(1, 64)[0]
        result = batch_cosine(empty, q)
        self.assertEqual(len(result), 0)

    def test_output_shape(self):
        mat = _random_unit(100, 128)
        q = _random_unit(1, 128)[0]
        result = batch_cosine(mat, q)
        self.assertEqual(result.shape, (100,))
        self.assertEqual(result.dtype, np.float32)


class TestTopKSimilar(unittest.TestCase):
    def test_finds_exact_match(self):
        mat = _random_unit(50, 64)
        q = mat[17].copy()
        indices, scores = top_k_similar(mat, q, k=5)
        self.assertIn(17, indices)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=4)

    def test_k_larger_than_n(self):
        mat = _random_unit(3, 32)
        q = mat[0]
        indices, scores = top_k_similar(mat, q, k=10)
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(scores), 3)

    def test_empty_matrix(self):
        empty = np.empty((0, 32), dtype=np.float32)
        q = _random_unit(1, 32)[0]
        indices, scores = top_k_similar(empty, q, k=5)
        self.assertEqual(len(indices), 0)

    def test_multi_query(self):
        mat = _random_unit(50, 64)
        queries = mat[[10, 20, 30]]
        indices, scores = top_k_similar(mat, queries, k=5)
        self.assertLessEqual(len(indices), 5)
        for idx in [10, 20, 30]:
            self.assertIn(idx, indices)

    def test_sorted_descending(self):
        mat = _random_unit(100, 64)
        q = _random_unit(1, 64)[0]
        _, scores = top_k_similar(mat, q, k=20)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(float(scores[i]), float(scores[i + 1]))


class TestMultiQueryMaxPool(unittest.TestCase):
    def test_basic(self):
        mat = _random_unit(20, 64)
        queries = _random_unit(3, 64)
        result = multi_query_max_pool(mat, queries)
        self.assertEqual(result.shape, (20,))

        expected = (mat.astype(np.float32) @ queries.T.astype(np.float32)).max(axis=1)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_single_query(self):
        mat = _random_unit(10, 32)
        q = _random_unit(1, 32)
        result = multi_query_max_pool(mat, q)
        expected = batch_cosine(mat, q[0])
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestMetadata(unittest.TestCase):
    def test_is_accelerated_returns_bool(self):
        self.assertIsInstance(is_accelerated(), bool)

    def test_preferred_dtype(self):
        dt = preferred_dtype(fp16=True)
        self.assertIn(dt, (np.dtype(np.float16), np.dtype(np.float32)))

        dt32 = preferred_dtype(fp16=False)
        self.assertEqual(dt32, np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
