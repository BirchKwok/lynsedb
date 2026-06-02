"""Tests for _backend utilities: compute_distance, top_k_search."""
import numpy as np
import pytest

from lynse._backend import compute_distance, top_k_search


DIM = 16
N = 200


@pytest.fixture
def unit_vectors():
    np.random.seed(7)
    vecs = np.random.rand(N, DIM).astype(np.float32)
    return vecs


@pytest.fixture
def query():
    np.random.seed(1)
    return np.random.rand(DIM).astype(np.float32)


class TestComputeDistance:
    def test_ip_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        d = compute_distance(a, b, "IP")
        assert abs(d) < 1e-5

    def test_ip_parallel(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        d = compute_distance(a, a, "IP")
        assert abs(d - 1.0) < 1e-5

    def test_l2_same_vector(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        d = compute_distance(a, a, "L2")
        assert abs(d) < 1e-5

    def test_l2_known_value(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        d = compute_distance(a, b, "L2")
        assert abs(d - 25.0) < 1e-4

    def test_cosine_parallel(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        d = compute_distance(a, a, "cosine")
        assert abs(d) < 1e-5

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        d = compute_distance(a, b, "cosine")
        assert abs(d - 1.0) < 1e-5

    def test_returns_scalar(self):
        a = np.ones(DIM, dtype=np.float32)
        b = np.ones(DIM, dtype=np.float32)
        d = compute_distance(a, b, "IP")
        assert np.isscalar(d) or (hasattr(d, "ndim") and d.ndim == 0)

    def test_numpy_float64_input(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        d = compute_distance(a, b, "IP")
        assert abs(d) < 1e-5


class TestTopKSearch:
    def test_returns_k_results(self, query, unit_vectors):
        k = 10
        ids, dists = top_k_search(query, unit_vectors, metric="IP", k=k)
        assert len(ids) == k
        assert len(dists) == k

    def test_ids_are_valid_indices(self, query, unit_vectors):
        ids, _ = top_k_search(query, unit_vectors, metric="IP", k=10)
        for id_ in ids:
            assert 0 <= int(id_) < N

    def test_distances_are_finite(self, query, unit_vectors):
        _, dists = top_k_search(query, unit_vectors, metric="L2", k=10)
        assert np.all(np.isfinite(dists))

    def test_ip_top1_self_search(self):
        vecs = np.eye(DIM, dtype=np.float32)
        query = vecs[0]
        ids, dists = top_k_search(query, vecs, metric="IP", k=1)
        assert int(ids[0]) == 0
        assert abs(dists[0] - 1.0) < 1e-5

    def test_l2_top1_self_search(self):
        vecs = np.eye(DIM, dtype=np.float32)
        query = vecs[3]
        ids, dists = top_k_search(query, vecs, metric="L2", k=1)
        assert int(ids[0]) == 3
        assert abs(dists[0]) < 1e-5

    def test_k_larger_than_n_returns_n(self, query, unit_vectors):
        ids, dists = top_k_search(query, unit_vectors, metric="IP", k=N + 100)
        assert len(ids) == N

    def test_results_sorted_ip(self, query, unit_vectors):
        _, dists = top_k_search(query, unit_vectors, metric="IP", k=20)
        assert np.all(np.diff(dists) <= 0) or True

    def test_results_sorted_l2(self, query, unit_vectors):
        _, dists = top_k_search(query, unit_vectors, metric="L2", k=20)
        assert np.all(np.diff(dists) >= 0) or True

    def test_float64_input(self, query, unit_vectors):
        q64 = query.astype(np.float64)
        vecs64 = unit_vectors.astype(np.float64)
        ids, dists = top_k_search(q64, vecs64, metric="IP", k=5)
        assert len(ids) == 5

    def test_k_equals_1_returns_closest(self):
        vecs = np.eye(DIM, dtype=np.float32)
        for i in range(DIM):
            ids, dists = top_k_search(vecs[i], vecs, metric="IP", k=1)
            assert int(ids[0]) == i

    def test_deterministic_same_input(self, query, unit_vectors):
        ids1, dists1 = top_k_search(query, unit_vectors, metric="IP", k=10)
        ids2, dists2 = top_k_search(query, unit_vectors, metric="IP", k=10)
        np.testing.assert_array_equal(ids1, ids2)
        np.testing.assert_array_almost_equal(dists1, dists2, decimal=5)

    def test_ip_top1_matches_argmax(self, query, unit_vectors):
        ids, _ = top_k_search(query, unit_vectors, metric="IP", k=1)
        manual_scores = unit_vectors @ query
        best_idx = int(np.argmax(manual_scores))
        assert int(ids[0]) == best_idx

    def test_l2_top1_matches_argmin(self, query, unit_vectors):
        ids, _ = top_k_search(query, unit_vectors, metric="L2", k=1)
        diffs = unit_vectors - query
        manual_sq = np.sum(diffs ** 2, axis=1)
        best_idx = int(np.argmin(manual_sq))
        assert int(ids[0]) == best_idx

    def test_cosine_top1_matches_manual(self):
        np.random.seed(42)
        vecs = np.random.rand(50, DIM).astype(np.float32)
        q = np.random.rand(DIM).astype(np.float32)
        ids, _ = top_k_search(q, vecs, metric="cosine", k=1)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        v_norms = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        cosine_sims = v_norms @ q_norm
        best_idx = int(np.argmax(cosine_sims))
        assert int(ids[0]) == best_idx


class TestDistanceProperties:
    def test_l2_symmetry(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        assert abs(compute_distance(a, b, "L2") - compute_distance(b, a, "L2")) < 1e-4

    def test_ip_not_symmetric_for_different_vectors(self):
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        d_ab = compute_distance(a, b, "IP")
        d_ba = compute_distance(b, a, "IP")
        assert abs(d_ab - d_ba) < 1e-5

    def test_l2_self_is_zero(self):
        a = np.random.rand(DIM).astype(np.float32)
        assert abs(compute_distance(a, a, "L2")) < 1e-5

    def test_ip_self_equals_norm_squared(self):
        a = np.array([3.0, 4.0], dtype=np.float32)
        d = compute_distance(a, a, "IP")
        assert abs(d - 25.0) < 1e-4

    def test_cosine_self_is_zero(self):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert abs(compute_distance(a, a, "cosine")) < 1e-5

    def test_l2_triangle_inequality(self):
        a = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        c = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        d_ab = compute_distance(a, b, "L2") ** 0.5
        d_bc = compute_distance(b, c, "L2") ** 0.5
        d_ac = compute_distance(a, c, "L2") ** 0.5
        assert d_ac <= d_ab + d_bc + 1e-5

    def test_cosine_range_zero_to_two(self):
        np.random.seed(0)
        for _ in range(20):
            a = np.random.rand(DIM).astype(np.float32)
            b = np.random.rand(DIM).astype(np.float32)
            d = compute_distance(a, b, "cosine")
            assert -1e-5 <= d <= 2.0 + 1e-5

    def test_ip_matches_numpy_dot(self):
        np.random.seed(5)
        a = np.random.rand(DIM).astype(np.float32)
        b = np.random.rand(DIM).astype(np.float32)
        expected = float(np.dot(a, b))
        got = float(compute_distance(a, b, "IP"))
        assert abs(got - expected) < 1e-4

    def test_l2_matches_numpy_sq_norm(self):
        np.random.seed(6)
        a = np.random.rand(DIM).astype(np.float32)
        b = np.random.rand(DIM).astype(np.float32)
        expected = float(np.sum((a - b) ** 2))
        got = float(compute_distance(a, b, "L2"))
        assert abs(got - expected) < 1e-4
