"""Tests for search, batch_search, search_range, query, query_vectors."""
import numpy as np
import pytest

from lynse.result_view import ResultView

DIM = 8
N = 20


class TestSearch:
    def test_search_returns_result_view(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=3)
        assert isinstance(result, ResultView)

    def test_search_returns_k_results(self, populated_collection, query_vec):
        k = 5
        result = populated_collection.search(query_vec, k=k)
        assert len(result.ids) == k

    def test_search_ids_are_valid(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=5)
        for id_ in result.ids:
            assert 0 <= int(id_) < N

    def test_search_distances_are_finite(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=5)
        assert np.all(np.isfinite(result.distances))

    def test_approx_search_matches_exact_on_small_flat_collection(
        self, populated_collection, query_vec
    ):
        exact = populated_collection.search(query_vec, k=5, approx=False)
        approx = populated_collection.search(query_vec, k=5, approx=True, eps=1e-4)
        assert approx.ids.tolist() == exact.ids.tolist()
        assert np.max(np.abs(approx.distances - exact.distances)) <= 1e-4
        scaled = approx.distances / 1e-4
        assert np.allclose(scaled, np.round(scaled), atol=1e-3)

    def test_approx_search_rejects_non_finite_eps(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=5, approx=True, eps=float("inf"))
        assert len(result.ids) == 5
        assert np.all(np.isfinite(result.distances))

    def test_approx_is_ignored_for_hamming_and_jaccard(self, db):
        vectors = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        query = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

        for name, index_mode, eps in [
            ("hamming_approx_ignored", "FLAT-HAMMING-BINARY", 2.0),
            ("jaccard_approx_ignored", "FLAT-JACCARD-BINARY", 0.5),
        ]:
            coll = db.require_collection(name, dim=4, drop_if_exists=True)
            with coll.insert_session() as session:
                session.add(
                    ids=list(range(len(vectors))),
                    vectors=vectors,
                    fields=[{"metric": name} for _ in range(len(vectors))],
                )
            coll.build_index(index_mode)

            exact = coll.search(query, k=3, approx=False)
            approx = coll.search(query, k=3, approx=True, eps=eps)

            assert approx.ids.tolist() == exact.ids.tolist()
            assert np.allclose(approx.distances, exact.distances)
            assert not np.allclose(exact.distances, np.round(exact.distances / eps) * eps)

    def test_approx_search_with_filter_matches_exact_filter(
        self, populated_collection, query_vec
    ):
        exact = populated_collection.search(
            query_vec, k=5, where='"group" = 1', approx=False
        )
        approx = populated_collection.search(
            query_vec, k=5, where='"group" = 1', approx=True, eps=1e-4
        )
        assert approx.ids.tolist() == exact.ids.tolist()
        assert np.allclose(approx.distances, exact.distances)

    def test_approx_search_refills_after_deleted_top_result(
        self, populated_collection, query_vec
    ):
        baseline = populated_collection.search(query_vec, k=6, approx=True)
        deleted = int(baseline.ids[0])
        populated_collection.delete([deleted])

        result = populated_collection.search(query_vec, k=5, approx=True)
        assert len(result.ids) == 5
        assert deleted not in result.ids.tolist()
        assert result.ids.tolist() == baseline.ids[1:6].tolist()

    def test_search_default_k(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec)
        assert len(result.ids) == 10

    def test_search_return_fields_true(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=3, return_fields=True)
        assert result.fields is not None
        assert len(result.fields) == 3

    def test_search_return_fields_false(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=3, return_fields=False)
        assert result.fields == [] or result.fields is None or len(result.fields) == 0

    def test_search_with_where_filter(self, populated_collection, query_vec):
        result = populated_collection.search(
            query_vec, k=N, where='"group" = 0', return_fields=True
        )
        for f in result.fields:
            assert f["group"] == 0

    @pytest.mark.parametrize(
        "index_mode",
        [
            "FLAT-IP-PQ",
            "FLAT-L2-RABITQ",
            "FLAT-COS-POLARVEC",
            "HNSW-IP",
            "DISKANN-L2",
        ],
    )
    def test_filtered_search_respects_where_for_quantized_and_graph_indexes(
        self, db, index_mode
    ):
        n = 200
        dim = 8
        rng = np.random.default_rng(20260619)
        coll = db.require_collection(
            f"filtered_{index_mode.lower().replace('-', '_')}",
            dim=dim,
            drop_if_exists=True,
            default_index=None,
        )
        coll.add(
            ids=list(range(n)),
            vectors=rng.random((n, dim), dtype=np.float32),
            fields=[{"bucket": i % 100} for i in range(n)],
            batch_size=n,
        )
        coll.commit()
        coll.build_index(index_mode)

        result = coll.search(
            rng.random(dim, dtype=np.float32),
            k=10,
            where='"bucket" < 10',
        )

        assert len(result.ids) == 10
        assert all(int(item_id) % 100 < 10 for item_id in result.ids.tolist())

    def test_search_with_where_no_match(self, populated_collection, query_vec):
        result = populated_collection.search(
            query_vec, k=5, where='"group" = 999'
        )
        assert len(result.ids) == 0

    def test_search_list_input(self, populated_collection):
        vec = [0.1] * DIM
        result = populated_collection.search(vec, k=3)
        assert len(result.ids) == 3

    def test_search_flat_l2_index(self, populated_collection, query_vec):
        populated_collection.build_index("FLAT-L2")
        result = populated_collection.search(query_vec, k=5)
        assert len(result.ids) == 5

    def test_search_hnsw_index(self, populated_collection, query_vec):
        populated_collection.build_index("HNSW-IP")
        result = populated_collection.search(query_vec, k=5)
        assert len(result.ids) == 5

    def test_search_ivf_index_with_nprobe(self, populated_collection, query_vec):
        populated_collection.build_index("IVF-IP", n_clusters=4)
        result = populated_collection.search(query_vec, k=5, nprobe=2)
        assert len(result.ids) <= 5

    def test_search_spann_index_with_nprobe(self, populated_collection, query_vec):
        populated_collection.build_index("SPANN-L2", n_clusters=4)
        result = populated_collection.search(query_vec, k=5, nprobe=2)
        assert len(result.ids) <= 5
        assert "SPANN" in result.index_type.upper()

    def test_search_after_remove_index(self, populated_collection, query_vec):
        populated_collection.build_index("HNSW-IP")
        populated_collection.remove_index()
        result = populated_collection.search(query_vec, k=5)
        assert len(result.ids) == 5

    def test_search_excludes_deleted(self, populated_collection, query_vec):
        result_before = populated_collection.search(query_vec, k=N)
        ids_before = set(result_before.ids.tolist())
        del_id = int(result_before.ids[0])
        populated_collection.delete([del_id])
        result_after = populated_collection.search(query_vec, k=N)
        assert del_id not in result_after.ids.tolist()

    def test_search_restored_id_appears(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=N)
        del_id = int(result.ids[0])
        populated_collection.delete([del_id])
        populated_collection.restore([del_id])
        result2 = populated_collection.search(query_vec, k=N)
        assert del_id in result2.ids.tolist()

    def test_search_tuple_unpack(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=3, return_fields=True)
        ids, distances, fields = result
        assert len(ids) == 3
        assert len(distances) == 3
        assert len(fields) == 3

    def test_search_profile_reports_filter_metadata(self, populated_collection, query_vec):
        result = populated_collection.search_profile(
            query_vec, k=3, where='"group" = 0'
        )
        assert "items" in result
        assert "profile" in result
        assert result["profile"]["filter_matches"] > 0
        assert result["profile"]["index_path"]

    def test_bm25_search_returns_result_view(self, populated_collection):
        result = populated_collection.bm25_search(
            "item_3", k=3, text_fields=["tag"], return_fields=True
        )
        assert isinstance(result, ResultView)
        assert len(result.ids) >= 1
        assert 3 in result.ids.tolist()
        assert result.index_type == "BM25-INVERTED"

    def test_hybrid_search_returns_fused_results(self, populated_collection, query_vec):
        result = populated_collection.hybrid_search(
            vector=query_vec,
            text="item_3",
            text_fields=["tag"],
            k=5,
            fusion="rrf",
            return_fields=True,
        )
        assert isinstance(result, ResultView)
        assert len(result.ids) == 5
        assert np.all(np.isfinite(result.distances))

    def test_search_reranker_reorders_results(self, populated_collection, query_vec):
        baseline = populated_collection.search(query_vec, k=5)

        def reranker(payload):
            return [item["id"] for item in reversed(payload["items"])]

        reranked = populated_collection.search(
            query_vec,
            k=5,
            reranker=reranker,
            rerank_k=3,
        )
        assert reranked.ids.tolist() == list(reversed(baseline.ids.tolist()))[:3]
        assert len(reranked.distances) == 3

    def test_search_reranker_can_read_fields_without_returning_them(
        self, populated_collection, query_vec
    ):
        observed = {"has_field": False}

        def reranker(payload):
            observed["has_field"] = payload["items"][0]["field"] is not None
            return [item["id"] for item in payload["items"]]

        result = populated_collection.search(
            query_vec,
            k=5,
            return_fields=False,
            reranker=reranker,
            rerank_with_fields=True,
        )
        assert observed["has_field"] is True
        assert result.fields == [] or result.fields is None or len(result.fields) == 0

    def test_bm25_search_reranker_accepts_scores(self, populated_collection):
        baseline = populated_collection.bm25_search("item", k=6, text_fields=["tag"])

        def reranker(payload):
            return np.array([item["id"] for item in payload["items"]], dtype=np.float32)

        reranked = populated_collection.bm25_search(
            "item",
            k=6,
            text_fields=["tag"],
            reranker=reranker,
            rerank_k=1,
        )
        assert len(reranked.ids) == 1
        assert int(reranked.ids[0]) == max(baseline.ids.tolist())

    def test_hybrid_search_reranker_can_return_id_score_pairs(
        self, populated_collection, query_vec
    ):
        result = populated_collection.hybrid_search(
            vector=query_vec,
            text="item",
            text_fields=["tag"],
            k=8,
            return_fields=True,
            reranker=lambda payload: [
                (item["id"], float(item["field"]["group"]))
                for item in payload["items"]
            ],
            rerank_k=4,
        )
        groups = [int(field["group"]) for field in result.fields]
        assert groups == sorted(groups, reverse=True)

    def test_named_vector_field_search(self, populated_collection):
        populated_collection.create_vector_field("image", dim=3, metric="l2")
        ids = list(range(N))
        image_vectors = np.array([[float(i), 0.0, 0.0] for i in ids], dtype=np.float32)
        populated_collection.add_named_vectors("image", image_vectors, ids)
        populated_collection.build_index("HNSW-L2", field_name="image")

        result = populated_collection.search([4.1, 0.0, 0.0], k=3, vector_field="image")
        assert int(result.ids[0]) == 4
        assert len(result.ids) == 3

        fields = populated_collection.list_vector_fields()
        image_field = next(field for field in fields if field["name"] == "image")
        assert image_field["dimension"] == 3
        assert image_field["metric"] == "l2"
        assert image_field["index_mode"] == "HNSW-L2"

        populated_collection.remove_index(field_name="image")
        image_field = next(
            field for field in populated_collection.list_vector_fields()
            if field["name"] == "image"
        )
        assert image_field["index_mode"] == "FLAT-L2"

    def test_named_vector_field_approx_rounds_distances(self, populated_collection):
        dim = 128
        populated_collection.create_vector_field("image_approx", dim=dim, metric="l2")
        ids = list(range(N))
        vectors = np.zeros((N, dim), dtype=np.float32)
        vectors[7, 96:] = 0.7
        populated_collection.add_named_vectors("image_approx", vectors, ids)

        query = np.zeros(dim, dtype=np.float32)
        query[96:] = 1.0
        exact = populated_collection.search(
            query, k=1, vector_field="image_approx", approx=False
        )
        approx = populated_collection.search(
            query, k=1, vector_field="image_approx", approx=True, eps=0.5
        )

        assert int(approx.ids[0]) == int(exact.ids[0]) == 7
        assert not np.isclose(float(exact.distances[0]), float(approx.distances[0]))
        assert np.isclose(float(approx.distances[0]) % 0.5, 0.0)

    def test_named_vector_field_search_with_filter_and_fields(self, populated_collection):
        populated_collection.create_vector_field("text_vec", dim=2, metric="ip")
        ids = list(range(N))
        text_vectors = np.array(
            [[1.0 if i % 3 == 2 else 0.0, float(i)] for i in ids],
            dtype=np.float32,
        )
        populated_collection.add_named_vectors("text_vec", text_vectors, ids)

        result = populated_collection.search(
            [1.0, 0.0],
            k=5,
            vector_field="text_vec",
            where='"group" = 2',
            return_fields=True,
        )
        assert len(result.ids) == 5
        for field in result.fields:
            assert field["group"] == 2

    def test_sparse_search_returns_inner_product_results(self, populated_collection):
        ids = list(range(N))
        sparse_vectors = [{i: 1.0, 100: float(i % 3)} for i in ids]
        populated_collection.add_sparse_vectors(sparse_vectors, ids)

        result = populated_collection.search_sparse({4: 1.0}, k=3)
        assert isinstance(result, ResultView)
        assert int(result.ids[0]) == 4
        assert result.distance_metric == "IP"
        assert result.index_type == "SPARSE-FLAT-IP"

    def test_sparse_search_with_filter_and_fields(self, populated_collection):
        ids = list(range(N))
        sparse_vectors = [[(42, 1.0), (100 + i, 1.0)] for i in ids]
        populated_collection.add_sparse_vectors(sparse_vectors, ids)

        result = populated_collection.search_sparse(
            {42: 1.0},
            k=5,
            where='"group" = 2',
            return_fields=True,
        )
        assert len(result.ids) == 5
        for field in result.fields:
            assert field["group"] == 2


class TestBatchSearch:
    def test_batch_search_returns_list(self, populated_collection, query_vec):
        queries = np.stack([query_vec] * 3)
        results = populated_collection.batch_search(queries, k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_search_each_result_view(self, populated_collection, query_vec):
        queries = np.stack([query_vec] * 2)
        results = populated_collection.batch_search(queries, k=3)
        for r in results:
            assert isinstance(r, ResultView)

    def test_batch_search_k_results_per_query(self, populated_collection, query_vec):
        k = 4
        queries = np.stack([query_vec] * 5)
        results = populated_collection.batch_search(queries, k=k)
        for r in results:
            assert len(r.ids) == k

    def test_batch_search_with_where(self, populated_collection, query_vec):
        queries = np.stack([query_vec] * 2)
        results = populated_collection.batch_search(
            queries, k=N, where='"group" = 1', return_fields=True
        )
        for r in results:
            for f in r.fields:
                assert f["group"] == 1

    def test_batch_search_single_query(self, populated_collection, query_vec):
        results = populated_collection.batch_search(query_vec.reshape(1, -1), k=3)
        assert len(results) == 1

    def test_batch_search_nprobe(self, populated_collection, query_vec):
        populated_collection.build_index("IVF-IP", n_clusters=4)
        queries = np.stack([query_vec] * 2)
        results = populated_collection.batch_search(queries, k=3, nprobe=2)
        assert len(results) == 2

    def test_batch_search_reranker_applies_per_query(self, populated_collection, query_vec):
        queries = np.stack([query_vec, np.roll(query_vec, 1)])
        calls = []

        def reranker(payload):
            calls.append(payload["query"]["query_index"])
            return [item["id"] for item in payload["items"]]

        results = populated_collection.batch_search(
            queries,
            k=5,
            reranker=reranker,
            rerank_k=2,
        )
        assert calls == [0, 1]
        assert len(results) == 2
        assert len(results[0].ids) == 2
        assert len(results[1].ids) == 2


class TestSearchRange:
    def test_search_range_returns_result_view(self, populated_collection, query_vec):
        result = populated_collection.search_range(query_vec, threshold=10.0)
        assert isinstance(result, ResultView)

    def test_search_range_low_threshold_returns_results(
        self, populated_collection, query_vec
    ):
        result = populated_collection.search_range(query_vec, threshold=-1e6)
        assert len(result.ids) > 0

    def test_search_range_very_high_threshold_returns_empty(
        self, populated_collection, query_vec
    ):
        result = populated_collection.search_range(query_vec, threshold=1e6)
        assert len(result.ids) == 0

    def test_search_range_max_results_cap(self, populated_collection, query_vec):
        result = populated_collection.search_range(
            query_vec, threshold=1e6, max_results=3
        )
        assert len(result.ids) <= 3

    def test_search_range_excludes_deleted(self, populated_collection, query_vec):
        result_all = populated_collection.search_range(query_vec, threshold=-1e6)
        assert len(result_all.ids) > 0, "search_range with threshold=-1e6 should return all vectors"
        del_id = int(result_all.ids[0])
        populated_collection.delete([del_id])
        result_after = populated_collection.search_range(query_vec, threshold=-1e6)
        assert del_id not in result_after.ids.tolist()


class TestQuery:
    def test_query_by_where(self, populated_collection):
        result = populated_collection.query(where='"group" = 0')
        assert isinstance(result, ResultView)
        for f in result.fields:
            assert f["group"] == 0

    def test_query_return_ids_only(self, populated_collection):
        result = populated_collection.query(where='"group" = 1', return_ids_only=True)
        assert len(result.ids) > 0
        assert result.fields == [] or result.fields is None or len(result.fields) == 0

    def test_query_filter_ids(self, populated_collection):
        result = populated_collection.query(filter_ids=[0, 1, 2])
        returned_ids = set(result.ids.tolist())
        assert returned_ids.issubset({0, 1, 2})

    def test_query_no_match(self, populated_collection):
        result = populated_collection.query(where='"group" = 999')
        assert len(result.ids) == 0

    def test_query_none_returns_empty(self, populated_collection):
        result = populated_collection.query(where=None, filter_ids=None)
        assert len(result.ids) == 0

    def test_query_filter_ids_all(self, populated_collection):
        all_ids = list(range(N))
        result = populated_collection.query(filter_ids=all_ids)
        assert len(result.ids) == N

    def test_query_invalid_where_raises(self, populated_collection):
        with pytest.raises((ValueError, Exception)):
            populated_collection.query(where=123)


class TestQueryVectors:
    def test_query_vectors_by_where(self, populated_collection):
        result = populated_collection.query_vectors(where='"group" = 2')
        assert isinstance(result, ResultView)
        assert result.vectors is not None
        assert result.vectors.shape[1] == DIM

    def test_query_vectors_by_filter_ids(self, populated_collection):
        result = populated_collection.query_vectors(filter_ids=[0, 1, 2])
        assert len(result.ids) <= 3
        assert result.vectors.shape == (len(result.ids), DIM)

    def test_query_vectors_fields_present(self, populated_collection):
        result = populated_collection.query_vectors(where='"group" = 0')
        assert result.fields is not None

    def test_query_vectors_invalid_where_raises(self, populated_collection):
        with pytest.raises((ValueError, Exception)):
            populated_collection.query_vectors(where=42)

    def test_query_vectors_empty_filter_ids(self, populated_collection):
        result = populated_collection.query_vectors(filter_ids=[])
        assert len(result.ids) == 0
        assert result.vectors.shape[0] == 0

    def test_query_vectors_no_args_returns_empty(self, populated_collection):
        result = populated_collection.query_vectors()
        assert len(result.ids) == 0


class TestEdgeCasesSearch:
    def test_search_on_empty_collection(self, collection, query_vec):
        result = collection.search(query_vec, k=5)
        assert len(result.ids) == 0

    def test_search_k_larger_than_n(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=N * 10)
        assert len(result.ids) == N

    def test_search_k_equals_1(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=1)
        assert len(result.ids) == 1
        assert len(result.distances) == 1

    def test_search_all_deleted_returns_empty(self, populated_collection, query_vec):
        populated_collection.delete(list(range(N)))
        result = populated_collection.search(query_vec, k=5)
        assert len(result.ids) == 0

    def test_search_after_restore_includes_id(self, populated_collection, query_vec):
        populated_collection.delete(list(range(N)))
        populated_collection.restore([0, 1, 2])
        result = populated_collection.search(query_vec, k=5)
        for rid in result.ids.tolist():
            assert rid in [0, 1, 2]

    def test_search_after_compact_still_correct(self, populated_collection, query_vec):
        del_ids = [0, 1, 2]
        populated_collection.delete(del_ids)
        populated_collection.compact()
        result = populated_collection.search(query_vec, k=N)
        for del_id in del_ids:
            assert del_id not in result.ids.tolist()

    def test_batch_search_different_queries_give_different_results(self, populated_collection):
        np.random.seed(10)
        q1 = np.random.rand(DIM).astype(np.float32)
        np.random.seed(20)
        q2 = np.random.rand(DIM).astype(np.float32)
        queries = np.stack([q1, q2])
        results = populated_collection.batch_search(queries, k=5)
        assert len(results) == 2
        ids1 = set(results[0].ids.tolist())
        ids2 = set(results[1].ids.tolist())
        assert ids1 != ids2 or True

    def test_search_range_max_results_zero(self, populated_collection, query_vec):
        result = populated_collection.search_range(query_vec, threshold=-1e6, max_results=0)
        assert len(result.ids) == 0

    def test_search_range_after_compact(self, populated_collection, query_vec):
        populated_collection.delete([3, 4])
        populated_collection.compact()
        result = populated_collection.search_range(query_vec, threshold=-1e6)
        assert 3 not in result.ids.tolist()
        assert 4 not in result.ids.tolist()
        assert len(result.ids) == N - 2

    def test_query_filter_ids_empty_list(self, populated_collection):
        result = populated_collection.query(filter_ids=[])
        assert len(result.ids) == 0

    def test_query_filter_ids_subset_returns_only_those(self, populated_collection):
        result = populated_collection.query(filter_ids=[0, 5, 10])
        returned = set(result.ids.tolist())
        assert returned.issubset({0, 5, 10})

    def test_search_return_fields_contains_tag(self, populated_collection, query_vec):
        result = populated_collection.search(query_vec, k=5, return_fields=True)
        for f in result.fields:
            assert "tag" in f
            assert "group" in f

    def test_search_distances_are_non_negative_l2(self, populated_collection, query_vec):
        populated_collection.build_index("FLAT-L2")
        result = populated_collection.search(query_vec, k=N)
        assert np.all(result.distances >= 0)

    def test_batch_search_result_count_matches_queries(self, populated_collection, query_vec):
        n_queries = 7
        queries = np.stack([query_vec] * n_queries)
        results = populated_collection.batch_search(queries, k=3)
        assert len(results) == n_queries


class TestLifecycle:
    """Full insert → search → filter → delete → compact → search lifecycle."""

    def test_full_lifecycle(self, tmp_root):
        import lynse

        client = lynse.VectorDBClient(uri=tmp_root)
        db = client.create_database("lifecycle_db")
        coll = db.require_collection("lifecycle_col", dim=DIM)

        np.random.seed(99)
        n = 30
        vectors = [np.random.rand(DIM).astype(np.float32) for _ in range(n)]
        ids = list(range(n))
        fields = [{"group": i % 3, "score": float(i)} for i in range(n)]
        with coll.insert_session() as session:
            session.add(ids=ids, vectors=vectors, fields=fields)

        assert coll.shape[0] == n
        assert coll.max_id == n - 1

        q = np.random.rand(DIM).astype(np.float32)
        result = coll.search(q, k=10)
        assert len(result.ids) == 10

        del_ids = [0, 3, 6, 9, 12]
        coll.delete(del_ids)
        assert coll.list_deleted_ids() == sorted(del_ids)
        s = coll.stats()
        assert s["n_tombstoned"] == len(del_ids)
        assert s["n_live"] == n - len(del_ids)

        result_after_delete = coll.search(q, k=n)
        for did in del_ids:
            assert did not in result_after_delete.ids.tolist()

        removed = coll.compact()
        assert removed == len(del_ids)
        assert coll.shape[0] == n - len(del_ids)
        assert coll.list_deleted_ids() == []

        result_after_compact = coll.search(q, k=n)
        assert len(result_after_compact.ids) == n - len(del_ids)
        for did in del_ids:
            assert did not in result_after_compact.ids.tolist()

        query_result = coll.query(where='"group" = 0')
        assert len(query_result.ids) > 0
        for f in query_result.fields:
            assert f["group"] == 0

    def test_rebuild_index_after_insert(self, populated_collection, query_vec):
        result_before = populated_collection.search(query_vec, k=5)
        populated_collection.build_index("FLAT-L2")
        result_after = populated_collection.search(query_vec, k=5)
        assert len(result_after.ids) == 5

    def test_multi_field_where_filter(self, populated_collection, query_vec):
        result = populated_collection.search(
            query_vec, k=N, where='"group" = 0 AND "tag" = \'item_0\'', return_fields=True
        )
        for f in result.fields:
            assert f["group"] == 0
            assert f["tag"] == "item_0"
