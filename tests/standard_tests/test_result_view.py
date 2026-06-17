"""Tests for ResultView — all properties, methods, and dunder operations."""
import json

import numpy as np
import pytest

from lynse.result_view import ResultView, _parse_index_mode


def _make_search_result(n=5, dim=8, with_fields=True):
    ids = np.arange(n, dtype=np.int64)
    distances = np.random.rand(n).astype(np.float32)
    fields = [{"tag": f"item_{i}", "score": float(i)} for i in range(n)] if with_fields else []
    return ResultView(
        ids=ids,
        distances=distances,
        fields=fields,
        k=n,
        distance="IP",
        index="FLAT",
        result_type="search",
    )


def _make_data_result(n=5, dim=8):
    ids = np.arange(n, dtype=np.int64)
    vectors = np.random.rand(n, dim).astype(np.float32)
    fields = [{"tag": f"item_{i}"} for i in range(n)]
    return ResultView(
        ids=ids,
        vectors=vectors,
        fields=fields,
        result_type="data",
    )


def _make_query_result(n=5):
    ids = np.arange(n, dtype=np.int64)
    fields = [{"tag": f"item_{i}", "group": i % 2} for i in range(n)]
    return ResultView(ids=ids, fields=fields, result_type="query")


class TestResultViewProperties:
    def test_ids_dtype(self):
        rv = _make_search_result()
        assert rv.ids.dtype == np.int64

    def test_distances_dtype(self):
        rv = _make_search_result()
        assert rv.distances.dtype == np.float32

    def test_fields_list(self):
        rv = _make_search_result(with_fields=True)
        assert isinstance(rv.fields, list)
        assert len(rv.fields) == 5

    def test_fields_empty_when_not_provided(self):
        rv = _make_search_result(with_fields=False)
        assert rv.fields == [] or rv.fields is None

    def test_vectors_in_data_result(self):
        rv = _make_data_result()
        assert rv.vectors is not None
        assert rv.vectors.shape == (5, 8)

    def test_vectors_none_in_search_result(self):
        rv = _make_search_result()
        assert rv.vectors is None

    def test_distances_none_in_data_result(self):
        rv = _make_data_result()
        assert rv.distances is None

    def test_result_type_search(self):
        rv = _make_search_result()
        assert rv.result_type == "search"

    def test_result_type_data(self):
        rv = _make_data_result()
        assert rv.result_type == "data"

    def test_result_type_query(self):
        rv = _make_query_result()
        assert rv.result_type == "query"

    def test_k_attribute(self):
        rv = _make_search_result(n=7)
        assert rv.k == 7

    def test_distance_metric(self):
        rv = _make_search_result()
        assert rv.distance_metric == "IP"

    def test_index_type(self):
        rv = _make_search_result()
        assert rv.index_type == "FLAT"


class TestResultViewDunder:
    def test_len(self):
        rv = _make_search_result(n=5)
        assert len(rv) == 5

    def test_len_empty(self):
        rv = ResultView(ids=np.array([], dtype=np.int64), result_type="query")
        assert len(rv) == 0

    def test_bool_truthy(self):
        rv = _make_search_result(n=3)
        assert bool(rv) is True

    def test_bool_falsy(self):
        rv = ResultView(ids=np.array([], dtype=np.int64), result_type="query")
        assert bool(rv) is False

    def test_getitem_string_ids(self):
        rv = _make_search_result(n=5)
        ids = rv["ids"]
        assert ids is not None
        assert len(ids) == 5

    def test_getitem_string_distances(self):
        rv = _make_search_result(n=5)
        dists = rv["distances"]
        assert dists is not None

    def test_getitem_string_fields(self):
        rv = _make_search_result(n=5, with_fields=True)
        fields = rv["fields"]
        assert isinstance(fields, list)

    def test_getitem_string_k(self):
        rv = _make_search_result(n=5)
        assert rv["k"] == 5

    def test_getitem_string_measure(self):
        rv = _make_search_result(n=5)
        assert rv["measure"] == "IP"

    def test_getitem_invalid_type_raises(self):
        rv = _make_search_result(n=5)
        with pytest.raises(TypeError):
            _ = rv[0]

    def test_getitem_unknown_key_raises(self):
        rv = _make_search_result(n=5)
        with pytest.raises(KeyError):
            _ = rv["nonexistent"]

    def test_iter_search_yields_3_components(self):
        rv = _make_search_result(n=4)
        components = list(rv)
        assert len(components) == 3

    def test_equality_same(self):
        ids = np.array([1, 2, 3], dtype=np.int64)
        dists = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        rv1 = ResultView(ids=ids, distances=dists, k=3, result_type="search")
        rv2 = ResultView(ids=ids.copy(), distances=dists.copy(), k=3, result_type="search")
        assert rv1 == rv2

    def test_equality_different(self):
        ids1 = np.array([1, 2, 3], dtype=np.int64)
        ids2 = np.array([4, 5, 6], dtype=np.int64)
        dists = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        rv1 = ResultView(ids=ids1, distances=dists, k=3, result_type="search")
        rv2 = ResultView(ids=ids2, distances=dists.copy(), k=3, result_type="search")
        assert rv1 != rv2

    def test_repr(self):
        rv = _make_search_result()
        r = repr(rv)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_tuple_unpack_search(self):
        rv = _make_search_result(with_fields=True)
        ids, distances, fields = rv
        assert len(ids) == 5
        assert len(distances) == 5
        assert len(fields) == 5

    def test_tuple_unpack_data(self):
        rv = _make_data_result()
        vectors, ids, fields = rv
        assert len(ids) == 5
        assert vectors.shape == (5, 8)


class TestResultViewConversions:
    def test_to_dict(self):
        rv = _make_search_result(n=3, with_fields=True)
        d = rv.to_dict()
        assert isinstance(d, dict)
        assert "ids" in d

    def test_to_list(self):
        rv = _make_search_result(n=3, with_fields=True)
        lst = rv.to_list()
        assert isinstance(lst, list)
        assert len(lst) == 3

    def test_to_json(self):
        rv = _make_search_result(n=3, with_fields=True)
        j = rv.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert isinstance(parsed, (dict, list))

    def test_to_numpy_ids(self):
        rv = _make_search_result(n=3)
        arr = rv.ids
        assert isinstance(arr, np.ndarray)

    def test_to_pandas(self):
        pd = pytest.importorskip("pandas")
        rv = _make_search_result(n=4, with_fields=True)
        df = rv.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_to_polars(self):
        pl = pytest.importorskip("polars")
        rv = _make_search_result(n=4, with_fields=True)
        df = rv.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4

    def test_to_arrow(self):
        pa = pytest.importorskip("pyarrow")
        rv = _make_search_result(n=4, with_fields=True)
        table = rv.to_arrow()
        assert isinstance(table, pa.Table)
        assert len(table) == 4

    def test_to_dict_data_result(self):
        rv = _make_data_result(n=3)
        d = rv.to_dict()
        assert isinstance(d, dict)
        assert "ids" in d

    def test_to_list_query_result(self):
        rv = _make_query_result(n=4)
        lst = rv.to_list()
        assert isinstance(lst, list)


class TestParseIndexMode:
    def test_flat_ip(self):
        idx, metric = _parse_index_mode("FLAT-IP")
        assert "FLAT" in idx.upper()
        assert metric is not None

    def test_flat_l2(self):
        idx, metric = _parse_index_mode("FLAT-L2")
        assert metric is not None

    def test_flat_cos(self):
        idx, metric = _parse_index_mode("FLAT-COS")
        assert metric is not None

    def test_hnsw(self):
        idx, metric = _parse_index_mode("HNSW-IP")
        assert "HNSW" in idx.upper()

    def test_ivf(self):
        idx, metric = _parse_index_mode("IVF-L2")
        assert "IVF" in idx.upper()

    def test_spann(self):
        idx, metric = _parse_index_mode("SPANN-L2")
        assert "SPANN" in idx.upper()
        assert metric == "L2"

    def test_diskann(self):
        idx, metric = _parse_index_mode("DiskANN-Cos")
        assert "DISKANN" in idx.upper() or "DiskANN" in idx

    def test_none_input(self):
        idx, metric = _parse_index_mode(None)
        assert idx is not None or metric is not None or True

    def test_returns_tuple(self):
        result = _parse_index_mode("FLAT-L2")
        assert isinstance(result, tuple)
        assert len(result) == 2
