"""Smoke coverage for the public Python API surface."""
import inspect

import numpy as np
import pytest

import lynse
from lynse.api.local_client import LocalClient, LocalCollection
from lynse.result_view import ResultView


DIM = 8


def _public_members(cls):
    return {
        name
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_")
        and (
            inspect.isfunction(value)
            or isinstance(value, property)
            or not inspect.isroutine(value)
        )
    }


def test_vectordbclient_public_api_contract():
    assert _public_members(lynse.VectorDBClient) == {
        "close",
        "create_collection",
        "create_database",
        "drop_database",
        "get_database",
        "list_databases",
        "restore_database",
        "snapshot_database",
    }


def test_localclient_public_api_contract():
    assert _public_members(LocalClient) == {
        "database_exists",
        "drop_collection",
        "drop_database",
        "export_collection",
        "get_collection",
        "import_collection",
        "is_read_only",
        "require_collection",
        "restore_collection",
        "restore_database",
        "show_collections",
        "show_collections_details",
        "snapshot_collection",
        "snapshot_database",
        "update_collection_description",
    }


def test_localcollection_public_api_contract():
    assert _public_members(LocalCollection) == {
        "add",
        "add_named_vectors",
        "add_sparse_vectors",
        "batch_search",
        "bm25_search",
        "build_index",
        "checkpoint",
        "close",
        "commit",
        "compact",
        "create_vector_field",
        "delete",
        "exists",
        "export_to",
        "flush",
        "head",
        "hybrid_search",
        "index_mode",
        "insert_session",
        "is_id_exists",
        "is_read_only",
        "list_deleted_ids",
        "list_fields",
        "list_vector_fields",
        "max_id",
        "name",
        "query",
        "query_vectors",
        "remove_index",
        "restore",
        "search",
        "search_profile",
        "search_range",
        "search_sparse",
        "shape",
        "snapshot_to",
        "stats",
        "tail",
        "update_description",
        "upsert",
        "vector_dtype",
    }


def test_resultview_public_api_contract():
    assert _public_members(ResultView) == {
        "distance_metric",
        "distances",
        "fields",
        "ids",
        "index_type",
        "k",
        "result_type",
        "to_arrow",
        "to_dict",
        "to_json",
        "to_list",
        "to_numpy",
        "to_pandas",
        "to_polars",
        "to_tuple",
        "vectors",
    }


@pytest.mark.parametrize(
    "api_name",
    [
        "create_database",
        "get_database",
        "list_databases",
        "create_collection",
        "snapshot_database",
        "restore_database",
        "drop_database",
        "close",
    ],
)
def test_vectordbclient_public_api_smoke(tmp_root, tmp_path, api_name):
    client = lynse.VectorDBClient(uri=tmp_root)
    try:
        if api_name == "create_database":
            db = client.create_database("api_db", drop_if_exists=True)
            assert isinstance(db, LocalClient)
        elif api_name == "get_database":
            client.create_database("api_db", drop_if_exists=True)
            assert isinstance(client.get_database("api_db"), LocalClient)
        elif api_name == "list_databases":
            assert isinstance(client.list_databases(), list)
        elif api_name == "create_collection":
            coll = client.create_collection(
                "api_db",
                "api_col",
                dim=DIM,
                drop_database_if_exists=True,
                drop_if_exists=True,
            )
            assert isinstance(coll, LocalCollection)
        elif api_name == "snapshot_database":
            db = client.create_database("api_db", drop_if_exists=True)
            db.require_collection("api_col", dim=DIM, drop_if_exists=True)
            target = tmp_path / "db_snapshot"
            assert client.snapshot_database("api_db", target) is None
            assert target.exists()
        elif api_name == "restore_database":
            db = client.create_database("api_db", drop_if_exists=True)
            db.require_collection("api_col", dim=DIM, drop_if_exists=True)
            snapshot = tmp_path / "db_snapshot"
            client.snapshot_database("api_db", snapshot)
            client.drop_database("api_db")
            assert client.restore_database("api_db", snapshot) is None
            assert "api_db" in client.list_databases()
        elif api_name == "drop_database":
            client.create_database("api_db", drop_if_exists=True)
            assert client.drop_database("api_db") is None
            assert "api_db" not in client.list_databases()
        elif api_name == "close":
            assert client.close() is None
            client = None
    finally:
        if client is not None:
            client.close()


@pytest.mark.parametrize(
    "api_name",
    [
        "is_read_only",
        "require_collection",
        "get_collection",
        "drop_collection",
        "snapshot_collection",
        "export_collection",
        "restore_collection",
        "import_collection",
        "snapshot_database",
        "restore_database",
        "drop_database",
        "database_exists",
        "show_collections",
        "update_collection_description",
        "show_collections_details",
    ],
)
def test_localclient_public_api_smoke(client, tmp_path, api_name):
    db = client.create_database(f"db_{api_name}", drop_if_exists=True)

    if api_name == "is_read_only":
        assert db.is_read_only is False
    elif api_name == "require_collection":
        assert isinstance(db.require_collection("api_col", dim=DIM), LocalCollection)
    elif api_name == "get_collection":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        assert isinstance(db.get_collection("api_col"), LocalCollection)
    elif api_name == "drop_collection":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        assert db.drop_collection("api_col") == {"status": "success"}
    elif api_name == "snapshot_collection":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        assert db.snapshot_collection("api_col", tmp_path / "col_snapshot") == {
            "status": "success"
        }
    elif api_name == "export_collection":
        coll = db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        coll.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))
        coll.commit()
        assert db.export_collection("api_col", tmp_path / "col_export") == {
            "status": "success"
        }
    elif api_name == "restore_collection":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        snapshot = tmp_path / "col_snapshot"
        db.snapshot_collection("api_col", snapshot)
        db.drop_collection("api_col")
        assert db.restore_collection("api_col", snapshot) == {"status": "success"}
    elif api_name == "import_collection":
        coll = db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        coll.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))
        coll.commit()
        export = tmp_path / "col_export"
        db.export_collection("api_col", export)
        db.drop_collection("api_col")
        assert db.import_collection("api_col", export) == {"status": "success"}
    elif api_name == "snapshot_database":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        assert db.snapshot_database(tmp_path / "db_snapshot") == {"status": "success"}
    elif api_name == "restore_database":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        snapshot = tmp_path / "db_snapshot"
        db.snapshot_database(snapshot)
        db.drop_database()
        assert db.restore_database(snapshot) == {"status": "success"}
    elif api_name == "drop_database":
        assert db.drop_database()["status"] == "success"
    elif api_name == "database_exists":
        assert db.database_exists()["params"]["exists"] is True
    elif api_name == "show_collections":
        assert isinstance(db.show_collections(), list)
    elif api_name == "update_collection_description":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        assert db.update_collection_description("api_col", "updated") == {
            "status": "success"
        }
    elif api_name == "show_collections_details":
        db.require_collection("api_col", dim=DIM, drop_if_exists=True)
        assert db.show_collections_details() is not None


@pytest.mark.parametrize(
    "api_name",
    [
        "is_read_only",
        "vector_dtype",
        "exists",
        "add",
        "upsert",
        "commit",
        "flush",
        "checkpoint",
        "close",
        "snapshot_to",
        "export_to",
        "is_id_exists",
        "max_id",
        "compact",
        "stats",
        "build_index",
        "remove_index",
        "create_vector_field",
        "list_vector_fields",
        "add_named_vectors",
        "add_sparse_vectors",
        "insert_session",
        "search",
        "search_sparse",
        "search_profile",
        "bm25_search",
        "hybrid_search",
        "batch_search",
        "shape",
        "head",
        "tail",
        "query",
        "query_vectors",
        "delete",
        "restore",
        "list_deleted_ids",
        "search_range",
        "list_fields",
        "update_description",
        "index_mode",
    ],
)
def test_localcollection_public_api_smoke(db, tmp_path, api_name):
    coll = db.require_collection(
        f"col_{api_name}",
        dim=DIM,
        drop_if_exists=True,
        default_index=None,
    )
    vectors = np.eye(4, DIM, dtype=np.float32)
    fields = [
        {"tag": "alpha", "group": 0},
        {"tag": "beta", "group": 1},
        {"tag": "gamma", "group": 0},
        {"tag": "delta", "group": 1},
    ]

    if api_name in {"add", "shape"}:
        assert coll.add(ids=[1, 2], vectors=vectors[:2], fields=fields[:2]) == [1, 2]
    else:
        coll.add(ids=[1, 2, 3, 4], vectors=vectors, fields=fields)
        coll.commit()

    if api_name == "is_read_only":
        assert coll.is_read_only is False
    elif api_name == "vector_dtype":
        assert coll.vector_dtype in {"float32", "float16"}
    elif api_name == "exists":
        assert coll.exists() is True
    elif api_name == "add":
        assert coll.shape == (2, DIM)
    elif api_name == "upsert":
        assert coll.upsert(ids=2, vectors=np.ones(DIM, dtype=np.float32)) == 2
    elif api_name == "commit":
        assert coll.commit() == {"status": "success"}
    elif api_name == "flush":
        assert coll.flush() == {"status": "success"}
    elif api_name == "checkpoint":
        assert coll.checkpoint() == {"status": "success"}
    elif api_name == "close":
        assert coll.close() == {"status": "success"}
    elif api_name == "snapshot_to":
        assert coll.snapshot_to(tmp_path / "snapshot") == {"status": "success"}
    elif api_name == "export_to":
        assert coll.export_to(tmp_path / "export") == {"status": "success"}
    elif api_name == "is_id_exists":
        assert coll.is_id_exists(1) is True
    elif api_name == "max_id":
        assert coll.max_id == 3
    elif api_name == "compact":
        coll.delete([1])
        assert coll.compact() >= 0
    elif api_name == "stats":
        assert coll.stats()["n_vectors"] == 4
    elif api_name == "build_index":
        assert coll.build_index("FLAT-IP") == {"status": "success"}
    elif api_name == "remove_index":
        coll.build_index("FLAT-IP")
        assert coll.remove_index() == {"status": "success"}
    elif api_name == "create_vector_field":
        assert coll.create_vector_field("image", 3, metric="l2") == {
            "status": "success"
        }
    elif api_name == "list_vector_fields":
        assert isinstance(coll.list_vector_fields(), list)
    elif api_name == "add_named_vectors":
        coll.create_vector_field("image", 3, metric="l2")
        assert coll.add_named_vectors("image", np.ones((4, 3), dtype=np.float32), [1, 2, 3, 4]) == {
            "status": "success"
        }
    elif api_name == "add_sparse_vectors":
        assert coll.add_sparse_vectors([{7: 1.0}, {8: 1.0}], [1, 2]) == {
            "status": "success"
        }
    elif api_name == "insert_session":
        with coll.insert_session() as session:
            assert session.add(ids=5, vectors=np.ones(DIM, dtype=np.float32)) == 5
    elif api_name == "search":
        assert len(coll.search(vectors[0], k=2).ids) == 2
    elif api_name == "search_sparse":
        coll.add_sparse_vectors([{7: 1.0}, {8: 1.0}], [1, 2])
        assert len(coll.search_sparse({7: 1.0}, k=1).ids) == 1
    elif api_name == "search_profile":
        profile = coll.search_profile(vectors[0], k=1)
        assert "items" in profile and "profile" in profile
    elif api_name == "bm25_search":
        assert len(coll.bm25_search("alpha", k=1, text_fields=["tag"]).ids) == 1
    elif api_name == "hybrid_search":
        assert len(coll.hybrid_search(vector=vectors[0], text="alpha", k=2).ids) == 2
    elif api_name == "batch_search":
        assert len(coll.batch_search(vectors[:2], k=1)) == 2
    elif api_name == "shape":
        assert coll.shape == (2, DIM)
    elif api_name == "head":
        assert len(coll.head(2).ids) == 2
    elif api_name == "tail":
        assert len(coll.tail(2).ids) == 2
    elif api_name == "query":
        assert coll.query(where='"group" = 0').ids.tolist() == [1, 3]
    elif api_name == "query_vectors":
        assert coll.query_vectors(filter_ids=[1, 2]).vectors.shape == (2, DIM)
    elif api_name == "delete":
        coll.delete([1])
        assert 1 in coll.list_deleted_ids()
    elif api_name == "restore":
        coll.delete([1])
        coll.restore([1])
        assert 1 not in coll.list_deleted_ids()
    elif api_name == "list_deleted_ids":
        assert coll.list_deleted_ids() == []
    elif api_name == "search_range":
        assert len(coll.search_range(vectors[0], threshold=-1e6).ids) > 0
    elif api_name == "list_fields":
        assert {"tag", "group"}.issubset(set(coll.list_fields()))
    elif api_name == "update_description":
        assert coll.update_description("updated") == {"status": "success"}
    elif api_name == "index_mode":
        assert coll.index_mode is None or isinstance(coll.index_mode, str)


def test_resultview_public_api_smoke():
    rv = ResultView(
        ids=np.array([1, 2], dtype=np.int64),
        distances=np.array([0.9, 0.5], dtype=np.float32),
        fields=[{"tag": "a"}, {"tag": "b"}],
        k=2,
        distance="IP",
        index="FLAT",
        result_type="search",
    )

    assert rv.ids.tolist() == [1, 2]
    assert rv.distances.shape == (2,)
    assert rv.vectors is None
    assert rv.fields == [{"tag": "a"}, {"tag": "b"}]
    assert rv.k == 2
    assert rv.distance_metric == "IP"
    assert rv.index_type == "FLAT"
    assert rv.result_type == "search"
    assert rv.to_tuple()[0].tolist() == [1, 2]
    assert rv.to_numpy()["ids"].tolist() == [1, 2]
    assert rv.to_dict()["ids"] == [1, 2]
    assert len(rv.to_list()) == 2
    assert '"id"' in rv.to_json()

    pytest.importorskip("pandas")
    assert rv.to_pandas() is not None

    pytest.importorskip("polars")
    assert rv.to_polars() is not None

    pytest.importorskip("pyarrow")
    assert rv.to_arrow() is not None
