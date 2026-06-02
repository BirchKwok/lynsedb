"""Tests for VectorDBClient and LocalClient (database-level API)."""
import pytest

import lynse

DIM = 8


class TestVectorDBClient:
    def test_local_mode_no_uri(self, tmp_root):
        c = lynse.VectorDBClient(uri=tmp_root)
        assert c is not None

    def test_create_database(self, client):
        db = client.create_database("db_create_test", drop_if_exists=True)
        assert db is not None

    def test_create_database_drop_if_exists(self, client):
        client.create_database("db_dup", drop_if_exists=True)
        db2 = client.create_database("db_dup", drop_if_exists=True)
        assert db2 is not None

    def test_get_database(self, client):
        client.create_database("db_get", drop_if_exists=True)
        db = client.get_database("db_get")
        assert db is not None

    def test_list_databases(self, client):
        client.create_database("db_list_a", drop_if_exists=True)
        client.create_database("db_list_b", drop_if_exists=True)
        dbs = client.list_databases()
        assert "db_list_a" in dbs
        assert "db_list_b" in dbs

    def test_drop_database(self, client):
        client.create_database("db_drop_test", drop_if_exists=True)
        dbs_before = client.list_databases()
        assert "db_drop_test" in dbs_before
        client.drop_database("db_drop_test")
        dbs_after = client.list_databases()
        assert "db_drop_test" not in dbs_after


class TestLocalClient:
    def test_require_collection_creates(self, db):
        coll = db.require_collection("col_new", dim=DIM, drop_if_exists=True)
        assert coll is not None

    def test_require_collection_drop_if_exists(self, db, populated_collection):
        coll2 = db.require_collection("test_col", dim=DIM, drop_if_exists=True)
        assert coll2.shape[0] == 0

    def test_require_collection_with_description(self, db):
        coll = db.require_collection(
            "col_desc", dim=DIM, drop_if_exists=True, description="my desc"
        )
        assert coll is not None

    def test_get_collection(self, db, collection):
        fetched = db.get_collection("test_col")
        assert fetched is not None

    def test_get_collection_nonexistent_raises(self, db):
        with pytest.raises(Exception):
            db.get_collection("nonexistent_col_xyz")

    def test_show_collections(self, db):
        db.require_collection("col_show_a", dim=DIM, drop_if_exists=True)
        db.require_collection("col_show_b", dim=DIM, drop_if_exists=True)
        cols = db.show_collections()
        assert "col_show_a" in cols
        assert "col_show_b" in cols

    def test_show_collections_details(self, db):
        db.require_collection("col_detail", dim=DIM, drop_if_exists=True)
        details = db.show_collections_details()
        assert details is not None

    def test_drop_collection(self, db):
        db.require_collection("col_drop", dim=DIM, drop_if_exists=True)
        assert "col_drop" in db.show_collections()
        db.drop_collection("col_drop")
        assert "col_drop" not in db.show_collections()

    def test_database_exists(self, db):
        result = db.database_exists()
        exists = result if isinstance(result, bool) else result.get("params", {}).get("exists", True)
        assert exists is True

    def test_update_collection_description(self, db):
        db.require_collection("col_upd_desc", dim=DIM, drop_if_exists=True)
        db.update_collection_description("col_upd_desc", "updated desc")

    def test_drop_database(self, client):
        db2 = client.create_database("db_to_drop", drop_if_exists=True)
        db2.drop_database()
        assert "db_to_drop" not in client.list_databases()


class TestEdgeCases:
    def test_client_repr(self, tmp_root):
        import lynse
        c = lynse.VectorDBClient(uri=tmp_root)
        r = repr(c)
        assert "VectorDBClient" in r

    def test_client_str(self, tmp_root):
        import lynse
        c = lynse.VectorDBClient(uri=tmp_root)
        assert isinstance(str(c), str)

    def test_reopen_same_root_path_in_process(self, tmp_root):
        """Notebook-style re-assignment must not hit the writer lock."""
        c1 = lynse.VectorDBClient(uri=tmp_root)
        c1.create_database("db_reopen", drop_if_exists=True)
        c2 = lynse.VectorDBClient(uri=tmp_root)
        assert "db_reopen" in c2.list_databases()

    def test_db_repr(self, db):
        r = repr(db)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_get_database_nonexistent_raises(self, client):
        with pytest.raises((ValueError, KeyError, Exception)):
            client.get_database("xyz_nonexistent_db")

    def test_require_collection_same_dim_twice_is_idempotent(self, db):
        c1 = db.require_collection("idempotent_col", dim=DIM, drop_if_exists=True)
        c2 = db.require_collection("idempotent_col", dim=DIM)
        assert c2 is not None

    def test_require_collection_same_dim_returns_collection(self, db):
        c = db.require_collection("same_dim_col", dim=DIM, drop_if_exists=True)
        assert c is not None
        assert c.shape[1] == DIM

    def test_database_not_in_list_after_drop(self, client):
        db2 = client.create_database("drop_check_db", drop_if_exists=True)
        assert "drop_check_db" in client.list_databases()
        db2.drop_database()
        assert "drop_check_db" not in client.list_databases()

    def test_collection_not_in_list_after_drop(self, db):
        db.require_collection("col_drop_check", dim=DIM, drop_if_exists=True)
        assert "col_drop_check" in db.show_collections()
        db.drop_collection("col_drop_check")
        assert "col_drop_check" not in db.show_collections()

    def test_database_exists_after_create(self, client):
        db = client.create_database("exists_check", drop_if_exists=True)
        result = db.database_exists()
        exists = result if isinstance(result, bool) else result.get("params", {}).get("exists", True)
        assert exists is True

    def test_show_collections_empty_database(self, db):
        result = db.show_collections()
        assert isinstance(result, list)
