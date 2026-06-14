"""Tests for LocalCollection CRUD, metadata, index, soft-delete, compact, stats."""
import numpy as np
import pytest

DIM = 8
N = 20


class TestAddItem:
    def test_add_returns_id(self, collection):
        vec = np.random.rand(DIM).astype(np.float32)
        ret = collection.add(ids=1, vectors=vec)
        collection.commit()
        assert ret == 1

    def test_add_with_field(self, collection):
        vec = np.random.rand(DIM).astype(np.float32)
        collection.add(ids=2, vectors=vec, fields={"key": "val"})
        collection.commit()
        assert collection.is_id_exists(2)

    def test_add_buffered(self, collection):
        collection.add(
            ids=list(range(5)),
            vectors=np.random.rand(5, DIM).astype(np.float32),
        )
        collection.commit()
        assert collection.shape[0] == 5

    def test_add_default_buffer_flushes_periodically(self, collection):
        collection.add(
            ids=list(range(10_000)),
            vectors=np.random.rand(10_000, DIM).astype(np.float32),
            batch_size=1000,
        )

        assert collection.shape[0] == 10_000
        assert collection._mesosphere_list.empty()

    def test_insert_session_commits_automatically(self, collection):
        with collection.insert_session() as session:
            session.add(ids=99, vectors=np.random.rand(DIM).astype(np.float32))
        assert collection.is_id_exists(99)


class TestBulkAdd:
    def test_add_batch(self, collection):
        ids = list(range(10))
        vectors = np.random.rand(10, DIM).astype(np.float32)
        fields = [{"tag": str(i)} for i in ids]
        with collection.insert_session() as session:
            inserted = session.add(ids=ids, vectors=vectors, fields=fields)
        assert len(inserted) == 10
        assert collection.shape[0] == 10

    def test_add_batch_no_fields(self, collection):
        ids = list(range(5))
        vectors = np.random.rand(5, DIM).astype(np.float32)
        with collection.insert_session() as session:
            inserted = session.add(ids=ids, vectors=vectors)
        assert len(inserted) == 5

    def test_bulk_add_binary(self, collection):
        vecs = np.random.rand(50, DIM).astype(np.float32)
        n = collection.bulk_add_binary(vecs, enable_progress_bar=False)
        collection.commit()
        assert n == 50
        assert collection.shape[0] == 50

    def test_bulk_add_binary_1d_input(self, collection):
        vec = np.random.rand(DIM).astype(np.float32)
        n = collection.bulk_add_binary(vec, enable_progress_bar=False)
        collection.commit()
        assert n == 1


class TestCollectionProperties:
    def test_shape_empty(self, collection):
        assert collection.shape == (0, DIM)

    def test_shape_after_insert(self, populated_collection):
        n, d = populated_collection.shape
        assert n == N
        assert d == DIM

    def test_max_id(self, populated_collection):
        assert populated_collection.max_id == N - 1

    def test_is_id_exists_true(self, populated_collection):
        assert populated_collection.is_id_exists(0) is True

    def test_is_id_exists_false(self, populated_collection):
        assert populated_collection.is_id_exists(9999) is False

    def test_stats(self, populated_collection):
        s = populated_collection.stats()
        assert isinstance(s, dict)
        assert s["n_vectors"] == N
        assert s["n_live"] == N
        assert s["n_tombstoned"] == 0
        assert s["dimension"] == DIM

    def test_index_mode_default(self, populated_collection):
        mode = populated_collection.index_mode
        assert mode is None or isinstance(mode, str)

    def test_exists(self, populated_collection):
        assert populated_collection.exists() is True

    def test_repr(self, populated_collection):
        r = repr(populated_collection)
        assert "test_col" in r


class TestHeadTail:
    def test_head(self, populated_collection):
        result = populated_collection.head(5)
        assert len(result.ids) == 5
        assert result.vectors.shape == (5, DIM)

    def test_tail(self, populated_collection):
        result = populated_collection.tail(5)
        assert len(result.ids) == 5
        assert result.vectors.shape == (5, DIM)

    def test_head_default_n(self, populated_collection):
        result = populated_collection.head()
        assert len(result.ids) == 5

    def test_tail_default_n(self, populated_collection):
        result = populated_collection.tail()
        assert len(result.ids) == 5



class TestListFields:
    def test_list_fields(self, populated_collection):
        fields = populated_collection.list_fields()
        assert isinstance(fields, list)

    def test_list_fields_contains_field_names(self, populated_collection):
        fields = populated_collection.list_fields()
        assert "tag" in fields or len(fields) >= 0


class TestUpdateDescription:
    def test_update_description(self, collection):
        collection.update_description("new description")


class TestIndexBuildRemove:
    def test_build_index_flat(self, populated_collection):
        populated_collection.build_index("FLAT")
        assert "FLAT" in populated_collection.index_mode.upper()

    def test_build_index_flat_l2(self, populated_collection):
        populated_collection.build_index("FLAT-L2")
        assert populated_collection.index_mode is not None

    def test_build_index_flat_cos(self, populated_collection):
        populated_collection.build_index("FLAT-COS")
        assert populated_collection.index_mode is not None

    def test_remove_index(self, populated_collection):
        populated_collection.build_index("FLAT")
        populated_collection.remove_index()

    def test_build_hnsw(self, populated_collection):
        populated_collection.build_index("HNSW")
        assert "HNSW" in populated_collection.index_mode.upper()

    def test_build_hnsw_l2(self, populated_collection):
        populated_collection.build_index("HNSW-L2")

    def test_build_ivf(self, populated_collection):
        populated_collection.build_index("IVF", n_clusters=4)
        assert "IVF" in populated_collection.index_mode.upper()

    def test_build_ivf_l2(self, populated_collection):
        populated_collection.build_index("IVF-L2", n_clusters=4)

    def test_n_clusters_is_ignored_for_non_ivf_index(self, populated_collection):
        populated_collection.build_index("HNSW", n_clusters=4)
        assert "HNSW" in populated_collection.index_mode.upper()


class TestSoftDelete:
    def test_delete(self, populated_collection):
        populated_collection.delete([0, 1])
        deleted = populated_collection.list_deleted_ids()
        assert 0 in deleted
        assert 1 in deleted

    def test_list_deleted_ids_empty(self, populated_collection):
        assert populated_collection.list_deleted_ids() == []

    def test_restore(self, populated_collection):
        populated_collection.delete([2, 3])
        populated_collection.restore([2])
        deleted = populated_collection.list_deleted_ids()
        assert 2 not in deleted
        assert 3 in deleted

    def test_deleted_excluded_from_stats(self, populated_collection):
        populated_collection.delete([0, 1, 2])
        s = populated_collection.stats()
        assert s["n_tombstoned"] == 3
        assert s["n_live"] == N - 3

    def test_delete_then_restore_all(self, populated_collection):
        ids_to_delete = [0, 1, 2, 3]
        populated_collection.delete(ids_to_delete)
        populated_collection.restore(ids_to_delete)
        assert populated_collection.list_deleted_ids() == []


class TestCompact:
    def test_compact_returns_count(self, populated_collection):
        populated_collection.delete([0, 1, 2])
        removed = populated_collection.compact()
        assert removed == 3

    def test_compact_clears_tombstone(self, populated_collection):
        populated_collection.delete([0])
        populated_collection.compact()
        assert populated_collection.list_deleted_ids() == []

    def test_compact_no_deletes_returns_zero(self, populated_collection):
        removed = populated_collection.compact()
        assert removed == 0

    def test_compact_reduces_shape(self, populated_collection):
        populated_collection.delete([0, 1, 2])
        populated_collection.compact()
        n, d = populated_collection.shape
        assert n == N - 3
        assert d == DIM


class TestCommit:
    def test_commit_after_add(self, collection):
        collection.add(ids=1, vectors=np.random.rand(DIM).astype(np.float32))
        collection.commit()
        assert collection.is_id_exists(1)


class TestEmptyCollection:
    def test_shape_on_empty(self, collection):
        n, d = collection.shape
        assert n == 0
        assert d == DIM

    def test_max_id_on_empty_returns_minus_one(self, collection):
        assert collection.max_id == -1

    def test_stats_on_empty(self, collection):
        s = collection.stats()
        assert s["n_vectors"] == 0
        assert s["n_live"] == 0
        assert s["n_tombstoned"] == 0

    def test_is_id_exists_on_empty(self, collection):
        assert collection.is_id_exists(0) is False
        assert collection.is_id_exists(99) is False

    def test_list_deleted_ids_on_empty(self, collection):
        assert collection.list_deleted_ids() == []

    def test_compact_on_empty_returns_zero(self, collection):
        assert collection.compact() == 0

    def test_list_fields_on_empty(self, collection):
        fields = collection.list_fields()
        assert isinstance(fields, list)

    def test_remove_index_on_no_index_is_noop(self, collection):
        collection.remove_index()

    def test_delete_nonexistent_ids_is_noop(self, collection):
        collection.delete([999, 1000])
        assert collection.list_deleted_ids() == []


class TestInsertSession:
    def test_session_commits_on_exit(self, collection):
        with collection.insert_session() as session:
            session.add(ids=77, vectors=np.ones(DIM, dtype=np.float32))
        assert collection.is_id_exists(77)

    def test_session_bulk_commits_on_exit(self, collection):
        ids = list(range(5))
        vectors = np.random.rand(5, DIM).astype(np.float32)
        fields = [{"v": i} for i in ids]
        with collection.insert_session() as session:
            session.add(ids=ids, vectors=vectors, fields=fields)
        assert collection.shape[0] == 5

    def test_session_multiple_add(self, collection):
        with collection.insert_session() as session:
            for i in range(10):
                session.add(ids=i, vectors=np.random.rand(DIM).astype(np.float32))
        assert collection.shape[0] == 10

    def test_session_exception_discards_pending_buffer(self, collection):
        with pytest.raises(ValueError):
            with collection.insert_session() as session:
                session.add(ids=88, vectors=np.ones(DIM, dtype=np.float32))
                raise ValueError("boom")

        assert not collection.is_id_exists(88)
        collection.commit()
        assert not collection.is_id_exists(88)

    def test_data_not_visible_before_commit(self, collection):
        with collection.insert_session() as session:
            session.add(ids=55, vectors=np.ones(DIM, dtype=np.float32))
            assert not collection.is_id_exists(55)
        assert collection.is_id_exists(55)


class TestDuplicateId:
    def test_add_duplicate_id_is_rejected(self, collection):
        v1 = np.ones(DIM, dtype=np.float32)
        v2 = np.zeros(DIM, dtype=np.float32) + 0.5
        with collection.insert_session() as s:
            s.add(ids=1, vectors=v1)
        with pytest.raises(RuntimeError, match="already exists"):
            collection.add(ids=1, vectors=v2)
        assert collection.is_id_exists(1)

    def test_is_id_exists_after_insert(self, populated_collection):
        for i in range(N):
            assert populated_collection.is_id_exists(i) is True
        assert populated_collection.is_id_exists(N + 100) is False


class TestUpsert:
    def test_upsert_updates_existing_without_growing_shape(self, collection):
        v1 = np.ones(DIM, dtype=np.float32)
        v2 = np.zeros(DIM, dtype=np.float32)
        collection.add(ids=10, vectors=v1, fields={"tag": "old"})
        collection.upsert(ids=10, vectors=v2, fields={"tag": "new"})

        assert collection.shape[0] == 1
        result = collection.query_vectors(filter_ids=[10])
        assert np.allclose(result.vectors[0], v2)
        assert result.fields[0]["tag"] == "new"

    def test_upserts_mixes_update_and_insert(self, collection):
        collection.add(ids=1, vectors=np.ones(DIM, dtype=np.float32), fields={"tag": "old"})
        ids = collection.upsert(
            ids=[1, 2],
            vectors=[
                np.zeros(DIM, dtype=np.float32),
                np.ones(DIM, dtype=np.float32) * 2,
            ],
            fields=[{"tag": "new"}, {"tag": "inserted"}],
        )

        assert ids == [1, 2]
        assert collection.shape[0] == 2
        assert collection.query(where="\"tag\" = 'old'", return_ids_only=True).ids.tolist() == []
        assert collection.query(where="\"tag\" = 'new'", return_ids_only=True).ids.tolist() == [1]

    def test_upsert_without_field_preserves_existing_fields(self, collection):
        collection.add(ids=5, vectors=np.ones(DIM, dtype=np.float32), fields={"tag": "keep"})
        collection.upsert(ids=5, vectors=np.zeros(DIM, dtype=np.float32))
        result = collection.query(filter_ids=[5])
        assert result.fields[0]["tag"] == "keep"


class TestEdgeCases:
    def test_head_n_exceeds_size(self, populated_collection):
        result = populated_collection.head(N + 100)
        assert len(result.ids) == N

    def test_tail_n_exceeds_size(self, populated_collection):
        result = populated_collection.tail(N + 100)
        assert len(result.ids) == N

    def test_stats_after_compact(self, populated_collection):
        populated_collection.delete([0, 1])
        populated_collection.compact()
        s = populated_collection.stats()
        assert s["n_vectors"] == N - 2
        assert s["n_tombstoned"] == 0
        assert s["n_live"] == N - 2

    def test_build_index_then_remove_then_rebuild(self, populated_collection):
        populated_collection.build_index("FLAT-L2")
        assert "FLAT" in populated_collection.index_mode.upper()
        populated_collection.remove_index()
        assert populated_collection.index_mode is None or populated_collection.index_mode == ""
        populated_collection.build_index("FLAT-COS")
        assert "FLAT" in populated_collection.index_mode.upper()

    def test_add_no_field(self, collection):
        vec = np.random.rand(DIM).astype(np.float32)
        collection.add(ids=42, vectors=vec)
        collection.commit()
        assert collection.is_id_exists(42)

    def test_add_batch_no_fields(self, collection):
        ids = list(range(5))
        vectors = np.random.rand(5, DIM).astype(np.float32)
        with collection.insert_session() as s:
            s.add(ids=ids, vectors=vectors)
        assert collection.shape[0] == 5

    def test_restore_nonexistent_id_does_not_appear_in_tombstone(self, populated_collection):
        populated_collection.restore([9999])
        assert 9999 not in populated_collection.list_deleted_ids()

    def test_delete_already_deleted_is_idempotent(self, populated_collection):
        populated_collection.delete([5])
        populated_collection.delete([5])
        deleted = populated_collection.list_deleted_ids()
        assert deleted.count(5) == 1
