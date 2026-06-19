"""Edge-case tests for public API inputs and lifecycle semantics."""
import numpy as np
import pytest


DIM = 8


@pytest.mark.parametrize(
    "bad_id, exc_type",
    [
        (True, TypeError),
        (-1, ValueError),
        ("", ValueError),
        (1.25, TypeError),
    ],
)
def test_add_rejects_invalid_public_ids(collection, bad_id, exc_type):
    with pytest.raises(exc_type):
        collection.add(ids=bad_id, vectors=np.ones(DIM, dtype=np.float32))


@pytest.mark.parametrize(
    "bad_ids, exc_type",
    [
        ([1, 1], ValueError),
        ([], ValueError),
        ([1, True], TypeError),
        ([1, -2], ValueError),
        ([1, ""], ValueError),
    ],
)
def test_add_rejects_invalid_id_sequences(collection, bad_ids, exc_type):
    vectors = np.ones((max(1, len(bad_ids)), DIM), dtype=np.float32)
    with pytest.raises(exc_type):
        collection.add(ids=bad_ids, vectors=vectors)


def test_add_rejects_single_vector_for_multiple_ids(collection):
    with pytest.raises(ValueError, match="single 1D vector"):
        collection.add(ids=[1, 2], vectors=np.ones(DIM, dtype=np.float32))


def test_add_rejects_vector_row_count_mismatch(collection):
    with pytest.raises(ValueError, match="vectors row count"):
        collection.add(
            ids=[1, 2, 3],
            vectors=np.ones((2, DIM), dtype=np.float32),
        )


def test_add_rejects_fields_dict_for_multiple_records(collection):
    with pytest.raises(ValueError, match="fields must be a list of dicts"):
        collection.add(
            ids=[1, 2],
            vectors=np.ones((2, DIM), dtype=np.float32),
            fields={"tag": "shared"},
        )


def test_add_rejects_field_count_mismatch(collection):
    with pytest.raises(ValueError, match="fields length"):
        collection.add(
            ids=[1, 2],
            vectors=np.ones((2, DIM), dtype=np.float32),
            fields=[{"tag": "one"}],
        )


def test_add_rejects_non_dict_field_entries(collection):
    with pytest.raises(TypeError, match="fields entries"):
        collection.add(
            ids=[1, 2],
            vectors=np.ones((2, DIM), dtype=np.float32),
            fields=[{"tag": "ok"}, "not-a-dict"],
        )


def test_add_rejects_document_count_mismatch(collection):
    with pytest.raises(ValueError, match="documents length"):
        collection.add(
            ids=[1, 2],
            vectors=np.ones((2, DIM), dtype=np.float32),
            documents=["only one"],
        )


def test_add_rejects_missing_vectors_and_documents(collection):
    with pytest.raises(ValueError, match="requires vectors or documents"):
        collection.add(ids=1)


@pytest.mark.parametrize(
    "bad_batch_size",
    [0, -1, 1.5, "2"],
)
def test_add_rejects_invalid_batch_size(collection, bad_batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        collection.add(
            ids=[1, 2],
            vectors=np.ones((2, DIM), dtype=np.float32),
            batch_size=bad_batch_size,
        )


def test_upsert_rejects_duplicate_ids_before_mutating(collection):
    collection.add(ids=1, vectors=np.zeros(DIM, dtype=np.float32), fields={"tag": "old"})

    with pytest.raises(ValueError, match="duplicate id"):
        collection.upsert(
            ids=[1, 1],
            vectors=np.ones((2, DIM), dtype=np.float32),
            fields=[{"tag": "new-a"}, {"tag": "new-b"}],
        )

    result = collection.query(filter_ids=[1])
    assert result.fields[0]["tag"] == "old"


def test_upsert_rejects_invalid_batch_size(collection):
    with pytest.raises(ValueError, match="batch_size"):
        collection.upsert(
            ids=[1, 2],
            vectors=np.ones((2, DIM), dtype=np.float32),
            batch_size=0,
        )


def test_query_filter_ids_rejects_invalid_id(collection):
    collection.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))
    with pytest.raises(TypeError):
        collection.query(filter_ids=[True])


def test_delete_restore_ignore_missing_ids_without_error(collection):
    collection.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))

    collection.delete([999])
    collection.restore([999])

    assert collection.list_deleted_ids() == []
    assert collection.is_id_exists(1) is True


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "exactly one"),
        ({"vector": np.ones(DIM, dtype=np.float32), "document": "text"}, "exactly one"),
    ],
)
def test_search_requires_exactly_one_query_input(collection, kwargs, match):
    with pytest.raises(ValueError, match=match):
        collection.search(**kwargs)


def test_search_document_path_uses_embedding(monkeypatch, collection):
    from lynse.api import local_client as local_client_module

    def fake_embed_documents(documents):
        assert documents == ["needle"]
        return np.ones((1, DIM), dtype=np.float32)

    monkeypatch.setattr(local_client_module, "embed_documents", fake_embed_documents)

    collection.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))
    result = collection.search(document="needle", k=1)

    assert result.ids.tolist() == [1]


def test_named_vector_operations_reject_missing_ids(populated_collection):
    populated_collection.create_vector_field("image", 3, metric="l2")

    with pytest.raises(Exception):
        populated_collection.add_named_vectors(
            "image",
            np.ones((1, 3), dtype=np.float32),
            [9999],
        )


def test_sparse_vector_operations_reject_missing_ids(populated_collection):
    with pytest.raises(Exception):
        populated_collection.add_sparse_vectors([{1: 1.0}], [9999])


def test_restore_collection_requires_overwrite_for_existing_collection(db, tmp_path):
    original = db.require_collection("snap_col", dim=DIM, drop_if_exists=True)
    original.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))
    original.commit()
    snapshot = tmp_path / "snap_col"
    db.snapshot_collection("snap_col", snapshot)

    with pytest.raises(Exception):
        db.restore_collection("snap_col", snapshot, overwrite=False)

    original.close()
    assert db.restore_collection("snap_col", snapshot, overwrite=True) == {
        "status": "success"
    }


def test_import_collection_requires_overwrite_for_existing_collection(db, tmp_path):
    original = db.require_collection("export_col", dim=DIM, drop_if_exists=True)
    original.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))
    original.commit()
    export = tmp_path / "export_col"
    db.export_collection("export_col", export)

    with pytest.raises(Exception):
        db.import_collection("export_col", export, overwrite=False)

    original.close()
    assert db.import_collection("export_col", export, overwrite=True) == {
        "status": "success"
    }


def test_restore_database_requires_overwrite_for_existing_database(client, tmp_path):
    db = client.create_database("restore_edge_db", drop_if_exists=True)
    coll = db.require_collection("items", dim=DIM, drop_if_exists=True)
    snapshot = tmp_path / "restore_edge_db_snapshot"
    client.snapshot_database("restore_edge_db", snapshot)

    with pytest.raises(Exception):
        client.restore_database("restore_edge_db", snapshot, overwrite=False)

    coll.close()
    assert client.restore_database("restore_edge_db", snapshot, overwrite=True) is None
