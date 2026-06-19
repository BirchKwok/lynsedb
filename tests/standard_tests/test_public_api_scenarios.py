"""End-to-end public API scenarios across common user workflows."""
import numpy as np
import pytest

import lynse


DIM = 8


def test_string_ids_round_trip_through_query_search_delete_restore_and_compact(db):
    coll = db.require_collection("string_ids", dim=4, drop_if_exists=True, default_index=None)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    coll.add(
        ids=["doc-a", "doc-b", "doc-c"],
        vectors=vectors,
        fields=[
            {"kind": "alpha", "rank": 1},
            {"kind": "beta", "rank": 2},
            {"kind": "alpha", "rank": 3},
        ],
    )

    assert coll.search(vectors[0], k=1).ids.tolist() == ["doc-a"]
    assert coll.query(where='"kind" = \'alpha\'').ids.tolist() == ["doc-a", "doc-c"]
    by_id = coll.query_vectors(filter_ids=["doc-b"])
    assert by_id.ids.tolist() == ["doc-b"]
    assert np.allclose(by_id.vectors[0], vectors[1])

    coll.delete(["doc-b"])
    assert coll.list_deleted_ids() == ["doc-b"]
    assert "doc-b" not in coll.search(vectors[1], k=3).ids.tolist()

    coll.restore(["doc-b"])
    assert coll.list_deleted_ids() == []
    assert coll.search(vectors[1], k=1).ids.tolist() == ["doc-b"]

    coll.delete(["doc-a"])
    assert coll.compact() == 1
    with pytest.raises(ValueError, match="does not exist"):
        coll.query(filter_ids=["doc-a"], return_ids_only=True)
    assert coll.query(filter_ids=["doc-b", "doc-c"], return_ids_only=True).ids.tolist() == [
        "doc-b",
        "doc-c",
    ]


def test_auto_ids_continue_from_internal_sequence_after_string_ids(db):
    coll = db.require_collection("mixed_auto_ids", dim=3, drop_if_exists=True, default_index=None)

    coll.add(ids=["first", "second"], vectors=np.ones((2, 3), dtype=np.float32))
    generated = coll.add(vectors=np.zeros((2, 3), dtype=np.float32))

    assert generated == [2, 3]
    assert coll.head(4).ids.tolist() == ["first", "second", 2, 3]


def test_documents_are_attached_to_fields_and_searchable_by_bm25(
    db, monkeypatch
):
    from lynse.api import local_client as local_client_module

    def fake_embed_documents(documents):
        return np.eye(len(documents), DIM, dtype=np.float32)

    monkeypatch.setattr(local_client_module, "embed_documents", fake_embed_documents)

    coll = db.require_collection("documents", dim=DIM, drop_if_exists=True, default_index=None)
    coll.add(
        ids=["doc-1", "doc-2"],
        documents=["alpha release notes", "beta migration guide"],
        fields=[{"category": "release"}, {"category": "docs"}],
    )

    fields = coll.query(filter_ids=["doc-1", "doc-2"]).fields
    assert fields[0]["document"] == "alpha release notes"
    assert fields[1]["document"] == "beta migration guide"

    result = coll.bm25_search("migration", k=1, text_fields=["document"], return_fields=True)
    assert result.ids.tolist() == ["doc-2"]
    assert result.fields[0]["category"] == "docs"


def test_collection_snapshot_restore_to_new_name_preserves_rows_and_fields(db, tmp_path):
    source = db.require_collection("snap_source", dim=4, drop_if_exists=True, default_index=None)
    source.add(
        ids=["a", "b"],
        vectors=np.eye(2, 4, dtype=np.float32),
        fields=[{"group": "x"}, {"group": "y"}],
    )
    source.commit()
    snapshot = tmp_path / "snap_source"
    source.snapshot_to(snapshot)

    assert db.restore_collection("snap_restored", snapshot) == {"status": "success"}
    restored = db.get_collection("snap_restored")

    assert restored.shape == (2, 4)
    assert restored.head(2).ids.tolist() == ["a", "b"]
    assert restored.query(where='"group" = \'y\'').ids.tolist() == ["b"]


def test_collection_export_import_to_new_name_preserves_external_ids(db, tmp_path):
    source = db.require_collection("export_source", dim=4, drop_if_exists=True, default_index=None)
    vectors = np.eye(3, 4, dtype=np.float32)
    source.add(
        ids=[10, 11, 12],
        vectors=vectors,
        fields=[{"label": "a"}, {"label": "b"}, {"label": "c"}],
    )
    source.commit()
    export = tmp_path / "export_source"
    source.export_to(export)

    assert db.import_collection("export_imported", export) == {"status": "success"}
    imported = db.get_collection("export_imported")

    result = imported.search(vectors[2], k=1, return_fields=True)
    assert result.ids.tolist() == [12]
    assert result.fields[0]["label"] == "c"


def test_read_only_client_can_read_but_rejects_writes(tmp_root):
    writer = lynse.VectorDBClient(uri=tmp_root)
    db = writer.create_database("readonly_db", drop_if_exists=True)
    coll = db.require_collection("items", dim=DIM, drop_if_exists=True)
    coll.add(ids=1, vectors=np.ones(DIM, dtype=np.float32), fields={"tag": "stored"})
    coll.commit()
    coll.close()
    writer.close()

    reader = lynse.VectorDBClient(uri=tmp_root, read_only=True)
    try:
        ro_coll = reader.get_database("readonly_db").get_collection("items")
        assert ro_coll.is_read_only is True
        assert ro_coll.search(np.ones(DIM, dtype=np.float32), k=1).ids.tolist() == [1]
        assert ro_coll.query(filter_ids=[1]).fields[0]["tag"] == "stored"

        with pytest.raises(RuntimeError, match="read-only"):
            ro_coll.add(ids=2, vectors=np.zeros(DIM, dtype=np.float32))
    finally:
        reader.close()


def test_collection_close_is_idempotent_and_prevents_accidental_reuse(collection):
    collection.add(ids=1, vectors=np.ones(DIM, dtype=np.float32))

    assert collection.close() == {"status": "success"}
    assert collection.close() == {"status": "success"}

    with pytest.raises(AttributeError):
        collection.search(np.ones(DIM, dtype=np.float32), k=1)


def test_drop_and_recreate_collection_resets_data_and_description(db):
    first = db.require_collection(
        "recreated",
        dim=DIM,
        drop_if_exists=True,
        description="old",
        default_index=None,
    )
    first.add(ids=1, vectors=np.ones(DIM, dtype=np.float32), fields={"version": "old"})
    first.close()

    second = db.require_collection(
        "recreated",
        dim=4,
        drop_if_exists=True,
        description="new",
        default_index=None,
    )

    assert second.shape == (0, 4)
    assert db._manager.get_collection_config(db.database_name, "recreated")["description"] == "new"
    with pytest.raises(ValueError, match="does not exist"):
        second.query(filter_ids=[1], return_ids_only=True)


def test_database_snapshot_restore_preserves_multiple_collections(client, tmp_path):
    db = client.create_database("multi_snapshot_db", drop_if_exists=True)
    left = db.require_collection("left", dim=2, drop_if_exists=True, default_index=None)
    right = db.require_collection("right", dim=3, drop_if_exists=True, default_index=None)
    left.add(ids="left-id", vectors=np.array([1.0, 0.0], dtype=np.float32))
    right.add(ids="right-id", vectors=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    left.commit()
    right.commit()

    snapshot = tmp_path / "multi_snapshot"
    client.snapshot_database("multi_snapshot_db", snapshot)
    left.close()
    right.close()
    client.drop_database("multi_snapshot_db")

    client.restore_database("multi_snapshot_db", snapshot)
    restored = client.get_database("multi_snapshot_db")

    assert set(restored.show_collections()) == {"left", "right"}
    assert restored.get_collection("left").head(1).ids.tolist() == ["left-id"]
    assert restored.get_collection("right").head(1).ids.tolist() == ["right-id"]
