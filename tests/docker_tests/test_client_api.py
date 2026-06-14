"""Remote HTTP client smoke tests."""
import numpy as np

import lynse
from lynse.api.http_api.client_api import HTTPClient


def test_http_client_accepts_https_uri():
    client = HTTPClient("https://example.com", "demo_db")
    assert client.uri == "https://example.com"


def test_remote_round_trip_search(remote_server, unique_name):
    client = lynse.VectorDBClient(remote_server.base_url)
    db = client.create_database(unique_name, drop_if_exists=True)
    collection = db.require_collection("remote_smoke", dim=4, drop_if_exists=True)

    with collection.insert_session() as session:
        session.add(
            ids=[1, 2],
            vectors=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            fields=[{"tag": "alpha"}, {"tag": "beta"}],
        )

    result = collection.search([1.0, 0.0, 0.0, 0.0], k=1, return_fields=True)

    assert unique_name in client.list_databases()
    assert collection.shape == (2, 4)
    assert result.ids.tolist() == [1]
    assert result.fields == [{"tag": "alpha"}]


def test_remote_delete_restore_and_stats(remote_server, unique_name):
    client = lynse.VectorDBClient(remote_server.base_url)
    db = client.create_database(unique_name, drop_if_exists=True)
    collection = db.require_collection("restore_smoke", dim=4, drop_if_exists=True)

    with collection.insert_session() as session:
        session.add(
            ids=[10, 11],
            vectors=np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            fields=[{"tag": "keep"}, {"tag": "drop"}],
        )

    collection.delete([11])
    assert collection.list_deleted_ids() == [11]
    assert collection.stats()["n_tombstoned"] == 1

    collection.restore([11])
    assert collection.list_deleted_ids() == []
    assert collection.stats()["n_live"] == 2


def test_remote_query_without_filter_returns_empty(remote_server, unique_name):
    client = lynse.VectorDBClient(remote_server.base_url)
    db = client.create_database(unique_name, drop_if_exists=True)
    collection = db.require_collection("query_contract", dim=4, drop_if_exists=True)

    with collection.insert_session() as session:
        session.add(ids=1, vectors=[1.0, 0.0, 0.0, 0.0], fields={"tag": "alpha"})

    assert len(collection.query().ids) == 0
    assert collection.query_vectors().vectors.shape == (0, 4)


def test_remote_named_vector_index_uses_field_endpoint(remote_server, unique_name):
    client = lynse.VectorDBClient(remote_server.base_url)
    db = client.create_database(unique_name, drop_if_exists=True)
    collection = db.require_collection("named_vector_remote", dim=4, drop_if_exists=True)

    with collection.insert_session() as session:
        session.add(
            ids=[1, 2],
            vectors=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            fields=[{"tag": "one"}, {"tag": "two"}],
        )

    collection.create_vector_field("image", dim=3, metric="l2")
    collection.add_named_vectors(
        "image",
        np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32),
        ids=[1, 2],
    )
    collection.build_index("IVF-L2", field_name="image", n_clusters=2)

    fields = collection.list_vector_fields()
    image_field = next(field for field in fields if field["name"] == "image")
    assert image_field["index_mode"] == "IVF-L2"

    result = collection.search([1.1, 0.0, 0.0], k=1, vector_field="image")
    assert result.ids.tolist() == [1]

    collection.remove_index(field_name="image")
    image_field = next(field for field in collection.list_vector_fields() if field["name"] == "image")
    assert image_field["index_mode"] == "FLAT-L2"


def test_remote_search_forwards_approx_options(remote_server, unique_name):
    client = lynse.VectorDBClient(remote_server.base_url)
    db = client.create_database(unique_name, drop_if_exists=True)
    collection = db.require_collection("approx_remote", dim=128, drop_if_exists=True)

    vectors = np.zeros((3, 128), dtype=np.float32)
    vectors[1, 96:] = 0.7
    with collection.insert_session() as session:
        session.add(
            ids=list(range(len(vectors))),
            vectors=vectors,
            fields=[{"tag": str(i)} for i in range(len(vectors))],
        )
    collection.build_index("FLAT-L2")

    query = np.zeros(128, dtype=np.float32)
    query[96:] = 1.0
    exact = collection.search(query, k=1, approx=False)
    approx = collection.search(query, k=1, approx=True, eps=0.5)

    assert int(approx.ids[0]) == int(exact.ids[0]) == 1
    assert not np.isclose(float(exact.distances[0]), float(approx.distances[0]))
    assert np.isclose(float(approx.distances[0]) % 0.5, 0.0)
