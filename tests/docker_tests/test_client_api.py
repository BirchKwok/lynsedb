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
        session.bulk_add_items(
            [
                (np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 1, {"tag": "alpha"}),
                (np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), 2, {"tag": "beta"}),
            ],
            enable_progress_bar=False,
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
        session.bulk_add_items(
            [
                (np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), 10, {"tag": "keep"}),
                (np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), 11, {"tag": "drop"}),
            ],
            enable_progress_bar=False,
        )

    collection.delete_items([11])
    assert collection.list_deleted_ids() == [11]
    assert collection.stats()["n_tombstoned"] == 1

    collection.restore_items([11])
    assert collection.list_deleted_ids() == []
    assert collection.stats()["n_live"] == 2
