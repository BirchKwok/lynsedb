"""Server CLI and authenticated remote-mode tests."""
import pytest

import lynse


def test_lynse_run_with_api_key_authenticates(remote_server_with_auth, unique_name):
    client = lynse.VectorDBClient(
        remote_server_with_auth.base_url,
        api_key=remote_server_with_auth.api_key,
    )
    db = client.create_database(unique_name, drop_if_exists=True)
    collection = db.require_collection("auth_smoke", dim=2, drop_if_exists=True)

    with collection.insert_session() as session:
        session.add(ids=1, vectors=[1.0, 0.0], fields={"tag": "secured"})

    result = collection.search([1.0, 0.0], k=1, return_fields=True)
    assert result.ids.tolist() == [1]
    assert result.fields == [{"tag": "secured"}]


def test_missing_api_key_is_rejected(remote_server_with_auth):
    with pytest.raises(ConnectionError, match="Authentication failed"):
        lynse.VectorDBClient(remote_server_with_auth.base_url)


def test_wrong_api_key_is_rejected(remote_server_with_auth):
    with pytest.raises(ConnectionError, match="Authentication failed"):
        lynse.VectorDBClient(remote_server_with_auth.base_url, api_key="wrong-secret")
