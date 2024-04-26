import pytest

from min_vec.api.http_api import app


@pytest.fixture()
def test_client():
    app.config.update({
        "TESTING": True,
    })

    yield app.test_client()


def test_require_collection(test_client):
    url = 'http://localhost:7637/required_collection'
    data = {
        "collection_name": "example_collection",
        "dim": 4,
        "n_clusters": 10,
        "chunk_size": 1024,
        "distance": "L2",
        "index_mode": "IVF-FLAT",
        "dtypes": "float32",
        "use_cache": True,
        "scaler_bits": 8,
        "n_threads": 4,
        "warm_up": True,
        "drop_if_exists": True
    }
    response = test_client.post(url, json=data)
    print(response.json)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params":
        {"collection_name": "example_collection",
         "dim": 4, "n_clusters": 10, "chunk_size": 1024, "distance": "L2",
         "index_mode": "IVF-FLAT", "dtypes": "float32", "use_cache": True, "scaler_bits": 8,
         "n_threads": 4, "warm_up": True, "drop_if_exists": True}}


def test_add_item(test_client):
    url = 'http://localhost:7637/add_item'
    data = {
        "collection_name": "example_collection",
        "item": {
            "vector": [0.1, 0.2, 0.3, 0.4],
            "id": 1,
            "field": {
                "name": "example",
                "age": 18
            }
        }
    }
    response = test_client.post(url, json=data)
    print(response.json)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params":
        {"collection_name": "example_collection", "item":
            {"vector": [0.1, 0.2, 0.3, 0.4], "id": 1, "field": {"name": "example", "age": 18}}}}


def test_bulk_add_items(test_client):
    url = 'http://localhost:7637/bulk_add_items'
    data = {
        "collection_name": "example_collection",
        "items": [
            {
                "vector": [0.1, 0.4, 0.3, 0.6],
                "id": 2,
                "field": {
                    "name": "example2",
                    "age": 18
                }
            },
            {
                "vector": [0.2, 0.3, 0.4, 0.5],
                "id": 3,
                "field": {
                    "name": "example3",
                    "age": 19
                }
            }
        ]
    }
    response = test_client.post(url, json=data)
    assert response.status_code == 200
    assert response.json == {
        "status": "success", "params":
            {
                "collection_name": "example_collection",
                "ids": [2, 3],
                "items": [
                    {"vector": [0.1, 0.4, 0.3, 0.6], "id": 2, "field": {"name": "example2", "age": 18}},
                    {"vector": [0.2, 0.3, 0.4, 0.5], "id": 3, "field": {"name": "example3", "age": 19}}
                ]
            }
    }


def test_collection_shape(test_client):
    url = 'http://localhost:7637/collection_shape'
    data = {
        "collection_name": "example_collection"
    }
    response = test_client.post(url, json=data)
    assert response.status_code == 200
    assert response.json == {"status": "success",
                             "params": {"collection_name": "example_collection", "shape": [3, 4]}}


def test_query(test_client):
    url = 'http://localhost:7637/query'

    data = {
        "collection_name": "example_collection",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 10,
        'distance': 'cosine',
        "query_filter": {
            "must": [
                {
                    "field": "name",
                    "operator": "eq",
                    "value": "example"
                }
            ],
            "any": [
                {
                    "field": "age",
                    "operator": "eq",
                    "value": 18
                }
            ]
        }
    }

    response = test_client.post(url, json=data)
    assert response.status_code == 200

    rjson = response.json
    rjson['params']['items']['query time'] = 0.0
    rjson['params']['items']['scores'] = [1]

    assert rjson == {"status": "success", "params":
        {"collection_name": "example_collection", "items": {
            "k": 10, "ids": [1], "scores": [1],
            "distance": 'cosine', "query time": 0.0
        }}}


def test_drop_collection(test_client):
    url = 'http://localhost:7637/drop_collection'
    data = {
        "collection_name": "example_collection"
    }
    response = test_client.post(url, json=data)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params": {"collection_name": "example_collection"}}


def test_show_collections(test_client):
    url = 'http://localhost:7637/show_collections'
    response = test_client.get(url)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params": {"collections": []}}


def test_drop_database(test_client):
    url = 'http://localhost:7637/drop_database'
    response = test_client.get(url)
    assert response.status_code == 200
    assert response.json == {"status": "success"}


def test_database_exists(test_client):
    url = 'http://localhost:7637/database_exists'
    response = test_client.get(url)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params": {"exists": False}}
