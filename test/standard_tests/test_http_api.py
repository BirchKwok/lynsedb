import pytest

from min_vec.api.http_api.http_api import app
from min_vec.api.http_api.client_api import pack_data


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

    vector = [0.1, 0.2, 0.3, 0.4]
    data = {
        "collection_name": "example_collection",
        "item": {
            "vector": vector,
            "id": 1,
            "field": {
                "name": "example",
                "age": 18
            },
        }
    }

    header = {
        "Content-Type": "application/msgpack"
    }

    response = test_client.post(url, data=pack_data(data), headers=header)
    print(response.json)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params":
        {"collection_name": "example_collection", "item":
            {"id": 1}}}

    url = 'http://localhost:7637/commit'
    data = {
        "collection_name": "example_collection"
    }
    response = test_client.post(url, json=data)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params": {"collection_name": "example_collection"}}


def test_bulk_add_items(test_client):
    url = 'http://localhost:7637/bulk_add_items'
    v1 = [0.1, 0.2, 0.3, 0.4]
    v2 = [0.2, 0.3, 0.4, 0.5]
    data = {
        "collection_name": "example_collection",
        "items": [
            {
                "vector": v1,
                "id": 2,
                "field": {
                    "name": "example2",
                    "age": 18
                },
            },
            {
                "vector": v2,
                "id": 3,
                "field": {
                    "name": "example3",
                    "age": 19
                },
            }
        ]
    }
    response = test_client.post(url, json=data)

    assert response.status_code == 200
    assert response.json == {
        "status": "success", "params":
            {
                "collection_name": "example_collection",
                "ids": [2, 3]
            }
    }

    url = 'http://localhost:7637/commit'
    data = {
        "collection_name": "example_collection"
    }
    response = test_client.post(url, json=data)
    assert response.status_code == 200
    assert response.json == {"status": "success", "params": {"collection_name": "example_collection"}}



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

    # 内部解析示例
    # {
    #     'must_fields': [{
    #         'key': condition.key,
    #         'matcher': {
    #             'value': condition.matcher.value,
    #             'comparator': condition.matcher.comparator.__name__
    #         }
    #     } for condition in self.must_fields] if self.must_fields else [],
    #     'any_fields': [{
    #         'key': condition.key,
    #         'matcher': {
    #             'value': condition.matcher.value,
    #             'comparator': condition.matcher.comparator.__name__
    #         }
    #     } for condition in self.any_fields] if self.any_fields else [],
    #     'must_not_fields': [{
    #         'key': condition.key,
    #         'matcher': {
    #             'value': condition.matcher.value,
    #             'comparator': condition.matcher.comparator.__name__
    #         }
    #     } for condition in self.must_not_fields] if self.must_not_fields else [],
    #     'must_ids': [{
    #         'matcher': {
    #             'ids': condition.matcher.indices.tolist()
    #         }
    #     } for condition in self.must_ids] if self.must_ids else [],
    #     'any_ids': [{
    #         'matcher': {
    #             'ids': condition.matcher.indices.tolist()
    #         }
    #     } for condition in self.any_ids] if self.any_ids else [],
    #     'must_not_ids': [{
    #         'matcher': {
    #             'ids': condition.matcher.indices.tolist()
    #         }
    #     } for condition in self.must_not_ids] if self.must_not_ids else []
    # }

    data = {
        "collection_name": "example_collection",
        "vector": [0.1, 0.2, 0.3, 0.4],
        "k": 10,
        'distance': 'cosine',
        "query_filter": {
            "must_fields": [
                {
                    "key": "name",
                    "matcher": {
                        "value": "example",
                        "comparator": "eq"
                    }
                }
            ],
            "any_fields": [
                {
                    "key": "age",
                    "matcher": {
                        "value": 18,
                        "comparator": "eq"
                    }
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
