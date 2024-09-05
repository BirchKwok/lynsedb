# The address of the docker container running the service is http://localhost:5403
import time
import numpy as np

from test import VectorDBClient
from test import HTTPClient, Collection


def test_initialization():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)

    assert isinstance(db, HTTPClient)


def test_create_collection():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    assert isinstance(collection, Collection)
    assert collection._collection_name == 'test_collection'


def test_add_item():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=False)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    item = {
        "vector": [0.01, 0.34, 0.74, 0.31],
        "id": 1,
        "field": {
            'field': 'test_1',
            'order': 0
        }
    }

    with collection.insert_session():
        id = collection.add_item(**item)
    assert id == 1


def test_bulk_add_items():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=False)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    time.sleep(3)
    with collection.insert_session():
        ids = collection.bulk_add_items([([0.01, 0.34, 0.74, 0.31], 1, {'field': 'test_1', 'order': 0}),
                                            ([0.36, 0.43, 0.56, 0.12], 2, {'field': 'test_1', 'order': 1}),
                                            ([0.03, 0.04, 0.10, 0.51], 3, {'field': 'test_2', 'order': 2}),
                                            ([0.11, 0.44, 0.23, 0.24], 4, {'field': 'test_2', 'order': 3}),
                                            ([0.91, 0.43, 0.44, 0.67], 5, {'field': 'test_2', 'order': 4}),
                                            ([0.92, 0.12, 0.56, 0.19], 6, {'field': 'test_3', 'order': 5}),
                                            ([0.18, 0.34, 0.56, 0.71], 7, {'field': 'test_1', 'order': 6}),
                                            ([0.01, 0.33, 0.14, 0.31], 8, {'field': 'test_2', 'order': 7}),
                                            ([0.71, 0.75, 0.91, 0.82], 9, {'field': 'test_3', 'order': 8}),
                                            ([0.75, 0.44, 0.38, 0.75], 10, {'field': 'test_1', 'order': 9})])
    assert ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_query():
    import operator

    from lynse.core_components.fields_cache.filter import Filter, FieldCondition, MatchField, MatchID

    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=False)
    collection = db.get_collection('test_collection')
    collection.build_index("Flat-Cos")
    ids, scores, fields = collection.search(
        vector=[0.36, 0.43, 0.56, 0.12],
        k=10,
        search_filter=Filter(
            must=[
                FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
            ],
            any=[
                FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
                FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
            ]
        )
    )
    print(ids)
    assert np.array_equal(ids, np.array([2, 1, 10]))


def test_drop_collection():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=False)
    db.drop_collection('test_collection')
    print(db.show_collections())
    assert 'test_collection' not in db.show_collections()
