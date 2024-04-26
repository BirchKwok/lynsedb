# The address of the docker container running the service is http://localhost:5403
import numpy as np

from test import MinVectorDB
from test import MinVectorDBHTTPClient, Collection


root_url = 'http://localhost:5403'


def test_initialization():
    db = MinVectorDB(root_url)

    assert isinstance(db, MinVectorDBHTTPClient)


def test_create_collection():
    db = MinVectorDB(root_url)
    collection = db.require_collection('test_collection')
    assert isinstance(collection, Collection)
    assert collection._collection_name == 'test_collection'


def test_add_item():
    db = MinVectorDB(root_url)
    collection = db.create_collection('test_collection')
    item = {
        "vector": [0.01, 0.34, 0.74, 0.31],
        "id": 1,
        "field": {
            'field': 'test_1',
            'order': 0
        }
    }
    id = collection.add_item(**item)
    assert id == 1


def test_bulk_add_items():
    db = MinVectorDB(root_url)
    collection = db.create_collection('test_collection')
    ids = collection.bulk_add_items([([0.36, 0.43, 0.56, 0.12], 2, {'field': 'test_1', 'order': 1}),
                                 ([0.03, 0.04, 0.10, 0.51], 3, {'field': 'test_2', 'order': 2}),
                                 ([0.11, 0.44, 0.23, 0.24], 4, {'field': 'test_2', 'order': 3}),
                                 ([0.91, 0.43, 0.44, 0.67], 5, {'field': 'test_2', 'order': 4}),
                                 ([0.92, 0.12, 0.56, 0.19], 6, {'field': 'test_3', 'order': 5}),
                                 ([0.18, 0.34, 0.56, 0.71], 7, {'field': 'test_1', 'order': 6}),
                                 ([0.01, 0.33, 0.14, 0.31], 8, {'field': 'test_2', 'order': 7}),
                                 ([0.71, 0.75, 0.91, 0.82], 9, {'field': 'test_3', 'order': 8}),
                                 ([0.75, 0.44, 0.38, 0.75], 10, {'field': 'test_1', 'order': 9})])
    assert ids == [2, 3]


def test_query():
    import operator

    from min_vec.structures.filter import Filter, FieldCondition, MatchField, IDCondition, MatchID

    db = MinVectorDB(root_url)
    collection = db.create_collection('test_collection')
    ids, scores = collection.query(
        vector=[0.01, 0.34, 0.74, 0.31],
        k=10,
        query_filter=Filter(
            must=[
                FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
            ],
            any=[

                FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
                IDCondition(MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
            ]
        )
    )

    assert ids == np.array([2, 1, 4, 5, 10, 3])
    assert scores == np.array([1., 0.86097705, 0.85727406, 0.813797, 0.78595245, 0.34695023])


def test_drop_collection():
    db = MinVectorDB(root_url)
    collection = db.require_collection('test_collection')
    collection.drop_collection()
    assert 'test_collection' not in db.show_collections()