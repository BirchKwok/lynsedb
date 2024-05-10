import concurrent.futures

from test import MinVectorDB, MinVectorDBHTTPClient


def test_initialization():
    db = MinVectorDB('http://localhost:7637')

    assert isinstance(db, MinVectorDBHTTPClient)


def test_create_collection():
    db = MinVectorDB('http://localhost:7637')
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    assert collection._collection_name == 'test_collection'


def test_add_item():
    db = MinVectorDB('http://localhost:7637')
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    with collection.insert_session():
        id = collection.add_item([0.01, 0.34, 0.74, 0.31], id=1, field={'name': 'John Doe'})
        assert id == 1
        id = collection.add_item([0.36, 0.43, 0.56, 0.12], id=2, field={'name': 'Jane Doe'})
        assert id == 2
        id = collection.add_item([0.03, 0.04, 0.10, 0.51], id=3, field={'name': 'John Smith'})
        assert id == 3
        id = collection.add_item([0.11, 0.44, 0.23, 0.24], id=4, field={'name': 'Jane Smith'})
        assert id == 4

    assert collection.shape == (4, 4)


def add_item(id, vector, field):
    db = MinVectorDB('http://localhost:7637')
    collection = db.get_collection('test_collection')
    collection.add_item(vector, id=id, field=field)
    collection.commit()


def test_multi_users_add_item():
    db = MinVectorDB('http://localhost:7637')
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(add_item, 1, [1, 2, 3, 4], {'name': 'John Doe'}),
                   executor.submit(add_item, 2, [5, 6, 7, 8], {'name': 'Jane Doe'}),
                   executor.submit(add_item, 3, [9, 10, 11, 12], {'name': 'John Smith'}),
                   executor.submit(add_item, 4, [13, 14, 15, 16], {'name': 'Jane Smith'})]

        concurrent.futures.wait(futures)

    assert collection.shape == (4, 4)


def bulk_add_items(items):
    db = MinVectorDB('http://localhost:7637')
    collection = db.get_collection('test_collection')
    collection.bulk_add_items(items)
    collection.commit()


def test_multi_users_bulk_add_items():
    db = MinVectorDB('http://localhost:7637')
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    items = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(bulk_add_items, items),
                   executor.submit(bulk_add_items, items),
                   executor.submit(bulk_add_items, items),
                   executor.submit(bulk_add_items, items)]

        concurrent.futures.wait(futures)

    assert collection.shape == (4, 4)
