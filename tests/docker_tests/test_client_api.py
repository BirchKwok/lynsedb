import concurrent.futures
import time

from test import VectorDBClient, HTTPClient


def test_initialization():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)

    assert isinstance(db, HTTPClient)


def test_create_collection():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    assert collection._collection_name == 'test_collection'


def test_add_item():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)
    time.sleep(3)
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
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=False)
    collection = db.get_collection('test_collection')
    with collection.insert_session() as session:
        session.add_item(vector, id=id, field=field, buffer_size=True)


def bulk_add_items(items):
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=False)
    collection = db.get_collection('test_collection')
    collection.bulk_add_items(items)
    collection.commit()


def test_multi_users_bulk_add_items():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)
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


def test_multi_thread_bulk_add_items():
    client = VectorDBClient('http://localhost:7637')
    db = client.create_database('test_db', drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    items = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(bulk_add_items, items),
                   executor.submit(bulk_add_items, items),
                   executor.submit(bulk_add_items, items),
                   executor.submit(bulk_add_items, items)]

        concurrent.futures.wait(futures)

    assert collection.shape == (4, 4)
