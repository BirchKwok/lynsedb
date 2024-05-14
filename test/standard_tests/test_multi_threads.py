from test import MinVectorDB, ExclusiveMinVectorDB
import concurrent.futures


def add_item(id, vector, field):
    db = MinVectorDB('test_min_vec')
    collection = db.get_collection('test_collection')
    collection.add_item(vector, id=id, field=field)
    collection.commit()


def bulk_add_items(items):
    db = MinVectorDB('test_min_vec')
    collection = db.get_collection('test_collection')
    collection.bulk_add_items(items)
    collection.commit()


def test_multi_thread_bulk_add_items():
    db = MinVectorDB('test_min_vec')
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
    db.drop_database()


def test_multi_thread_add_item():
    db = MinVectorDB('test_min_vec')
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(add_item, 1, [1, 2, 3, 4], {'name': 'John Doe'}),
                   executor.submit(add_item, 2, [5, 6, 7, 8], {'name': 'Jane Doe'}),
                   executor.submit(add_item, 3, [9, 10, 11, 12], {'name': 'John Smith'}),
                   executor.submit(add_item, 4, [13, 14, 15, 16], {'name': 'Jane Smith'})]

        concurrent.futures.wait(futures)

    assert collection.shape == (4, 4)

    db.drop_database()
