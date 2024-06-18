import lynse
import concurrent.futures

client = lynse.VectorDBClient()


def add_item(id, vector, field):
    db = client.create_database("test_min_vec", drop_if_exists=False)
    collection = db.get_collection('test_collection')
    collection.add_item(vector, id=id, field=field)
    collection.commit()


def bulk_add_items(items):
    db = client.create_database("test_min_vec", drop_if_exists=False)
    collection = db.get_collection('test_collection')
    collection.bulk_add_items(items)
    collection.commit()


def test_multi_thread_bulk_add_items():
    db = client.create_database("test_min_vec", drop_if_exists=False)
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
