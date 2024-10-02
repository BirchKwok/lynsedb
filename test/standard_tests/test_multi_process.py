import lynse
import concurrent.futures


client = lynse.VectorDBClient()


def add_item(vector, id, field):
    db = client.create_database("test_min_vec", drop_if_exists=False)
    collection = db.get_collection('test_collection')
    try:
        with collection.insert_session() as session:
            session.add_item(vector, id=id, field=field, buffer_size=True)
    except Exception as e:
        print(e)


def bulk_add_items(items):
    db = client.create_database("test_min_vec", drop_if_exists=False)
    collection = db.get_collection('test_collection')
    try:
        with collection.insert_session() as session:
            session.bulk_add_items(items)

        print(collection.max_id, collection.shape)
    except Exception as e:
        print(e)


def test_multi_users_bulk_add_items():
    db = client.create_database("test_min_vec", drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    items_a = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    items_b = [
        ([1, 2, 3, 4], 5, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 6, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 7, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 8, {'name': 'Jane Smith'})
    ]

    items_c = [
        ([1, 2, 3, 4], 9, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 10, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 11, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 12, {'name': 'Jane Smith'})
    ]

    items_d = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(bulk_add_items, items_a),
                   executor.submit(bulk_add_items, items_b),
                   executor.submit(bulk_add_items, items_c),
                   executor.submit(bulk_add_items, items_d)]

        concurrent.futures.wait(futures)

    # Only three process can successfully add items
    assert collection.shape == (12, 4)
    db.drop_database()
