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


def search(vector, k, search_filter=None):
    db = client.create_database("test_min_vec", drop_if_exists=False)
    collection = db.get_collection('test_collection')
    try:
        return collection.search(vector, k=k, search_filter=search_filter)
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


def test_multi_users_add_item():
    db = client.create_database("test_min_vec", drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    items = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(add_item, items[0][0], items[0][1], items[0][2]),
                   executor.submit(add_item, items[1][0], items[1][1], items[1][2]),
                   executor.submit(add_item, items[2][0], items[2][1], items[2][2]),
                   executor.submit(add_item, items[3][0], items[3][1], items[3][2])]

        concurrent.futures.wait(futures)

    assert collection.shape == (4, 4)

    db.drop_database()


def test_multi_users_search():
    db = client.create_database("test_min_vec", drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    items = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(add_item, items[0][0], items[0][1], items[0][2]),
                   executor.submit(add_item, items[1][0], items[1][1], items[1][2]),
                   executor.submit(add_item, items[2][0], items[2][1], items[2][2]),
                   executor.submit(add_item, items[3][0], items[3][1], items[3][2])]

        concurrent.futures.wait(futures)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(search, items[0][0], 3, None),
                   executor.submit(search, items[1][0], 3, None),
                   executor.submit(search, items[2][0], 3, None),
                   executor.submit(search, items[3][0], 3, None)]

        concurrent.futures.wait(futures)

    assert len(futures[0].result()[0]) == 3
    assert len(futures[1].result()[0]) == 3
    assert len(futures[2].result()[0]) == 3
    assert len(futures[3].result()[0]) == 3

    db.drop_database()


def test_multi_users_search_with_filter():
    db = client.create_database("test_min_vec", drop_if_exists=True)
    collection = db.require_collection('test_collection', dim=4, drop_if_exists=True)

    items = [
        ([1, 2, 3, 4], 1, {'name': 'John Doe'}),
        ([5, 6, 7, 8], 2, {'name': 'Jane Doe'}),
        ([9, 10, 11, 12], 3, {'name': 'John Smith'}),
        ([13, 14, 15, 16], 4, {'name': 'Jane Smith'})
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(add_item, items[0][0], items[0][1], items[0][2]),
                   executor.submit(add_item, items[1][0], items[1][1], items[1][2]),
                   executor.submit(add_item, items[2][0], items[2][1], items[2][2]),
                   executor.submit(add_item, items[3][0], items[3][1], items[3][2])]

        concurrent.futures.wait(futures)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(search, items[0][0], 3, search_filter=':name: == "John Doe"'),
                   executor.submit(search, items[1][0], 3, search_filter=':name: == "Jane Doe"'),
                   executor.submit(search, items[2][0], 3, search_filter=':name: == "John Smith"'),
                   executor.submit(search, items[3][0], 3, search_filter=':name: == "Jane Smith"')]

        concurrent.futures.wait(futures)

    assert len(futures[0].result()[0]) == 1
    assert len(futures[1].result()[0]) == 1
    assert len(futures[2].result()[0]) == 1
    assert len(futures[3].result()[0]) == 1

    db.drop_database()
