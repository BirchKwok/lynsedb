import os
import shutil
import time
from pathlib import Path

from test import MinVectorDB
import numpy as np


def get_database(dim=100, database_path='test_min_vec.mvdb', chunk_size=1000, dtypes=np.float32):
    if Path('.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])).exists():
        shutil.rmtree(Path('.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])))

    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)
    if database._database_chunk_path:
        database.delete()
    database._database_chunk_path = []
    return database


def get_database_for_query(*args, with_field=True, field_prefix='test_', **kwargs):
    database = get_database(*args, **kwargs)
    np.random.seed(2023)
    with database.insert_session():
        items = []
        for i in range(100):
            items.append((np.random.random(100), i, field_prefix + str(i // 10) if with_field else None))

        database.bulk_add_items(items)

    return database


def test_database_initialization():
    database = get_database()
    assert isinstance(database.database, np.ndarray)
    assert database.database.shape == (1, 100)
    assert database.database.sum() == 0
    assert database.fields == []
    assert database.indices == []
    assert database.chunk_size == 1000
    assert database.dtypes == np.float32
    assert database._database_chunk_path == []


def test_add_single_item_without_id_and_field():
    database = get_database()
    id = database.add_item(np.ones(100), save_immediately=True)

    assert database.database.sum() == 0
    assert database.fields == []
    assert database.indices == []

    database.commit()
    assert database.shape == (1, 100)
    database.delete()


def test_add_single_item_with_id_and_field():
    database = get_database()
    id = database.add_item(np.ones(100), id=1, field="test", save_immediately=True)

    assert database.database.sum() == 0
    assert database.fields == []
    assert database.indices == []
    assert id == 1

    database.commit()

    assert database.shape == (1, 100)

    database.delete()


def test_bulk_add_item_without_id_and_field():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100),))

    database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    database.commit()
    assert database.shape == (101, 100)
    database.delete()


def test_add_single_item_with_vector_normalize():
    database = get_database()
    id = database.add_item(np.random.random(100), id=1, field="test", normalize=True, save_immediately=True)

    assert database.fields == []
    assert database.indices == []
    assert id == 1

    database.commit()

    assert database.shape == (1, 100)

    database.delete()


def test_add_bulk_item_with_id_and_field():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), i, "test_" + str(i // 10)))

    database.bulk_add_items(items)

    assert database.fields == []
    assert database.indices == []

    database.commit()

    assert database.shape == (101, 100)

    database.delete()


def test_add_bulk_item_with_normalize():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), i, "test_" + str(i // 10)))

    database.bulk_add_items(items, normalize=True, save_immediately=True)

    assert database.fields == []
    assert database.indices == []

    database.commit()

    assert database.shape == (101, 100)
    database.delete()


def test_add_bulk_item_with_id_and_chinese_field():
    database = get_database()
    items = []
    for i in range(100):
        items.append((np.ones(100), i, "测试_" + str(i // 10)))

    database.bulk_add_items(items, save_immediately=True)

    assert database.fields == []
    assert database.indices == []

    database.commit()

    assert database.shape == (100, 100)

    database.delete()


def test_query_without_field():
    database = get_database_for_query(with_field=False)
    vec = np.random.random(100)
    n, d = database.query(vec, k=12, field=None)

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)

    database.delete()


def test_query_with_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='test_1')

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_normalize():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='test_1', normalize=True)

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_list_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(vec, k=12, field=['test_1', 'test_7'])

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_with_chinese_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='测试_1')

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_chinese_list_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(vec, k=12, field=['测试_1', '测试_7'])

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_stability_of_mvdb_files():
    """Test whether the modification time of the mvdb file changes when querying multiple times."""
    database = get_database_for_query(with_field=True)
    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='test_1')
    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)
    mvdb_path = database._database_chunk_path
    mvdb_mtime = [os.path.getmtime(_) for _ in mvdb_path]
    time.sleep(1)

    for i in range(20):
        n, d = database.query(vec, k=6, field='test_1')
        assert len(n) == len(d) == 6
        assert list(d) == sorted(d, key=lambda s: -s)
        assert all(i in database._id_filter for i in n)
        assert all(10 <= i < 20 for i in n)
        assert mvdb_mtime == [os.path.getmtime(_) for _ in mvdb_path]
        time.sleep(1)

    database.delete()


def test_lazy_bulk_add_items():
    database = get_database()
    items = []
    for i in range(100):
        items.append((np.ones(100), i, "test_" + str(i // 10)))

    with database.insert_session():
        database.bulk_add_items(items, save_immediately=False)

    assert database.shape == (100, 100)
    assert int(database.head(100).sum()) == 100 * 100

    database.delete()


def test_lazy_add_item():
    database = get_database()
    with database.insert_session():
        database.add_item(np.ones(100), id=1, field="test", save_immediately=False)

    assert database.shape == (1, 100)
    assert int(database.head(1).sum()) == 100
    database.delete()


def test_lazy_add_item_and_lazy_bulk_add_items():
    database = get_database()
    items = []
    id = None
    for i in range(100):
        id = i
        items.append((np.ones(100), id, "test_" + str(i // 10)))

    with database.insert_session():
        database.add_item(np.ones(100), id=id+1, field="test", save_immediately=False)
        database.bulk_add_items(items, save_immediately=False)

    assert database.shape == (101, 100)
    assert int(database.head(101).sum()) == 101 * 100
    database.delete()


def test_query_stability_of_query_function():
    database = get_database_for_query(with_field=True)
    vec = np.random.random(100)
    last_n, last_d = database.query(vec, k=6, field='test_1')
    assert len(last_n) == len(last_d) == 6
    for i in range(20):
        n, d = database.query(vec, k=6, field='test_1')
        assert last_d.tolist() == d.tolist()
        assert last_n.tolist() == n.tolist()
        assert len(n) == len(d) == 6
        assert list(d) == sorted(d, key=lambda s: -s)
        assert all(i in database._id_filter for i in n)
        assert all(10 <= i < 20 for i in n)

    database.delete()


# Test whether calling bulk_add_items multiple times can insert data normally
def test_multiple_bulk_add_items():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), ))

    database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    database.commit()
    assert database.shape == (101, 100)

    database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    database.commit()
    assert database.shape == (202, 100)
    database.delete()


def test_multiple_bulk_add_items_with_insert_session():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), ))

    with database.insert_session():
        database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    assert database.shape == (101, 100)

    with database.insert_session():
        database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    assert database.shape == (202, 100)
    database.delete()


# Test if secondary initialization can properly initialize and query
def test_multiple_initialization(dim=100, database_path='test_min_vec.mvdb', chunk_size=1000, dtypes=np.float32):
    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)
    items = []
    for i in range(101):
        items.append((np.ones(100), i, "test_" + str(i // 10)))

    database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    database.commit()
    assert database.shape == (101, 100)
    del database

    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)

    # query
    vec = np.random.random(100)
    n, d = database.query(vec, k=10, field='test_1')
    assert len(n) == len(d) == 10
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)
    database.delete()


# Test if secondary initialization can properly initialize and insert
def test_multiple_initialization_with_insert_session(dim=100, database_path='test_min_vec.mvdb', chunk_size=1000, dtypes=np.float32):
    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)
    items = []
    for i in range(101):
        items.append((np.ones(100), i, "test_" + str(i // 10)))

    with database.insert_session():
        database.bulk_add_items(items, save_immediately=False)
    assert database.fields == []
    assert database.indices == []

    assert database.shape == (101, 100)
    del database

    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)

    items = []
    for i in range(101):
        items.append((np.ones(100), None, "test_" + str(i // 10)))
    # insert
    with database.insert_session():
        database.bulk_add_items(items, save_immediately=True)
    assert database.fields == []
    assert database.indices == []

    assert database.shape == (202, 100)
    database.delete()
