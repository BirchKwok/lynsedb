import shutil
from pathlib import Path

from test import MinVectorDB
import numpy as np


def get_database(dim=100, database_path='test_min_vec.mvdb', chunk_size=1000, dtypes=np.float32):
    if Path('.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])).exists():
        shutil.rmtree(Path('.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])))

    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)
    if database.database_chunk_path:
        database.delete()
    database.database_chunk_path = []
    return database


def get_database_for_query(*args, with_field=True, field_prefix='test_', **kwargs):
    database = get_database(*args, **kwargs)

    np.random.seed(2023)
    items = []
    for i in range(100):
        items.append((np.random.random(100), i, field_prefix + str(i // 10) if with_field else None))

    database.bulk_add_items(items)

    database.commit()

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
    assert database.database_chunk_path == []


def test_add_single_item_without_id_and_field():
    database = get_database()
    id = database.add_item(np.ones(100))

    assert database.database.sum() == 100
    assert database.fields == [None]
    assert database.indices == [id]

    database.commit()

    assert database.shape == (1, 100)
    database.delete()


def test_bulk_add_item_without_id_and_field():
    database = get_database()
    items = []
    for i in range(100):
        items.append((np.ones(100),))

    indices = database.bulk_add_items(items)
    assert database.fields == [None for i in range(100)]
    assert database.indices == indices

    database.commit()

    assert database.shape == (100, 100)

    database.delete()


def test_add_single_item_with_id_and_field():
    database = get_database()
    id = database.add_item(np.ones(100), id=1, field="test")

    assert database.database.sum() == 100
    assert database.fields == ["test"]
    assert database.indices == [id]
    assert id == 1

    database.commit()

    assert database.shape == (1, 100)

    database.delete()


def test_add_single_item_with_vector_normalize():
    database = get_database()
    id = database.add_item(np.random.random(100), id=1, field="test", normalize=True)

    assert database.fields == ["test"]
    assert database.indices == [id]
    assert id == 1

    database.commit()
    assert database.shape == (1, 100)

    database.delete()


def test_add_bulk_item_with_id_and_field():
    database = get_database()
    items = []
    for i in range(100):
        items.append((np.ones(100), i, "test_"+str(i // 10)))

    indices = database.bulk_add_items(items)

    assert database.fields == ["test_" + str(i // 10) for i in range(100)]
    assert database.indices == indices

    database.commit()

    assert database.shape == (100, 100)

    database.delete()


def test_add_bulk_item_with_normalize():
    database = get_database()
    items = []
    for i in range(100):
        items.append((np.ones(100), i, "test_"+str(i // 10)))

    indices = database.bulk_add_items(items, normalize=True)

    assert database.fields == ["test_" + str(i // 10) for i in range(100)]
    assert database.indices == indices

    database.commit()

    assert database.shape == (100, 100)
    database.delete()


def test_add_bulk_item_with_id_and_chinese_field():
    database = get_database()
    items = []
    for i in range(100):
        items.append((np.ones(100), i, "测试_" + str(i // 10)))

    indices = database.bulk_add_items(items)

    assert database.fields == ["测试_" + str(i // 10) for i in range(100)]
    assert database.indices == indices

    database.commit()

    assert database.shape == (100, 100)

    database.delete()


def test_query_without_field():
    database = get_database_for_query(with_field=False)
    vec = np.random.random(100)
    n, d = database.query(vec, k=12, field=None)

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database.all_indices for i in n)

    database.delete()


def test_query_with_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='test_1')

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database.all_indices for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_normalize():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='test_1', normalize=True)

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database.all_indices for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_list_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(vec, k=12, field=['test_1', 'test_7'])

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database.all_indices for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_with_chinese_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(vec, k=6, field='测试_1')

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database.all_indices for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_chinese_list_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(vec, k=12, field=['测试_1', '测试_7'])

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database.all_indices for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()
