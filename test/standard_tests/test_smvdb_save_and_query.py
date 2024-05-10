import shutil
from pathlib import Path

import pytest

from test import StandaloneMinVectorDB, Filter, FieldCondition, MatchField, IDCondition, MatchID
import numpy as np


def get_database(dim=100, database_path='test_min_vec', chunk_size=1000, dtypes='float32'):
    if Path(database_path).exists():
        shutil.rmtree(database_path)

    database = StandaloneMinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)
    return database


def get_database_for_query(*args, with_field=True, field_prefix='test_', **kwargs):
    database = get_database(*args, **kwargs)
    np.random.seed(2023)
    with database.insert_session():
        items = []
        for i in range(100):
            items.append((np.random.random(100), i, {"test": field_prefix + str(i // 10) if with_field else None}))

        database.bulk_add_items(items)

    return database


def test_add_single_item_without_id_and_field():
    database = get_database()
    id = database.add_item(np.ones(100), id=1)

    database.commit()

    assert database._matrix_serializer.database == []
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (1, 100)

    database.delete()


def test_add_single_item_with_id_and_field():
    database = get_database()
    id = database.add_item(np.ones(100), id=1, field={"test": 1})

    database.commit()

    assert database._matrix_serializer.database == []
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (1, 100)

    database.delete()


def test_bulk_add_item_without_id_and_field():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100),i))

    database.bulk_add_items(items)

    database.commit()
    assert database._matrix_serializer.database == []
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (101, 100)

    database.delete()


def test_bulk_add_item_with_id_and_field():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), i, {"test": "test_" + str(i // 10)}))

    database.bulk_add_items(items)

    database.commit()

    assert database._matrix_serializer.database == []
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (101, 100)

    database.delete()


def test_add_bulk_item_with_id_and_chinese_field():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), i, {"测试": "测试_" + str(i // 10)}))

    database.bulk_add_items(items)

    database.commit()

    assert database._matrix_serializer.database == []
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (101, 100)

    database.delete()


def test_query_without_field():
    database = get_database_for_query(with_field=False)

    print(database.shape)
    vec = np.random.random(100)
    n, d = database.query(vec, k=6)

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)

    database.delete()


def test_query_with_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_list_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=12, query_filter=Filter(
            any=[
                FieldCondition(key="test", matcher=MatchField('test_1')),
                FieldCondition(key="test", matcher=MatchField('test_7'))
            ]
        )
    )

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_with_chinese_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[
                FieldCondition(key="test", matcher=MatchField('测试_1'))
            ]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    database.delete()


def test_query_with_chinese_list_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=12, query_filter=Filter(
            any=[
                FieldCondition(key="test", matcher=MatchField('测试_1')),
                FieldCondition(key="test", matcher=MatchField('测试_7'))
            ]
        )
    )

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_with_subset_indices():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[IDCondition(MatchID(list(range(10))))]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(i < 10 for i in n)

    database.delete()


def test_query_with_subset_indices_and_field():
    database = get_database_for_query(with_field=True, field_prefix='test_')

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[
                FieldCondition(key="test", matcher=MatchField('test_0')),
                IDCondition(MatchID(list(range(10))))
            ]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(i < 10 for i in n)

    database.delete()


def test_query_with_subset_indices_and_list_field():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=12, query_filter=Filter(
            must=[
                IDCondition(MatchID(list(range(10, 20)) + list(range(70, 80))))
            ],
            any=[
                FieldCondition(key="test", matcher=MatchField('test_1')),
                FieldCondition(key="test", matcher=MatchField('test_7')),
            ]
        )
    )

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_with_subset_indices_and_chinese_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[
                FieldCondition(key="test", matcher=MatchField('测试_0')),
                IDCondition(MatchID(list(range(10))))
            ]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(i < 10 for i in n)

    database.delete()


def test_query_with_subset_indices_and_chinese_list_field():
    database = get_database_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=12, query_filter=Filter(
            must=[
                IDCondition(MatchID(list(range(10, 20)) + list(range(70, 80))))
            ],
            any=[
                FieldCondition(key="test", matcher=MatchField('测试_1')),
                FieldCondition(key="test", matcher=MatchField('测试_7')),
            ]
        )
    )

    print(n)
    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)

    database.delete()


def test_query_stability_of_mvdb_files():
    database = get_database_for_query(with_field=True)
    vec = np.random.random(100)
    last_n, last_d = database.query(
        vec, k=6,
        query_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
        )
    )

    assert len(last_n) == len(last_d) == 6
    for i in range(20):
        n, d = database.query(
            vec, k=6,
            query_filter=Filter(
                must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
            )
        )
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
        items.append((np.ones(100), i))

    database.bulk_add_items(items)
    assert len(database._matrix_serializer.fields) == 101
    assert database._matrix_serializer.indices == list(range(101))

    database.commit()
    assert database.shape == (101, 100)

    items = []
    for i in range(101):
        items.append((np.ones(100), i + 101))

    database.bulk_add_items(items)
    database.commit()
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (202, 100)

    database.delete()


def test_multiple_bulk_add_items_with_insert_session():
    database = get_database()
    items = []
    for i in range(101):
        items.append((np.ones(100), i))

    with database.insert_session():
        database.bulk_add_items(items)
        assert len(database._matrix_serializer.fields) == 101
        assert database._matrix_serializer.indices == list(range(101))

    assert database.shape == (101, 100)

    items = []
    for i in range(101, 202):
        items.append((np.ones(100), i))

    with database.insert_session():
        database.bulk_add_items(items)

    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (202, 100)

    database.delete()


# Test if secondary initialization can properly initialize and query
def test_multiple_initialization(dim=100, database_path='test_min_vec', chunk_size=1000, dtypes='float32'):
    database = StandaloneMinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)
    items = []
    for i in range(101):
        items.append((np.ones(100), i, {"test": "test_" + str(i // 10)}))

    with database.insert_session():
        database.bulk_add_items(items)
    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (101, 100)
    del database

    database = StandaloneMinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes)

    items = []
    for i in range(101):
        items.append((np.ones(100), i + 101, {"test": "test_" + str(i // 10)}))
    # insert
    with database.insert_session():
        database.bulk_add_items(items)

    assert database._matrix_serializer.fields == []
    assert database._matrix_serializer.indices == []

    assert database.shape == (202, 100)
    database.delete()


def test_result_order():
    def get_test_vectors(shape):
        for i in range(shape[0]):
            yield np.random.random(shape[1])

    for index_mode in ['FLAT', 'IVF-FLAT']:
        db = StandaloneMinVectorDB(dim=1024, database_path='test_min_vec',
                                   chunk_size=10000, index_mode=index_mode)

        # You can perform this operation multiple times, and the data will be appended to the database.
        with db.insert_session():
            # Define the initial ID.
            id = 0
            vectors = []
            for t in get_test_vectors((100000, 1024)):
                if id == 0:
                    query = t
                vectors.append((t, id))
                id += 1

            # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
            db.bulk_add_items(vectors)

        index, score = db.query(query, k=10)

        assert len(index) == len(score) == 10
        assert index[0] == 0

        db.delete()


def test_refit(capsys):
    db = StandaloneMinVectorDB(dim=1024, database_path='test_min_vec', chunk_size=10000, index_mode='IVF-FLAT')
    with db.insert_session():
        for i in range(100000):
            db.add_item(np.random.rand(1024), id=i)

    out, err = capsys.readouterr()

    assert 'Start to fit the index for cluster' in err
    assert db._matrix_serializer.ann_model.fitted

    with db.insert_session():
        for i in range(100000, 110000):
            db.add_item(np.random.rand(1024), id=i)

    out, err = capsys.readouterr()
    assert 'Incrementally building the index for cluster' in err
    assert db._matrix_serializer.ann_model.fitted

    with db.insert_session():
        for i in range(110000, 200000):
            db.add_item(np.random.rand(1024), id=i)

    out, err = capsys.readouterr()
    assert 'Refitting the index for cluster' in err
    assert db._matrix_serializer.ann_model.fitted

    db.delete()


def test_transactions():
    db = StandaloneMinVectorDB(dim=1024, database_path='test_min_vec', chunk_size=10000, index_mode='IVF-FLAT')

    def get_test_vectors(shape):
        for i in range(shape[0]):
            yield np.random.random(shape[1])

    with db.insert_session():
        id = 0
        vectors = []
        for t in get_test_vectors((100000, 1024)):
            vectors.append((t, id))
            id += 1

        db.bulk_add_items(vectors, batch_size=1000)

    assert db.shape == (100000, 1024)

    with pytest.raises(ValueError):
        with db.insert_session():
            id = 0
            vectors = []
            for t in get_test_vectors((100000, 1024)):
                vectors.append((t, id))
                id += 1
            db.bulk_add_items(vectors, batch_size=1000)

    assert db.shape == (100000, 1024)

    with pytest.raises(ValueError):
        db.bulk_add_items(vectors, batch_size=1000)

    assert db.shape == (100000, 1024)


def test_filter():
    database = get_database_for_query(with_field=True)

    vec = np.random.random(100)
    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))],
            any=[FieldCondition(key="test", matcher=MatchField('test_7'))]
        )
    )

    assert len(n) == len(d) == 0

    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[FieldCondition(key="test", matcher=MatchField('test_0'))]
        )
    )

    assert len(n) == len(d) == 0

    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[IDCondition(MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
        )
    )

    assert len(n) == len(d) == 3
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(1 <= i < 4 for i in n)

    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[IDCondition(MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[FieldCondition(key="test", matcher=MatchField('test_0'))]
        )
    )

    assert len(n) == len(d) == 0

    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[IDCondition(MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[IDCondition(MatchID([1, 2, 3]))]
        )
    )

    assert len(n) == len(d) == 0

    n, d = database.query(
        vec, k=6, query_filter=Filter(
            must=[IDCondition(MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[IDCondition(MatchID([4, 5, 6]))]
        )
    )

    assert len(n) == len(d) == 3
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(1 <= i < 4 for i in n)

    n, d = database.query(
        vec, k=6, query_filter=Filter(
        must=[
            FieldCondition(key='test', matcher=MatchField('test_0')),
        ],
        any=[
            IDCondition(MatchID([1, 2, 3, 4, 5])),
        ]
    ))

    assert len(n) == len(d) == 5
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(0 <= i < 6 for i in n)

    n, d = database.query(
        vec, k=6, query_filter=Filter(
        must=[
            FieldCondition(key='test', matcher=MatchField(['test_0', 'test_00'], all_comparators=True)),
        ],
        any=[
            IDCondition(MatchID([1, 2, 3, 4, 5])),
        ]
    ))

    assert len(n) == len(d) == 0

    n, d = database.query(
        vec, k=6, query_filter=Filter(
        must=[
            FieldCondition(key='test', matcher=MatchField(['test_0', 'test_00'], all_comparators=False)),
        ],
        any=[
            IDCondition(MatchID([1, 2, 3, 4, 5])),
        ]
    ))

    assert len(n) == len(d) == 5
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in database._id_filter for i in n)
    assert all(0 <= i < 6 for i in n)
