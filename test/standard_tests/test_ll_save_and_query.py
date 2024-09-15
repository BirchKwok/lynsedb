import os
import shutil
import time
import pytest

from test import Filter, FieldCondition, MatchField, MatchID
import numpy as np
import lynse

client = lynse.VectorDBClient()
database = client.create_database(database_name='test_local_db', drop_if_exists=True)


@pytest.fixture
def cleanup():
    # 在测试前执行清理
    db_path = os.path.expanduser('~/.LynseDB/databases/test_local_db')
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    time.sleep(0.1)  # 给系统一些时间来完全释放资源

    yield  # 这里会运行测试

    # 在测试后执行清理
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    time.sleep(0.1)


def get_collection(dim=100, chunk_size=1000, dtypes='float32', drop_if_exists=True):
    collection = database.require_collection('test_collection', dim=dim, chunk_size=chunk_size,
                                             dtypes=dtypes, drop_if_exists=drop_if_exists)
    return collection


def get_collection_for_query(*args, with_field=True, field_prefix='test_', **kwargs):
    collection = get_collection(*args, **kwargs)
    np.random.seed(2023)
    with collection.insert_session():
        items = []
        for i in range(100):
            items.append((np.random.random(100), i, {"test": field_prefix + str(i // 10) if with_field else None}))

        collection.bulk_add_items(items, enable_progress_bar=False)

    return collection


def test_add_single_item_without_id_and_field(cleanup):
    collection = get_collection(drop_if_exists=True)
    id = collection.add_item(np.ones(100), id=1)

    collection.commit()

    assert collection.shape == (1, 100)


def test_add_single_item_with_id_and_field(cleanup):
    collection = get_collection(drop_if_exists=True)
    id = collection.add_item(np.ones(100), id=1, field={"test": 1})

    collection.commit()

    assert collection.shape == (1, 100)


def test_bulk_add_item_without_id_and_field(cleanup):
    collection = get_collection(drop_if_exists=True)
    items = []
    for i in range(101):
        items.append((np.ones(100), i))

    collection.bulk_add_items(items)

    collection.commit()

    assert collection.shape == (101, 100)


def test_bulk_add_item_with_id_and_field(cleanup):
    collection = get_collection(drop_if_exists=True)
    items = []
    for i in range(101):
        items.append((np.ones(100), i, {"test": "test_" + str(i // 10)}))

    collection.bulk_add_items(items)

    collection.commit()

    assert collection.shape == (101, 100)


def test_add_bulk_item_with_id_and_chinese_field(cleanup):
    collection = get_collection(drop_if_exists=True)
    items = []
    for i in range(101):
        items.append((np.ones(100), i, {"测试": "测试_" + str(i // 10)}))

    collection.bulk_add_items(items)

    collection.commit()

    assert collection.shape == (101, 100)


def test_query_without_field(cleanup):
    collection = get_collection_for_query(with_field=False)

    vec = np.random.random(100)
    n, d, f = collection.search(vec, k=6)

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)


def test_query_with_field(cleanup):
    collection = get_collection_for_query(with_field=True, drop_if_exists=True)

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)


def test_query_with_list_field(cleanup):
    collection = get_collection_for_query(with_field=True, drop_if_exists=True)

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=12, search_filter=Filter(
            any=[
                FieldCondition(key="test", matcher=MatchField('test_1')),
                FieldCondition(key="test", matcher=MatchField('test_7'))
            ]
        )
    )

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)


def test_query_with_chinese_field(cleanup):
    collection = get_collection_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[
                FieldCondition(key="test", matcher=MatchField('测试_1'))
            ]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)


def test_query_with_chinese_list_field(cleanup):
    collection = get_collection_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=12, search_filter=Filter(
            any=[
                FieldCondition(key="test", matcher=MatchField('测试_1')),
                FieldCondition(key="test", matcher=MatchField('测试_7'))
            ]
        )
    )

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)


def test_query_with_subset_indices(cleanup):
    collection = get_collection_for_query(with_field=True, drop_if_exists=True)

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key=":match_id:", matcher=MatchID(list(range(10))))]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(i < 10 for i in n)


def test_query_with_subset_indices_and_field(cleanup):
    collection = get_collection_for_query(with_field=True, field_prefix='test_')

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[
                FieldCondition(key="test", matcher=MatchField('test_0')),
                FieldCondition(key=":match_id:", matcher=MatchID(list(range(10))))
            ]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(i < 10 for i in n)


def test_query_with_subset_indices_and_list_field(cleanup):
    collection = get_collection_for_query(with_field=True, drop_if_exists=True)

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=12, search_filter=Filter(
            must=[
                FieldCondition(key=":match_id:", matcher=MatchID(list(range(10, 20)) + list(range(70, 80))))
            ],
            any=[
                FieldCondition(key="test", matcher=MatchField('test_1')),
                FieldCondition(key="test", matcher=MatchField('test_7')),
            ]
        )
    )

    assert len(n) == len(d) == 12
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)


def test_query_with_subset_indices_and_chinese_field(cleanup):
    collection = get_collection_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[
                FieldCondition(key="test", matcher=MatchField('测试_0')),
                FieldCondition(key=":match_id:", matcher=MatchID(list(range(10))))
            ]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(i < 10 for i in n)


def test_query_with_subset_indices_and_chinese_list_field(cleanup):
    collection = get_collection_for_query(with_field=True, field_prefix='测试_')

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=12, search_filter=Filter(
            must=[
                FieldCondition(key=":match_id:", matcher=MatchID(list(range(10, 20)) + list(range(70, 80))))
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
    assert all(i in collection._id_filter for i in n)
    assert all((10 <= i < 20) or (70 <= i < 80) for i in n)


def test_query_stability_of_mvdb_files(cleanup):
    collection = get_collection_for_query(with_field=True, drop_if_exists=True)
    vec = np.random.random(100)
    last_n, last_d, last_f = collection.search(
        vec, k=6,
        search_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
        )
    )

    assert len(last_n) == len(last_d) == 6
    for i in range(20):
        n, d, f = collection.search(
            vec, k=6,
            search_filter=Filter(
                must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
            )
        )
        assert last_d.tolist() == d.tolist()
        assert last_n.tolist() == n.tolist()
        assert len(n) == len(d) == 6
        assert list(d) == sorted(d, key=lambda s: -s)
        assert all(i in collection._id_filter for i in n)
        assert all(10 <= i < 20 for i in n)


def test_multiple_bulk_add_items(cleanup):
    collection = get_collection(drop_if_exists=True)
    items = []
    for i in range(101):
        items.append((np.ones(100), i))

    collection.bulk_add_items(items)
    collection.commit()
    assert collection.shape == (101, 100)

    items = []
    for i in range(101):
        items.append((np.ones(100), i + 101))

    collection.bulk_add_items(items)
    collection.commit()

    assert collection.shape == (202, 100)


def test_multiple_bulk_add_items_with_insert_session(cleanup):
    collection = get_collection(drop_if_exists=True)
    items = []
    for i in range(101):
        items.append((np.ones(100), i))

    with collection.insert_session():
        collection.bulk_add_items(items)

    assert collection.shape == (101, 100)

    items = []
    for i in range(101, 202):
        items.append((np.ones(100), i))

    with collection.insert_session():
        collection.bulk_add_items(items)

    assert collection.shape == (202, 100)


def test_multiple_initialization(cleanup):
    dim = 100; chunk_size = 1000; dtypes = 'float32'
    collection = get_collection(drop_if_exists=True, dim=dim, chunk_size=chunk_size, dtypes=dtypes)
    items = []
    for i in range(101):
        items.append((np.ones(100), i, {"test": "test_" + str(i // 10)}))

    with collection.insert_session():
        collection.bulk_add_items(items)

    assert collection.shape == (101, 100)

    collection = get_collection(dim=dim, chunk_size=chunk_size, dtypes=dtypes, drop_if_exists=False)

    items = []
    for i in range(101):
        items.append((np.ones(100), i + 101, {"test": "test_" + str(i // 10)}))
    # insert
    with collection.insert_session():
        collection.bulk_add_items(items)

    assert collection.shape == (202, 100)


def test_result_order(cleanup):
    def get_test_vectors(shape):
        for i in range(shape[0]):
            yield np.random.random(shape[1])

    collection = get_collection(drop_if_exists=True, dim=10, chunk_size=10000)
    with collection.insert_session():
        # Define the initial ID.
        id = 0
        vectors = []
        for t in get_test_vectors((100000, 10)):
            if id == 0:
                query = t
            vectors.append((t, id))
            id += 1

        # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
        collection.bulk_add_items(vectors)

    for index_mode in ['IVF-IP-SQ8', 'IVF-IP', 'IVF-L2sq-SQ8', 'IVF-L2sq',
                       'IVF-Cos-SQ8', 'IVF-Cos', 'IVF-Jaccard-Binary', 'IVF-Hamming-Binary',
                       'Flat-IP-SQ8', 'Flat-IP', 'Flat-L2sq-SQ8', 'Flat-L2sq', 'Flat-Cos-SQ8', 'Flat-Cos',
                       'Flat-Jaccard-Binary', 'Flat-Hamming-Binary']:
        collection.build_index(index_mode=index_mode)
        index, score, field = collection.search(query, k=10)

        assert len(index) == len(score) == 10


def test_transactions(cleanup):
    collection = get_collection(dim=1024, chunk_size=10000, drop_if_exists=True)

    def get_test_vectors(shape):
        for i in range(shape[0]):
            yield np.random.random(shape[1])

    with collection.insert_session():
        id = 0
        vectors = []
        for t in get_test_vectors((100000, 1024)):
            vectors.append((t, id))
            id += 1

        collection.bulk_add_items(vectors, batch_size=1000)

    collection.build_index(index_mode='IVF-IP')
    assert collection.shape == (100000, 1024)

    with pytest.raises(ValueError):
        with collection.insert_session():
            id = 0
            vectors = []
            for t in get_test_vectors((100000, 1024)):
                vectors.append((t, id))
                id += 1
            collection.bulk_add_items(vectors, batch_size=1000)

    collection.build_index(index_mode='IVF-IP')
    assert collection.shape == (100000, 1024)

    with pytest.raises(ValueError):
        collection.bulk_add_items(vectors, batch_size=1000)

    collection.build_index(index_mode='IVF-IP')
    assert collection.shape == (100000, 1024)


def test_filter(cleanup):
    collection = get_collection_for_query(with_field=True, drop_if_exists=True)

    vec = np.random.random(100)
    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))]
        )
    )

    assert len(n) == len(d) == 6
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(10 <= i < 20 for i in n)

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))],
            any=[FieldCondition(key="test", matcher=MatchField('test_7'))]
        )
    )

    assert len(n) == len(d) == 0

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key="test", matcher=MatchField('test_1'))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[FieldCondition(key="test", matcher=MatchField('test_0'))]
        )
    )

    assert len(n) == len(d) == 0

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
        )
    )

    assert len(n) == len(d) == 3
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(1 <= i < 4 for i in n)

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[FieldCondition(key="test", matcher=MatchField('test_0'))]
        )
    )

    assert len(n) == len(d) == 0

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3]))]
        )
    )

    assert len(n) == len(d) == 0

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3]))],
            any=[FieldCondition(key="test", matcher=MatchField('test_0'))],
            must_not=[FieldCondition(key=":match_id:", matcher=MatchID([4, 5, 6]))]
        )
    )

    assert len(n) == len(d) == 3
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(1 <= i < 4 for i in n)

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[
                FieldCondition(key='test', matcher=MatchField('test_0')),
            ],
            any=[
                FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3, 4, 5])),
            ]
        ))

    assert len(n) == len(d) == 5
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(0 <= i < 6 for i in n)

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[
                FieldCondition(key='test', matcher=MatchField(['test_0', 'test_00'], all_comparators=True)),
            ],
            any=[
                FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3, 4, 5])),
            ]
        ))

    assert len(n) == len(d) == 0

    n, d, f = collection.search(
        vec, k=6, search_filter=Filter(
            must=[
                FieldCondition(key='test', matcher=MatchField(['test_0', 'test_00'], all_comparators=False)),
            ],
            any=[
                FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3, 4, 5])),
            ]
        ))

    assert len(n) == len(d) == 5
    assert list(d) == sorted(d, key=lambda s: -s)
    assert all(i in collection._id_filter for i in n)
    assert all(0 <= i < 6 for i in n)
