import pytest
import numpy as np
import lynse
from lynse.core_components.fields_cache import Filter, FieldCondition, MatchField, MatchID

client = lynse.VectorDBClient()
database = client.create_database(database_name='test_ivf_db', drop_if_exists=True)


def get_collection(dim=100, chunk_size=1000, dtypes='float32', drop_if_exists=True):
    collection = database.require_collection('test_collection', dim=dim, chunk_size=chunk_size,
                                          dtypes=dtypes, drop_if_exists=drop_if_exists)
    return collection


def get_collection_with_data(dim=100, size=200000, chunk_size=10000, dtypes='float32'):
    """创建包含测试数据的集合"""
    collection = get_collection(dim=dim, chunk_size=chunk_size, dtypes=dtypes)

    # 生成测试数据
    np.random.seed(42)
    with collection.insert_session():
        vectors = []
        for i in range(size):
            vec = np.random.random(dim)
            vectors.append((vec, i, {"group": f"group_{i//1000}"}))

        collection.bulk_add_items(vectors)

    return collection


def test_ivf_binary_index():
    """测试二进制量化的IVF索引"""
    collection = get_collection_with_data(dim=128)

    # 测试Jaccard距离的IVF索引
    collection.build_index(index_mode='IVF-Jaccard-Binary')
    query = np.random.random(128)

    # 基本搜索
    ids, scores, fields = collection.search(query, k=10)
    assert len(ids) == len(scores) == 10
    assert list(scores) == sorted(scores, reverse=True)

    # 带过滤器的搜索
    ids, scores, fields = collection.search(
        query, k=10,
        where=Filter(
            must=[FieldCondition(key=":group:", matcher=MatchField('group_1'))]
        )
    )
    assert len(ids) == 10
    assert all(1000 <= id < 2000 for id in ids)

    # 测试Hamming距离的IVF索引
    collection.build_index(index_mode='IVF-Hamming-Binary')
    ids, scores, fields = collection.search(query, k=10)
    assert len(ids) == len(scores) == 10
    assert list(scores) == sorted(scores, reverse=True)


def test_ivf_sq_index():
    """测试标量量化的IVF索引"""
    collection = get_collection_with_data(dim=128)

    # 测试IP距离的IVF索引
    collection.build_index(index_mode='IVF-IP-SQ8')
    query = np.random.random(128)

    # 基本搜索
    ids, scores, fields = collection.search(query, k=10)
    assert len(ids) == len(scores) == 10
    assert list(scores) == sorted(scores, reverse=True)

    # 带过滤器的搜索
    ids, scores, fields = collection.search(
        query, k=10,
        where=Filter(
            must=[FieldCondition(key=":group:", matcher=MatchField('group_1'))]
        )
    )
    assert len(ids) == 10
    assert all(1000 <= id < 2000 for id in ids)

    # 测试L2距离的IVF索引
    collection.build_index(index_mode='IVF-L2sq-SQ8')
    ids, scores, fields = collection.search(query, k=10)
    assert len(ids) == len(scores) == 10
    assert list(scores) == sorted(scores)  # L2距离越小越好


def test_ivf_nprobe():
    """测试IVF索引的nprobe参数"""
    collection = get_collection_with_data(dim=128)
    query = np.random.random(128)

    # 使用不同的nprobe值测试
    collection.build_index(index_mode='IVF-IP-SQ8')

    # nprobe较小时的结果
    ids1, scores1, _ = collection.search(query, k=10, nprobe=1)

    # nprobe较大时的结果
    ids2, scores2, _ = collection.search(query, k=10, nprobe=10)

    # nprobe较大时应该能找到更好的结果
    assert np.mean(scores2) >= np.mean(scores1)


def test_ivf_with_updates():
    """测试IVF索引在数据更新后的表现"""
    collection = get_collection_with_data(dim=128, size=50000)
    collection.build_index(index_mode='IVF-IP-SQ8')

    # 记录初始搜索结果
    query = np.random.random(128)
    ids1, scores1, _ = collection.search(query, k=10)

    # 添加新数据
    with collection.insert_session():
        vectors = []
        for i in range(50000, 100000):
            vec = np.random.random(128)
            vectors.append((vec, i, {"group": f"group_{i//1000}"}))
        collection.bulk_add_items(vectors)

    # 重建索引
    collection.build_index(index_mode='IVF-IP-SQ8')

    # 验证搜索结果
    ids2, scores2, _ = collection.search(query, k=10)
    assert len(ids2) == 10
    # 新的结果应该不会比旧的差
    assert np.mean(scores2) >= np.mean(scores1)


def test_ivf_edge_cases():
    """测试IVF索引的边界情况"""
    # 小数据集
    collection = get_collection_with_data(dim=128, size=1000)
    collection.build_index(index_mode='IVF-IP-SQ8')

    query = np.random.random(128)
    ids, scores, _ = collection.search(query, k=10)
    assert len(ids) == 10

    # 大k值
    ids, scores, _ = collection.search(query, k=900)
    assert len(ids) == 900

    # 空过滤器结果
    ids, scores, _ = collection.search(
        query, k=10,
        where=Filter(
            must=[FieldCondition(key=":group:", matcher=MatchField('non_existent_group'))]
        )
    )
    assert len(ids) == 0
