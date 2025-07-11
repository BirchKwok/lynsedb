"""
测试NumPack适配器功能
"""

import os
import numpy as np
import pytest
from tempfile import NamedTemporaryFile
from pathlib import Path

from lynse.core_components.numpack_adapter import (
    save_nnp, load_nnp, replace_arrays, drop_arrays,
    get_array_info, migrate_nnp_to_numpack,
    NnpFileSavingError, NnpValueError
)


def test_save_and_load_single_array():
    """测试保存和加载单个数组"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建测试数组
        test_array = np.random.rand(100, 50).astype(np.float32)

        # 保存数组
        save_nnp(temp_path, test_array=test_array)

        # 加载并验证数组
        loaded_arrays = load_nnp(temp_path)
        assert 'test_array' in loaded_arrays
        assert np.allclose(test_array, loaded_arrays['test_array'])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        # 清理可能的元数据文件
        metadata_file = Path(temp_path).with_suffix('.meta')
        if metadata_file.exists():
            metadata_file.unlink()


def test_save_and_load_multiple_arrays():
    """测试保存和加载多个数组"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建测试数组
        arrays = {
            'array1': np.random.rand(100, 50).astype(np.float32),
            'array2': np.random.rand(200, 30).astype(np.float32),
            'array3': np.random.rand(150).astype(np.float32)  # 1D数组
        }

        # 保存数组
        save_nnp(temp_path, **arrays)

        # 加载并验证数组
        loaded_arrays = load_nnp(temp_path)
        assert set(loaded_arrays.keys()) == set(arrays.keys())
        for name, array in arrays.items():
            assert np.allclose(array, loaded_arrays[name])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_selective_loading():
    """测试选择性加载数组"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建测试数组
        arrays = {
            'array1': np.random.rand(100, 50).astype(np.float32),
            'array2': np.random.rand(200, 30).astype(np.float32),
            'array3': np.random.rand(150, 40).astype(np.float32)
        }

        # 保存数组
        save_nnp(temp_path, **arrays)

        # 选择性加载
        selected = load_nnp(temp_path, array_names=['array1', 'array3'])
        assert len(selected) == 2
        assert 'array2' not in selected
        assert np.allclose(arrays['array1'], selected['array1'])
        assert np.allclose(arrays['array3'], selected['array3'])

        # 加载单个数组
        single = load_nnp(temp_path, array_names='array2')
        assert len(single) == 1
        assert np.allclose(arrays['array2'], single['array2'])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_append_arrays():
    """测试追加数组功能（通过手动合并实现）"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建初始数组
        initial_arrays = {
            'array1': np.random.rand(100, 50).astype(np.float32),
            'array2': np.random.rand(50, 30).astype(np.float32)
        }

        # 保存初始数组
        save_nnp(temp_path, **initial_arrays)

        # 创建追加数据
        append_data = {
            'array1': np.random.rand(50, 50).astype(np.float32),
            'array2': np.random.rand(25, 30).astype(np.float32)
        }

        # 手动实现追加功能：加载现有数据，合并，然后保存
        existing_data = load_nnp(temp_path)
        combined_data = {}
        for name in initial_arrays.keys():
            if name in append_data:
                combined_data[name] = np.vstack([existing_data[name], append_data[name]])
            else:
                combined_data[name] = existing_data[name]

        # 保存合并后的数据
        save_nnp(temp_path, **combined_data)

        # 验证结果
        loaded = load_nnp(temp_path)
        assert loaded['array1'].shape == (150, 50)  # 100 + 50
        assert loaded['array2'].shape == (75, 30)   # 50 + 25

        # 验证数据完整性
        assert np.allclose(initial_arrays['array1'], loaded['array1'][:100])
        assert np.allclose(append_data['array1'], loaded['array1'][100:])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_replace_arrays_functionality():
    """测试替换数组功能"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建初始数组
        initial_array = np.random.rand(100, 50).astype(np.float32)
        save_nnp(temp_path, {'test_array': initial_array})

        # 创建替换数据
        replace_data = np.random.rand(3, 50).astype(np.float32)
        indexes = np.array([0, 10, 50])

        # 执行替换
        replace_arrays(temp_path, {'test_array': replace_data}, indexes)

        # 验证替换结果
        loaded = load_nnp(temp_path)
        result_array = loaded['test_array']

        # 检查替换的位置
        assert np.allclose(result_array[indexes], replace_data)

        # 检查未替换的位置保持不变
        unchanged_mask = np.ones(100, dtype=bool)
        unchanged_mask[indexes] = False
        assert np.allclose(result_array[unchanged_mask], initial_array[unchanged_mask])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_drop_arrays_functionality():
    """测试删除数组功能"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建初始数组
        arrays = {
            'array1': np.random.rand(100, 50).astype(np.float32),
            'array2': np.random.rand(100, 30).astype(np.float32)
        }
        save_nnp(temp_path, **arrays)

        # 删除指定索引的数据
        indexes_to_drop = np.array([0, 10, 50, 99])
        drop_arrays(temp_path, indexes_to_drop, array_names=['array1'])

        # 验证结果
        loaded = load_nnp(temp_path)
        assert loaded['array1'].shape == (96, 50)  # 100 - 4
        assert loaded['array2'].shape == (100, 30)  # 未变化

        # 验证删除的数据不在结果中
        expected_array1 = np.delete(arrays['array1'], indexes_to_drop, axis=0)
        assert np.allclose(loaded['array1'], expected_array1)
        assert np.allclose(loaded['array2'], arrays['array2'])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_get_array_info():
    """测试获取数组信息功能"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建测试数组
        arrays = {
            'float_array': np.random.rand(100, 50).astype(np.float32),
            'int_array': np.random.randint(0, 100, (200, 30)).astype(np.int32)
        }
        save_nnp(temp_path, **arrays)

        # 获取数组信息
        info = get_array_info(temp_path)

        # 验证信息
        assert 'float_array' in info
        assert 'int_array' in info
        assert info['float_array']['shape'] == (100, 50)
        assert info['int_array']['shape'] == (200, 30)
        assert 'float32' in info['float_array']['dtype']
        assert 'int32' in info['int_array']['dtype']

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_mmap_mode():
    """测试内存映射模式"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建测试数组
        test_array = np.random.rand(1000, 100).astype(np.float32)
        save_nnp(temp_path, {'test_array': test_array})

        # 使用内存映射模式加载
        loaded_mmap = load_nnp(temp_path, mmap_mode=True)
        loaded_normal = load_nnp(temp_path, mmap_mode=False)

        # 验证数据一致性
        assert np.allclose(loaded_mmap['test_array'], loaded_normal['test_array'])
        assert np.allclose(loaded_mmap['test_array'], test_array)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_error_handling():
    """测试错误处理"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 测试无效的数组名称
        with pytest.raises(NnpFileSavingError):
            save_nnp(temp_path, **{'': np.random.rand(10, 10)})

        # 测试无效的数组维度
        with pytest.raises(NnpFileSavingError):
            save_nnp(temp_path, {'test': np.random.rand(10, 10, 10)})

        # 测试超过最大行数限制
        with pytest.raises(NnpFileSavingError):
            save_nnp(temp_path, {'test': np.random.rand(10000001, 10)})

        # 测试加载不存在的文件
        with pytest.raises(FileNotFoundError):
            load_nnp('nonexistent_file.npk')

        # 测试加载不存在的数组
        save_nnp(temp_path, {'existing': np.random.rand(10, 10)})
        with pytest.raises(KeyError):
            load_nnp(temp_path, array_names=['nonexistent'])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_data_types_compatibility():
    """测试不同数据类型的兼容性"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 测试不同的numpy数据类型
        arrays = {
            'float32': np.random.rand(50, 10).astype(np.float32),
            'float64': np.random.rand(50, 10).astype(np.float64),
            'int32': np.random.randint(0, 100, (50, 10)).astype(np.int32),
            'int64': np.random.randint(0, 100, (50, 10)).astype(np.int64),
        }

        # 保存和加载
        save_nnp(temp_path, **arrays)
        loaded = load_nnp(temp_path)

        # 验证所有数据类型
        for name, original in arrays.items():
            assert np.allclose(original, loaded[name])
            assert original.dtype == loaded[name].dtype

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_large_arrays():
    """测试大数组处理"""
    with NamedTemporaryFile(delete=False, suffix='.npk') as temp_file:
        temp_path = temp_file.name

    try:
        # 创建较大的测试数组（约40MB）
        large_array = np.random.rand(5000, 1000).astype(np.float32)

        # 保存和加载
        save_nnp(temp_path, {'large_array': large_array})
        loaded = load_nnp(temp_path)

        # 验证数据完整性
        assert np.allclose(large_array, loaded['large_array'])

        # 测试内存映射模式的性能
        loaded_mmap = load_nnp(temp_path, mmap_mode=True)
        assert np.allclose(large_array, loaded_mmap['large_array'])

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__])
