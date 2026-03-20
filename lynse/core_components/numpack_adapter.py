"""
NumPack适配器模块
提供与现有nnp API兼容的接口，底层使用NumPack实现
"""

import numpy as np
import os
import glob
from pathlib import Path
from typing import Union, Dict, List, Optional, Any
import filelock

# 为了兼容性，保留原有的异常类
class NnpFileSavingError(Exception):
    """NumPack文件保存错误"""
    pass

class NnpValueError(ValueError):
    """NumPack数值错误"""
    pass


def save_nnp(filename: Union[str, Path], **arrays) -> None:
    """
    保存一个或多个命名数组到NumPack文件。

    Parameters:
        filename (str or PathLike): 文件路径，扩展名会自动更改为.npk
        **arrays: 命名数组，格式为 key=value

    Returns:
        None
    """
    try:
        import numpack
    except ImportError:
        raise ImportError("numpack库未安装。请运行: pip install numpack")

    # 确保文件扩展名为.npk
    filename = Path(filename).with_suffix('.npk')

    # 文件锁保护
    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        # 验证数组名称和数据
        for name, array in arrays.items():
            if not isinstance(name, str) or not name:
                raise NnpFileSavingError(f"Array name must be non-empty string, got {type(name)}")
            if not isinstance(array, np.ndarray):
                raise NnpFileSavingError(f"Array '{name}' must be numpy array, got {type(array)}")
            if array.ndim not in [1, 2]:
                raise NnpFileSavingError(f"Array '{name}' must be 1D or 2D, got {array.ndim}D")
            if array.shape[0] > 10000000:
                raise NnpFileSavingError(f"Array '{name}' exceeds maximum rows limit")

            # 转换为2D数组
            if array.ndim == 1:
                arrays[name] = array.reshape(-1, 1)

        try:
            # 创建新文件或覆盖现有文件
            _save_numpack_arrays(filename, arrays)

        except Exception as e:
            raise NnpFileSavingError(f"Error saving arrays: {str(e)}")


def load_nnp(filename: Union[str, Path], array_names: Optional[Union[str, List[str]]] = None,
             mmap_mode: bool = False, parallel: bool = True,
             cache_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    从NumPack文件加载一个或多个命名数组。

    Parameters:
        filename (str or PathLike): 文件路径，扩展名会自动更改为.npk
        array_names (str or list, optional): 要加载的数组名称。如果为None，加载所有数组
        mmap_mode (bool): 是否使用内存映射模式加载
        parallel (bool): 是否使用并行加载（为了兼容性保留）
        cache_size (int, optional): 缓存大小（为了兼容性保留）

    Returns:
        dict: 数组名称到numpy数组的映射
    """
    try:
        import numpack
    except ImportError:
        raise ImportError("numpack库未安装。请运行: pip install numpack")

    # 确保文件扩展名为.npk
    filename = Path(filename).with_suffix('.npk')

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    # 处理数组名称参数
    if array_names is not None:
        if isinstance(array_names, str):
            array_names = [array_names]
        # 验证数组名称
        for name in array_names:
            if not isinstance(name, str) or not name:
                raise NnpFileSavingError("Array name must be non-empty string")

    try:
        # 加载所有数组
        all_arrays = _load_numpack_arrays(filename, mmap_mode=mmap_mode)

        # 确定要返回的数组
        if array_names is None:
            arrays_to_return = all_arrays
        else:
            # 检查请求的数组是否存在
            missing = set(array_names) - set(all_arrays.keys())
            if missing:
                raise KeyError(f"Arrays not found in file: {missing}")

            # 过滤数组
            arrays_to_return = {name: all_arrays[name] for name in array_names}

        # 处理单列数组，转换为1D
        result = {}
        for name, array in arrays_to_return.items():
            if array.ndim == 2 and array.shape[1] == 1:
                array = array.reshape(-1)
            result[name] = array

        return result

    except Exception as e:
        raise NnpFileSavingError(f"Error loading arrays: {str(e)}")


def replace_arrays(filename: Union[str, Path], arrays: Union[Dict[str, np.ndarray], np.ndarray],
                   indexes: np.ndarray, array_name: Optional[str] = None) -> None:
    """
    替换NumPack文件中指定数组的指定索引位置的数据。

    Parameters:
        filename (str or PathLike): 文件路径，扩展名会自动更改为.npk
        arrays (dict or np.ndarray): 要替换的新数据
        indexes (np.ndarray): 要替换的索引位置
        array_name (str, optional): 当arrays为单个numpy数组时的数组名称

    Returns:
        None
    """
    import numpack

    # 确保文件扩展名为.npk
    filename = Path(filename).with_suffix('.npk')

    if isinstance(arrays, np.ndarray):
        if array_name is None:
            raise NnpFileSavingError("Must provide array_name when replacing single array")
        arrays = {array_name: arrays}

    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        try:
            # 使用context manager创建NumPack实例
            with numpack.NumPack(str(filename)) as np_instance:
                # 转换indexes为列表
                indices_list = indexes.tolist() if isinstance(indexes, np.ndarray) else list(indexes)

                # 转换数组数据
                arrays_dict = {}
                for name, new_data in arrays.items():
                    # 转换为2D数组
                    if new_data.ndim == 1:
                        new_data = new_data.reshape(-1, 1)
                    arrays_dict[name] = new_data

                # 使用新API: replace({'name': data}, indices=[...])
                np_instance.replace(arrays_dict, indices=indices_list)

        except Exception as e:
            raise NnpFileSavingError(f"Error replacing arrays: {str(e)}")


def drop_arrays(filename: Union[str, Path], indexes: np.ndarray,
                array_names: Optional[Union[str, List[str]]] = None) -> None:
    """
    从NumPack文件中删除指定数组的指定索引位置的数据。

    Parameters:
        filename (str or PathLike): 文件路径，扩展名会自动更改为.npk
        indexes (np.ndarray): 要删除的索引位置
        array_names (str or list, optional): 要操作的数组名称。如果为None，对所有数组进行操作

    Returns:
        None
    """
    import numpack

    # 确保文件扩展名为.npk
    filename = Path(filename).with_suffix('.npk')

    if isinstance(array_names, str):
        array_names = [array_names]

    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        try:
            # 使用context manager创建NumPack实例
            with numpack.NumPack(str(filename)) as np_instance:
                # 转换indexes为列表
                indices_list = indexes.tolist() if isinstance(indexes, np.ndarray) else list(indexes)

                # 确定要操作的数组
                if array_names is None:
                    # 获取所有成员名称
                    array_names = list(np_instance.get_member_list())

                # 删除指定索引的数据
                for name in array_names:
                    # 使用新API: drop('name', [indices])
                    np_instance.drop(name, indices_list)

        except Exception as e:
            raise NnpFileSavingError(f"Error dropping array data: {str(e)}")


def get_array_info(filename: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    获取NumPack文件中所有数组的信息。

    Parameters:
        filename (str or PathLike): 文件路径，扩展名会自动更改为.npk

    Returns:
        dict: 数组信息，格式为 {array_name: {'shape': tuple, 'dtype': str}}
    """
    # 确保文件扩展名为.npk
    filename = Path(filename).with_suffix('.npk')

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        # 加载数组信息（不加载实际数据）
        arrays = _load_numpack_arrays(filename, mmap_mode=True)
        info = {}
        for name, array in arrays.items():
            info[name] = {
                'shape': array.shape,
                'dtype': str(array.dtype)
            }
        return info
    except Exception as e:
        raise NnpFileSavingError(f"Error getting array info: {str(e)}")


def migrate_nnp_to_numpack(old_filename: Union[str, Path], new_filename: Union[str, Path]) -> None:
    """
    将旧的nnp文件迁移到NumPack格式。

    Parameters:
        old_filename (str or PathLike): 旧的nnp文件路径
        new_filename (str or PathLike): 新的NumPack文件路径

    Returns:
        None
    """
    # 动态导入旧的nnp模块
    from .io import load_nnp as old_load_nnp

    # 加载旧格式的数据
    old_arrays = old_load_nnp(old_filename)

    # 保存为新格式
    save_nnp(new_filename, **old_arrays)

    print(f"Successfully migrated {old_filename} to {new_filename}")


def _save_numpack_arrays(filename: Union[str, Path], arrays: Dict[str, np.ndarray]) -> None:
    """
    使用NumPack保存数组到文件的内部函数。
    """
    import numpack

    filename = Path(filename)

    try:
        # 使用context manager创建NumPack实例
        with numpack.NumPack(str(filename)) as np_instance:
            # 保存所有数组
            np_instance.save(arrays)

    except Exception as e:
        raise NnpFileSavingError(f"Failed to save using NumPack: {str(e)}")


def _load_numpack_arrays(filename: Union[str, Path], mmap_mode: bool = False) -> Dict[str, np.ndarray]:
    """
    使用NumPack从文件加载数组的内部函数。
    """
    import numpack

    filename = Path(filename)

    try:
        # 使用context manager创建NumPack实例
        with numpack.NumPack(str(filename)) as np_instance:
            # 获取所有数组名称
            array_names = np_instance.get_member_list()

            # 加载所有数组
            arrays = {}
            for name in array_names:
                # 使用lazy参数实现mmap模式
                arrays[name] = np_instance.load(name, lazy=mmap_mode)

            return arrays

    except Exception as e:
        raise NnpFileSavingError(f"Failed to load using NumPack: {str(e)}")


# 为了兼容性，导出所有主要函数
__all__ = [
    'save_nnp',
    'load_nnp',
    'replace_arrays',
    'drop_arrays',
    'get_array_info',
    'migrate_nnp_to_numpack',
    'NnpFileSavingError',
    'NnpValueError'
]
