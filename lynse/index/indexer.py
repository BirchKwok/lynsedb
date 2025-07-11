from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List
import json
import time

import numpy as np
from spinesUtils.asserts import raise_if
from ultralog import UltraLog as Logger

from ..core_components.locks import ThreadLock
from ..storage_layer.storage import PersistentFileStorage
from ..utils.utils import drop_duplicated_substr, find_first_file_with_substr, SafeMmapReader

from .builder import IndexBuilder, create_index_from_string


class Indexer:
    """Manager class for vector indices."""

    # 索引类型映射表
    _INDEX_ALIAS = {
        # Flat indices
        'FLAT': 'flat-ip',
        'Flat-IP': 'flat-ip',
        'Flat-L2': 'flat-l2',
        'Flat-Cos': 'flat-cosine',
        'Flat-IP-SQ8': 'flat-ip-sq8',
        'Flat-L2-SQ8': 'flat-l2-sq8',
        'Flat-Cos-SQ8': 'flat-cosine-sq8',
        'Flat-Jaccard-Binary': 'flat-jaccard',
        'Flat-Hamming-Binary': 'flat-hamming',

        # HNSW indices
        'HNSW': 'hnsw-ip',
        'HNSW-IP': 'hnsw-ip',
        'HNSW-L2': 'hnsw-l2',
        'HNSW-Cos': 'hnsw-cosine',
        'HNSW-IP-SQ8': 'hnsw-ip-sq8',
        'HNSW-L2-SQ8': 'hnsw-l2-sq8',
        'HNSW-Cos-SQ8': 'hnsw-cosine-sq8',

        # DiskANN indices
        'DiskANN': 'diskann-ip',
        'DiskANN-IP': 'diskann-ip',
        'DiskANN-L2': 'diskann-l2',
        'DiskANN-Cos': 'diskann-cosine',
        'DiskANN-IP-SQ8': 'diskann-ip-sq8',
        'DiskANN-L2-SQ8': 'diskann-l2-sq8',
        'DiskANN-Cos-SQ8': 'diskann-cosine-sq8',

        # IVF indices
        'IVF': 'ivf-ip',
        'IVF-IP': 'ivf-ip',
        'IVF-L2': 'ivf-l2',
        'IVF-Cos': 'ivf-cosine',
        'IVF-IP-SQ8': 'ivf-ip-sq8',
        'IVF-L2-SQ8': 'ivf-l2-sq8',
        'IVF-Cos-SQ8': 'ivf-cosine-sq8',
        'IVF-Jaccard-Binary': 'ivf-jaccard',
        'IVF-Hamming-Binary': 'ivf-hamming'
    }

    def __init__(
            self,
            logger: Logger,
            dataloader: Any,
            storage_worker: PersistentFileStorage,
            collections_path_parent: Path,
    ):
        """
        Initialize the index manager.

        Parameters:
            logger: Logger instance
            dataloader: Data loading function
            storage_worker: Storage manager instance
            collections_path_parent: Path to collections directory
        """
        self.logger = logger
        self.dataloader = dataloader
        self.storage_worker = storage_worker
        self.collections_path_parent = Path(collections_path_parent)

        # Create necessary directories
        self.index_data_path = self.collections_path_parent / 'index_data'
        self.index_ids_path = self.collections_path_parent / 'index_ids'
        self.index_path = self.collections_path_parent / 'index'
        self.index_meta_path = self.collections_path_parent / 'index_meta'

        for path in [self.index_data_path, self.index_ids_path, self.index_path, self.index_meta_path]:
            path.mkdir(parents=True, exist_ok=True)

        self.lock = ThreadLock()
        self.mmap_reader = SafeMmapReader()

        # Index state
        self.index = None
        self.index_mode = None
        self.current_index_mode = None

        # 数据同步状态
        self.last_sync_fingerprint = None
        self.pending_updates = []

        # 索引元数据
        self.index_metadata = self._load_index_metadata()

    def _get_data_file_paths(self, index_type: str, fingerprint: str = None) -> Tuple[Path, Path]:
        """
        获取数据文件路径，使用一致的.npk扩展名。

        参数:
            index_type: 索引类型
            fingerprint: 数据指纹，如果为None则使用当前指纹

        返回:
            (data_path, ids_path): 数据文件路径和ID文件路径
        """
        if fingerprint is None:
            fingerprint = self.storage_worker.fingerprint

        if 'SQ8' in index_type:
            data_path = self.index_data_path / f'{fingerprint}.sqd.npk'
            ids_path = self.index_ids_path / f'{fingerprint}.sqi.npk'
        elif 'Binary' in index_type:
            data_path = self.index_data_path / f'{fingerprint}.bd.npk'
            ids_path = self.index_ids_path / f'{fingerprint}.bi.npk'
        else:
            data_path = self.index_data_path / f'{fingerprint}.dat.npk'
            ids_path = self.index_ids_path / f'{fingerprint}.ids.npk'

        return data_path, ids_path

    def _get_index_metadata_path(self, index_type: str) -> Path:
        """获取索引元数据文件路径。"""
        return self.index_meta_path / f"{index_type}.meta.json"

    def _load_index_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载所有索引的元数据。"""
        metadata = {}
        for meta_file in self.index_meta_path.glob("*.meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    index_type = meta_file.stem.replace('.meta', '')
                    metadata[index_type] = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {meta_file}: {e}")
        return metadata

    def _save_index_metadata(self, index_type: str, metadata: Dict[str, Any]) -> None:
        """保存索引元数据。"""
        meta_path = self._get_index_metadata_path(index_type)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_index_metadata(self, index_type: str, **kwargs) -> None:
        """更新索引元数据。"""
        if index_type not in self.index_metadata:
            self.index_metadata[index_type] = {}

        metadata = self.index_metadata[index_type]
        metadata.update({
            'last_modified': time.time(),
            'vector_count': len(self.index.ids) if self.index is not None else 0,
            'data_fingerprint': self.storage_worker.fingerprint,
            **kwargs
        })

        self._save_index_metadata(index_type, metadata)

    def _serialize_index(self, index_type: str) -> None:
        """
        序列化索引到磁盘。

        参数:
            index_type: 索引类型
        """
        with self.lock:
            if self.index is None:
                return

            # 保存索引结构
            index_path = self.index_path / f'{index_type}.index'
            self.index.save(index_path)

            # 保存索引数据
            if hasattr(self.index, "save_data"):
                data_path, ids_path = self._get_data_file_paths(index_type)
                self.index.save_data(data_path, ids_path)

            # 更新元数据
            self._update_index_metadata(index_type,
                                     serialization_time=time.time(),
                                     data_paths={
                                         'index': str(index_path),
                                         'data': str(data_path) if hasattr(self.index, "save_data") else None,
                                         'ids': str(ids_path) if hasattr(self.index, "save_data") else None
                                     })

    def _deserialize_index(self, index_type: str) -> bool:
        """
        从磁盘反序列化索引。

        参数:
            index_type: 索引类型

        返回:
            是否成功加载
        """
        try:
            # 检查元数据
            if index_type not in self.index_metadata:
                return False

            metadata = self.index_metadata[index_type]
            data_paths = metadata.get('data_paths', {})

            # 检查文件是否存在
            index_path = Path(data_paths.get('index', ''))
            if not index_path.exists():
                return False

            # 创建新的索引实例
            self.index = self._create_index(index_type)

            # 加载索引结构
            self.index.load(index_path)

            # 加载索引数据
            if hasattr(self.index, "load_data"):
                data_path, ids_path = self._get_data_file_paths(index_type)

                if data_path.exists() and ids_path.exists():
                    self.index.load_data(data_path, ids_path)
                    self.last_sync_fingerprint = self.storage_worker.fingerprint

            # 更新元数据
            self._update_index_metadata(index_type,
                                     deserialization_time=time.time(),
                                     last_loaded=time.time())

            return True

        except Exception as e:
            self.logger.error(f"Failed to deserialize index {index_type}: {e}")
            return False

    def _resolve_index_type(self, index_type: str) -> str:
        """Resolve index type alias to canonical name (case-insensitive)."""
        # 尝试大小写不敏感匹配
        # 1) 直接匹配（原始大小写）
        if index_type in self._INDEX_ALIAS:
            return self._INDEX_ALIAS[index_type]

        # 2) 忽略大小写匹配
        lower_type = index_type.lower()
        for key, value in self._INDEX_ALIAS.items():
            if key.lower() == lower_type:
                return value

        # 3) 输入本身已是规范化名称？（比较 value 列表，忽略大小写）
        for value in self._INDEX_ALIAS.values():
            if lower_type == value.lower():
                return value

        raise ValueError(f"Unknown index type: {index_type}")

    def _create_index(self, index_type: str, **kwargs) -> Any:
        """Create an index instance."""
        # 先解析别名
        index_type = self._resolve_index_type(index_type)

        # Handle quantization parameters
        if 'SQ8' in index_type:
            kwargs['quantizer'] = 'sq'
            kwargs['bits'] = 8
        elif 'Binary' in index_type:
            kwargs['quantizer'] = 'binary'

        # Create index using string specification
        return create_index_from_string(index_type.lower())

    def _try_load_existing_index(self, index_type: str) -> bool:
        """
        尝试加载已存在的索引。

        参数:
            index_type: 索引类型

        返回:
            是否成功加载
        """
        index_path = self.index_path / f'{index_type}.index'
        if not index_path.exists():
            return False

        try:
            # 创建新的索引实例
            self.index = self._create_index(index_type)
            # 加载索引状态
            self.index.load(index_path)

            # 检查是否需要加载数据
            if hasattr(self.index, "load_data"):
                data_path, ids_path = self._get_data_file_paths(index_type)

                if data_path.exists() and ids_path.exists():
                    self.index.load_data(data_path, ids_path)
                    self.last_sync_fingerprint = self.storage_worker.fingerprint
                    return True

            return True
        except Exception as e:
            self.logger.error(f"Failed to load existing index: {e}")
            return False

    def _check_data_updates(self) -> bool:
        """
        检查是否有数据更新。

        返回:
            是否有更新
        """
        if self.last_sync_fingerprint != self.storage_worker.fingerprint:
            return True

        # 检查是否有未同步的文件
        current_files = set(self.storage_worker.get_all_files())
        indexed_files = set(self.index.metadata.get('indexed_files', []))

        return bool(current_files - indexed_files)

    def _sync_data_updates(self) -> None:
        """同步数据更新到索引。"""
        with self.lock:
            # 获取新增文件
            current_files = set(self.storage_worker.get_all_files())
            indexed_files = set(self.index.metadata.get('indexed_files', []))
            new_files = current_files - indexed_files

            # 处理新增文件
            for filename in new_files:
                data_dict = self.dataloader(filename)

                # 从字典中提取实际的数组数据
                if isinstance(data_dict, dict):
                    if "data" in data_dict:
                        data = data_dict["data"]
                    else:
                        # 如果没有 "data" 键，使用第一个数组
                        data = list(data_dict.values())[0]
                else:
                    # 向后兼容：如果直接返回数组
                    data = data_dict

                self.index.fit_transform(data)
                indexed_files.add(filename)

            # 更新元数据
            self.index.metadata['indexed_files'] = list(indexed_files)

            # 保存更新后的索引
            self.index.save(self.index_path / f'{self.current_index_mode}.index')

            # 更新同步状态
            self.last_sync_fingerprint = self.storage_worker.fingerprint

    def build_index(self, index_type: str = 'IVF-IP-SQ8', **kwargs) -> None:
        """
        Build the index.

        Parameters:
            index_type: Type of index to build
            **kwargs: Additional parameters for index construction
        """
        # Close any existing memory-mapped files
        self.close_mapped_index()

        # Resolve index type
        index_type = self._resolve_index_type(index_type)

        # Check if we need to switch to flat index for small datasets
        all_partition_size = self.storage_worker.get_shape()[0]
        if all_partition_size < 100000 and ('Binary' in index_type or 'SQ8' in index_type):
            substr = '-Binary' if 'Binary' in index_type else '-SQ8'
            index_type = drop_duplicated_substr(index_type, substr).replace("IVF", "Flat")
            self.logger.info(
                'Index is not built because the number of data points is less than 100000. '
                f'Continue to build the {index_type} index.'
            )

        # 检查是否可以复用现有索引
        if self.index_mode == index_type:
            # 检查数据更新
            if self._check_data_updates():
                self._sync_data_updates()
            self.logger.info('Index reused and synchronized.')
            return

        # 尝试加载已存在的索引
        if self._try_load_existing_index(index_type):
            self.index_mode = index_type
            self.current_index_mode = index_type
            # 检查数据更新
            if self._check_data_updates():
                self._sync_data_updates()
            self.logger.info('Existing index loaded and synchronized.')
            return

        # 如果无法复用,则重新构建索引
        self.logger.info(f'Building an index using the `{index_type}` index mode...')

        with self.lock:
            # Create and build the index
            self.index = self._create_index(index_type, **kwargs)

            # Load and index all data
            indexed_files = set()
            for filename in self.storage_worker.get_all_files():
                data_dict = self.dataloader(filename)

                # 从字典中提取实际的数组数据
                if isinstance(data_dict, dict):
                    if "data" in data_dict:
                        data = data_dict["data"]
                    else:
                        # 如果没有 "data" 键，使用第一个数组
                        data = list(data_dict.values())[0]
                else:
                    # 向后兼容：如果直接返回数组
                    data = data_dict

                self.index.fit_transform(data)
                indexed_files.add(filename)

            # 更新元数据
            self.index.metadata['indexed_files'] = list(indexed_files)

            # Save the index
            self.index.save(self.index_path / f'{index_type}.index')

            # Save encoded data if needed
            if hasattr(self.index, "save_data"):
                data_path, ids_path = self._get_data_file_paths(index_type)
                self.index.save_data(data_path, ids_path)

            # 更新状态
            self.index_mode = index_type
            self.current_index_mode = index_type
            self.last_sync_fingerprint = self.storage_worker.fingerprint

        self.logger.info('Index built.')

        # 保存索引和更新元数据
        self._serialize_index(index_type)

    def search(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index.

        Parameters:
            **kwargs: Search parameters

        Returns:
            Tuple of (indices, distances)
        """
        # 检查数据更新
        if self._check_data_updates():
            self._sync_data_updates()

        return self.index.search(**kwargs)

    def index_insert(self, data: np.ndarray, ids: np.ndarray) -> None:
        """
        Insert new data into the index.

        Parameters:
            data: Data vectors to insert
            ids: IDs for the vectors
        """
        raise_if(ValueError, self.index is None, 'The index must be built before inserting data.')

        with self.lock:
            self.index.fit_transform(data, ids)
            # 保存更新后的索引
            self.index.save(self.index_path / f'{self.current_index_mode}.index')

            # 保存编码数据
            if hasattr(self.index, "save_data"):
                data_path, ids_path = self._get_data_file_paths(self.current_index_mode)
                self.index.save_data(data_path, ids_path)

    def update_filenames(self) -> None:
        """Update filenames after storage changes."""
        # Get new fingerprint
        with open(self.storage_worker.fingerprint_path, 'r') as file:
            new_fingerprint = file.readlines()[-1].strip()

        self._remove_old_data()
        if hasattr(self.index, 'save_data'):
            data_path, ids_path = self._get_data_file_paths(self.index_mode, new_fingerprint)
            self.index.save_data(data_path, ids_path)
            self.last_sync_fingerprint = new_fingerprint

    def _remove_old_data(self) -> None:
        """Remove old index data files."""
        if not self.storage_worker.fingerprint_path.exists():
            return

        with open(self.storage_worker.fingerprint_path, 'r') as file:
            old_fingerprints = file.readlines()[:-1]

        if old_fingerprints:
            for fingerprint in old_fingerprints:
                fingerprint = fingerprint.strip()
                for path in [self.index_data_path, self.index_ids_path, self.index_path]:
                    for file in path.iterdir():
                        if fingerprint in file.name:
                            file.unlink()

    def remove_index(self) -> None:
        """Remove the current index."""
        if self.current_index_mode is not None:
            # 删除索引文件和元数据
            self.cleanup_old_indices(max_age_days=0)

        self.index = None
        self.index_mode = None
        self.current_index_mode = None
        self.last_sync_fingerprint = None
        self.logger.info('Index removed.')

    def close_mapped_index(self) -> None:
        """Close memory-mapped files."""
        self.mmap_reader.close()

    def __del__(self):
        self.close_mapped_index()

    def add_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        向索引中添加新的向量。

        参数:
            vectors: 要添加的向量数据
            ids: 向量的ID，如果为None则自动生成

        返回:
            添加的向量ID
        """
        with self.lock:
            if self.index is None:
                raise ValueError("索引尚未初始化")

            # 添加向量到索引
            encoded_data = self.index.fit_transform(vectors, ids)

            # 保存更新后的索引
            self.index.save(self.index_path / f'{self.current_index_mode}.index')

            # 如果索引支持单独保存数据，也保存数据
            if hasattr(self.index, "save_data"):
                data_path, ids_path = self._get_data_file_paths(self.current_index_mode)
                self.index.save_data(data_path, ids_path)

            return self.index.ids[-len(vectors):]

    def delete_vectors(self, ids: Union[int, List[int], np.ndarray]) -> None:
        """
        从索引中删除指定ID的向量。

        参数:
            ids: 要删除的向量ID，可以是单个ID或ID列表
        """
        with self.lock:
            if self.index is None:
                raise ValueError("索引尚未初始化")

            # 删除向量
            self.index.delete(ids)

            # 保存更新后的索引
            self.index.save(self.index_path / f'{self.current_index_mode}.index')

            # 如果索引支持单独保存数据，也保存数据
            if hasattr(self.index, "save_data"):
                data_path, ids_path = self._get_data_file_paths(self.current_index_mode)
                self.index.save_data(data_path, ids_path)

    def update_vectors(self, vectors: np.ndarray, ids: Union[int, List[int], np.ndarray]) -> None:
        """
        更新索引中指定ID的向量。

        参数:
            vectors: 新的向量数据
            ids: 要更新的向量ID，可以是单个ID或ID列表
        """
        with self.lock:
            if self.index is None:
                raise ValueError("索引尚未初始化")

            # 更新向量
            self.index.update(vectors, ids)

            # 保存更新后的索引
            self.index.save(self.index_path / f'{self.current_index_mode}.index')

            # 如果索引支持单独保存数据，也保存数据
            if hasattr(self.index, "save_data"):
                data_path, ids_path = self._get_data_file_paths(self.current_index_mode)
                self.index.save_data(data_path, ids_path)

    def get_index_stats(self) -> Dict[str, Any]:
        """
        获取所有索引的统计信息。

        返回:
            包含索引统计信息的字典
        """
        stats = {
            'current_index': self.current_index_mode,
            'total_indices': len(self.index_metadata),
            'indices': self.index_metadata
        }

        if self.index is not None:
            stats.update({
                'current_index_size': len(self.index.ids),
                'current_index_dimension': self.index.encoded_data.shape[1] if self.index.encoded_data is not None else None,
                'current_index_memory': self.index.encoded_data.nbytes if self.index.encoded_data is not None else None
            })

        return stats

    def cleanup_old_indices(self, max_age_days: float = 30) -> None:
        """
        清理旧的索引文件。

        参数:
            max_age_days: 最大保留天数
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        with self.lock:
            for index_type, metadata in list(self.index_metadata.items()):
                last_modified = metadata.get('last_modified', 0)
                if current_time - last_modified > max_age_seconds:
                    # 删除索引文件
                    data_paths = metadata.get('data_paths', {})
                    for path in data_paths.values():
                        if path:
                            try:
                                Path(path).unlink(missing_ok=True)
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {path}: {e}")

                    # 删除元数据
                    meta_path = self._get_index_metadata_path(index_type)
                    try:
                        meta_path.unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {meta_path}: {e}")

                    # 从内存中移除元数据
                    del self.index_metadata[index_type]

                    self.logger.info(f"Cleaned up old index: {index_type}")
