from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from ultralog import UltraLog as Logger
from spinesUtils.asserts import raise_if
from ..storage_layer.storage import PersistentFileStorage
from ..core_components.fields_cache import FieldsCache
from ..core_components.limited_sort import LimitedSorted
from ..computational_layer import engines
from ..configs.config import config
from lynse.index.indexer import Indexer
# 新增：Result 对象
from .result import Result


class Search:
    """向量搜索模块。"""

    def __init__(
            self,
            logger: Logger,
            storage_worker: PersistentFileStorage,
            indexer: Indexer,
            field_index: FieldsCache,
            dtypes: Any,
            n_threads: int = 10,
            use_simd: Optional[bool] = None
    ):
        """
        初始化搜索模块。

        参数:
            logger: 日志实例
            storage_worker: 存储管理器实例
            indexer: 索引管理器实例
            field_index: 字段索引实例
            dtypes: 数据类型
            n_threads: 搜索线程数
            use_simd: 是否使用SIMD加速，None表示使用配置默认值
        """
        self.logger = logger
        self.storage_worker = storage_worker
        self.indexer = indexer
        self.field_index = field_index
        self.n_threads = n_threads
        self.use_simd = use_simd if use_simd is not None else config.LYNSE_USE_SIMSIMD

        self.dtypes = dtypes

        # 初始化线程池
        self._executor = ThreadPoolExecutor(max_workers=n_threads)
        self._is_closed = False

    def _apply_filter(self, search_filter: str) -> list:
        """
        应用搜索过滤器。

        参数:
            search_filter (str): the sql-like string to apply

        返回:
            过滤后的索引数组
        """
        # 使用字段索引进行过滤
        return self.field_index.query(search_filter)

    def _process_chunk(
            self,
            query_vec: np.ndarray,
            chunk_data: np.ndarray,
            chunk_ids: np.ndarray,
            k: int,
            limited_sorted: LimitedSorted
    ) -> None:
        """
        处理数据块。

        参数:
            query_vec: 查询向量
            chunk_data: 数据块
            chunk_ids: 数据块对应的ID
            k: 返回的邻居数量
            limited_sorted: 排序器实例
        """
        try:
            # 确保query_vec是二维数组
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)

            # 根据索引类型选择合适的距离计算方法，传递use_simd参数
            if hasattr(self.indexer.index, 'name'):
                index_name = self.indexer.index.name.lower()
                if 'ip' in index_name:
                    # 内积
                    indices, distances = engines.ip(query_vec, chunk_data, k, use_simd=self.use_simd)
                elif 'l2' in index_name:
                    # L2距离
                    indices, distances = engines.l2sq(query_vec, chunk_data, k, use_simd=self.use_simd)
                elif 'cos' in index_name:
                    # 余弦距离
                    indices, distances = engines.cosine(query_vec, chunk_data, k, use_simd=self.use_simd)
                elif 'hamming' in index_name:
                    # 汉明距离
                    indices, distances = engines.hamming(query_vec, chunk_data, k, use_simd=self.use_simd)
                elif 'jaccard' in index_name:
                    # Jaccard距离
                    indices, distances = engines.jaccard(query_vec, chunk_data, k, use_simd=self.use_simd)
                else:
                    # 默认使用内积
                    indices, distances = engines.ip(query_vec, chunk_data, k, use_simd=self.use_simd)
            else:
                # 默认使用内积
                indices, distances = engines.ip(query_vec, chunk_data, k, use_simd=self.use_simd)

            # 使用原始的chunk_ids映射回实际的ID
            actual_indices = chunk_ids[indices.squeeze()]

            # 更新排序结果
            limited_sorted.add(distances, actual_indices)
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}")
            raise

    def get_simd_info(self) -> Dict[str, Any]:
        """
        获取SimSIMD相关信息。

        返回:
            包含SimSIMD配置的字典
        """
        return {
            'use_simd': self.use_simd,
            'global_config': config.LYNSE_USE_SIMSIMD,
            'auto_fallback': config.LYNSE_SIMSIMD_AUTO_FALLBACK,
            'log_fallback': config.LYNSE_SIMSIMD_LOG_FALLBACK,
            'capabilities': engines.get_simsimd_capabilities()
        }

    def set_simd_usage(self, use_simd: Optional[bool]) -> None:
        """
        设置SIMD使用方式。

        参数:
            use_simd: True启用，False禁用，None使用配置默认值
        """
        self.use_simd = use_simd if use_simd is not None else config.LYNSE_USE_SIMSIMD

    def search(
            self,
            vector: np.ndarray,
            k: int = 10,
            nprobe: int = 10,
            search_filter: str = None,
            use_simd: Optional[bool] = None,
            **kwargs
    ) -> Result:
        """
        执行向量搜索。

        参数:
            vector: 查询向量
            k: 返回的最近邻数量
            nprobe: IVF索引的探测数量
            search_filter: 搜索过滤条件
            use_simd: 是否使用SIMD加速，None表示使用实例默认设置
            **kwargs: 额外的搜索参数

        返回:
            Result: 封装查询结果的对象，可通过解包方式获得 (indices, distances, fields) ，
            也可使用 ``to_*`` 方法进行格式转换。
        """
        raise_if(ValueError, self._is_closed, "Search module has been closed")

        # 设置本次搜索的SIMD使用方式
        original_use_simd = self.use_simd
        if use_simd is not None:
            self.use_simd = use_simd

        try:
            # 检查输入
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)

            if vector.dtype != self.dtypes:
                vector = vector.astype(self.dtypes)

            if vector.shape[1] != self.storage_worker.dimension:
                raise ValueError(
                    f"Query vector dimension {vector.shape[1]} does not match "
                    f"index dimension {self.storage_worker.dimension}"
                )

            # 应用过滤器
            subset_indices = self._apply_filter(search_filter) if search_filter is not None else None

            # 准备搜索参数
            search_params = {
                'k': k,
                'nprobe': nprobe,
                'subset_indices': subset_indices,
                'original_vec': vector,  # 用于重排序
                **kwargs
            }

            # 检查是否使用IVF索引
            if hasattr(self.indexer.index, 'ivf_centers') and self.indexer.index.ivf_centers is not None:
                # 使用IVF索引搜索
                indices, distances = self.indexer.search(**search_params)
            else:
                # 使用原始搜索方法
                limited_sorted = LimitedSorted(n=k)

                # 获取所有数据文件
                filenames = self.storage_worker.get_all_files()

                # 多线程处理数据块
                futures = []
                for filename in filenames:
                    # 优先从内存缓存获取数据
                    data_dict = self.storage_worker.dataloader.read(filename, use_mmap=True)

                    # 从字典中提取实际的数组数据
                    if isinstance(data_dict, dict):
                        if "data" in data_dict:
                            chunk_data = data_dict["data"]
                        else:
                            # 如果没有 "data" 键，使用第一个数组
                            chunk_data = list(data_dict.values())[0]
                    else:
                        # 向后兼容：如果直接返回数组
                        chunk_data = data_dict

                    # 获取数据块对应的ID
                    chunk_ids = self.storage_worker.id_mapper[filename].generate_ids(as_range=False)

                    # 如果有过滤条件,应用过滤
                    if subset_indices is not None:
                        mask = np.isin(chunk_ids, subset_indices)
                        if not np.any(mask):
                            continue
                        chunk_data = chunk_data[mask]
                        chunk_ids = chunk_ids[mask]

                    # 提交处理任务
                    future = self._executor.submit(
                        self._process_chunk,
                        vector,
                        chunk_data,
                        chunk_ids,
                        k,
                        limited_sorted
                    )
                    futures.append(future)

                # 等待所有任务完成
                for future in futures:
                    future.result()

                # 获取最终结果
                indices, distances = limited_sorted.get_top_n()

                # 如果是单个查询向量，需要调整输出形状
                if vector.shape[0] == 1:
                    indices = indices.reshape(-1)
                    distances = distances.reshape(-1)

            # 获取字段信息
            fields = self.field_index.retrieve_many(indices.ravel().tolist())

            # 构造 Result，附带简要元信息供 __repr__ 使用
            # 获取 index_mode：优先 current_index_mode > index_mode > index 对象 name
            if getattr(self.indexer, 'current_index_mode', None):
                index_mode = self.indexer.current_index_mode
            elif getattr(self.indexer, 'index_mode', None):
                index_mode = self.indexer.index_mode
            elif hasattr(self.indexer, 'index') and getattr(self.indexer.index, 'name', None):
                index_mode = self.indexer.index.name
            else:
                index_mode = 'flat-ip'
            res_num = len(indices) if isinstance(indices, np.ndarray) else None
            return Result(
                indices,
                distances,
                fields,
                dim=self.storage_worker.dimension,
                k=k,
                index_mode=index_mode,
                res_num=res_num,
            )

        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
        finally:
            # 恢复原始的SIMD设置
            if use_simd is not None:
                self.use_simd = original_use_simd

    def batch_search(
            self,
            query_vecs: np.ndarray,
            k: int = 10,
            nprobe: int = 10,
            search_filter: Optional[Union[str, Dict]] = None,
            use_simd: Optional[bool] = None,
            **kwargs
    ) -> List[Result]:
        """
        执行批量向量搜索。

        参数:
            query_vecs: 查询向量批次
            k: 每个查询返回的最近邻数量
            nprobe: IVF索引的探测数量
            search_filter: 搜索过滤条件
            use_simd: 是否使用SIMD加速，None表示使用实例默认设置
            **kwargs: 额外的搜索参数

        返回:
            List[Result]: 每个查询的搜索结果对象
        """
        raise_if(ValueError, self._is_closed, "Search module has been closed")

        # 检查输入
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)

        if query_vecs.shape[1] != self.storage_worker.dimension:
            raise ValueError(
                f"Query vector dimension {query_vecs.shape[1]} does not match "
                f"index dimension {self.storage_worker.dimension}"
            )

        try:
            # 多线程执行批量搜索
            futures = []
            for query_vec in query_vecs:
                future = self._executor.submit(
                    self.search,
                    query_vec,
                    k,
                    nprobe,
                    search_filter,
                    use_simd,
                    **kwargs
                )
                futures.append(future)

            results_batch: List[Result] = []

            for future in futures:
                result = future.result()
                results_batch.append(result)

            return results_batch

        except Exception as e:
            self.logger.error(f"Batch search failed: {e}")
            raise

    def close(self):
        """关闭搜索模块并释放资源。"""
        if not self._is_closed:
            self._executor.shutdown(wait=True)
            self._is_closed = True
            self.logger.debug('Search module closed.')

    def __del__(self):
        """清理资源。"""
        self.close()
