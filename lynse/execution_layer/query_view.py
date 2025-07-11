import json
from typing import List, Tuple, Dict, Any

import numpy as np

# 延迟导入重型依赖，以减少未安装环境报错
try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:
    pd = None  # type: ignore

try:
    import polars as pl  # type: ignore
except ModuleNotFoundError:
    pl = None  # type: ignore

try:
    import pyarrow as pa  # type: ignore
except ModuleNotFoundError:
    pa = None  # type: ignore


class QueryView:
    def __init__(self, results):
        self.results = results

    # ------------------------------------------------------------------
    # 表格相关
    # ------------------------------------------------------------------

    def to_df(self):
        if len(self.results) == 2:  # head/tail/query case
            embeddings, metadata = self.results
            if pd is None:
                raise ImportError("`pandas` is required for `to_df()` / `to_pandas()`")
            df = pd.DataFrame(metadata)
            # 创建一个新的 DataFrame，首先放置 vectors 列，然后是其他列
            result_df = pd.DataFrame()
            result_df['vectors'] = embeddings.tolist()
            # 将原有列按顺序添加到结果中
            for col in df.columns:
                result_df[col] = df[col]
            return result_df
        else:  # search case
            indices, distances, metadata = self.results
            if pd is None:
                raise ImportError("`pandas` is required for `to_df()` / `to_pandas()`")
            # 首先创建包含 vectors 的 DataFrame
            result_df = pd.DataFrame()
            result_df['index'] = indices
            result_df['distance'] = distances
            # 按顺序添加元数据列
            for k in metadata[0].keys():
                result_df[k] = [d[k] for d in metadata]
            return result_df

    def to_pandas(self):
        """`to_df` 的别名。"""
        return self.to_df()

    def to_polars(self):
        if pl is None:
            raise ImportError("`polars` is required for `to_polars()`")
        return pl.from_pandas(self.to_df())

    def to_arrow(self):
        if pa is None:
            raise ImportError("`pyarrow` is required for `to_arrow()`")
        return pa.Table.from_pandas(self.to_df())

    def to_list(self) -> List[Tuple[float, Dict[str, Any]]]:
        if len(self.results) == 2:  # head/tail/query case
            embeddings, metadata = self.results
            return list(zip(embeddings.tolist(), metadata))
        else:  # search case
            indices, distances, metadata = self.results
            return list(zip(distances.tolist(), metadata))

    def to_dict(self) -> Dict[str, Any]:
        if len(self.results) == 2:  # head/tail/query case
            embeddings, metadata = self.results
            return {
                'embeddings': embeddings.tolist(),
                'metadata': metadata
            }
        else:  # search case
            indices, distances, metadata = self.results
            return {
                'indices': indices.tolist(),
                'distances': distances.tolist(),
                'metadata': metadata
            }

    # ------------------------------------------------------------------
    # numpy / tuple / values
    # ------------------------------------------------------------------

    def to_numpy(self):
        """仅在搜索结果（含距离）时返回 `(indices, distances)`。"""
        if len(self.results) == 3:
            indices, distances, _ = self.results
            return indices, distances
        raise ValueError("当前 QueryView 不包含可转换为 numpy 的 (indices, distances)")

    def to_tuple(self):
        return self.results

    @property
    def values(self):
        return self.results

    def __repr__(self):
        # concise summary representation
        if len(self.results) == 2:
            embeddings, _ = self.results
            dim = embeddings.shape[1] if isinstance(embeddings, np.ndarray) else None
            res_num = len(embeddings)
        else:
            # search case
            indices, _, _ = self.results
            dim = None  # vector dim not directly available here
            res_num = len(indices)
        return f"QueryView(dim={dim}, res_num={res_num})"

    def __str__(self):
        return self.__repr__()

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, **kwargs):
        """将结果序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), **kwargs)
