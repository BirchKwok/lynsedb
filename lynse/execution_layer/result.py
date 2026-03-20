from __future__ import annotations

import json
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

# Optional heavy deps import inside functions when needed


class Result:
    """封装向量搜索结果的对象。

    该对象既向后兼容 *tuple* 解包（``indices, distances, fields = result``），
    又提供 ``to_*`` 系列方法方便转换。
    """

    def __init__(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        fields: Optional[List[Dict[str, Any]]] = None,
        *,
        dim: Optional[int] = None,
        k: Optional[int] = None,
        index_mode: Optional[str] = None,
        res_num: Optional[int] = None,
    ) -> None:
        self.indices: np.ndarray = indices
        self.distances: np.ndarray = distances
        self.fields: Optional[List[Dict[str, Any]]] = fields

        # meta info for concise repr
        self._meta: Dict[str, Any] = {
            'dim': dim,
            'k': k,
            'index_mode': index_mode,
            'res_num': res_num if res_num is not None else (len(indices) if isinstance(indices, np.ndarray) else None),
        }

    # ---------------------------------------------------------------------
    # 魔术方法 —— 保持与旧接口兼容
    # ---------------------------------------------------------------------

    def __iter__(self):
        """支持 ``indices, distances, fields = result`` 解包。"""
        yield self.indices
        yield self.distances
        yield self.fields

    def __len__(self):
        """使 ``len(result)`` == 3，以保持与 tuple 行为一致。"""
        return 3

    def __getitem__(self, item):
        """保持切片 / 索引访问习惯。"""
        if item == 0:
            return self.indices
        if item == 1:
            return self.distances
        if item == 2 or item == -1:
            return self.fields
        raise IndexError("Result only contains 3 elements: indices, distances, fields")

    def __repr__(self) -> str:  # pragma: no cover
        m = self._meta
        return (
            f"SearchResult(dim={m.get('dim')}, "
            f"k={m.get('k')}, index_mode='{m.get('index_mode')}', res_num={m.get('res_num')})"
        )

    __str__ = __repr__

    # ---------------------------------------------------------------------
    # to_* 系列方法
    # ---------------------------------------------------------------------

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """以 ``(indices, distances)`` 的形式返回 ``numpy.ndarray``。"""
        return self.indices, self.distances

    def _flatten(self):
        """内部工具：将结果展开成一维列表，方便后续转换。"""
        if self.indices.ndim == 1:
            ids = self.indices
            dists = self.distances
        else:
            # (n_queries, k)  -> 扁平化
            ids = self.indices.reshape(-1)
            dists = self.distances.reshape(-1)
        return ids, dists

    def to_list(self) -> List[Dict[str, Any]]:
        """转换为 ``List[Dict]``，包含 ``id``、 ``distance`` 及 ``fields``。"""
        ids, dists = self._flatten()
        out: List[Dict[str, Any]] = []
        if self.fields is None:
            for idx, dist in zip(ids, dists):
                out.append({"id": int(idx), "distance": float(dist)})
        else:
            # 当字段存在但可能为空时进行检查
            if not self.fields:
                for idx, dist in zip(ids, dists):
                    out.append({"id": int(idx), "distance": float(dist)})
            else:
                # fields 可能是嵌套 list；同样扁平化处理
                if isinstance(self.fields[0], list):
                    flat_fields = [f for sub in self.fields for f in sub]
                else:
                    flat_fields = self.fields
                for idx, dist, fld in zip(ids, dists, flat_fields):
                    out.append({"id": int(idx), "distance": float(dist), "fields": fld})
        return out

    def to_dict(self) -> List[Dict[str, Any]]:
        """``to_list`` 的别名，为向前兼容保留。"""
        return self.to_list()

    def to_json(self, **kwargs) -> str:
        """转换为 JSON 字符串。``**kwargs`` 会直接透传给 ``json.dumps``。"""
        return json.dumps(self.to_list(), **kwargs)

    def to_pandas(self):  # type: ignore[valid-type]
        """转换为 **pandas** ``DataFrame``。若未安装 pandas 则抛出 ``ImportError``。

        返回的 DataFrame 格式:
        - id: 向量 ID
        - distance: 距离
        - 其他字段列（按照字段首次出现的顺序）

        列顺序: id, distance, 第一个字段, 第二个字段, ...
        """
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError("`pandas` is required for `to_pandas()`") from exc

        # 直接构建 DataFrame 列，避免先转换为字典列表的开销
        ids, dists = self._flatten()

        if self.fields is None or not self.fields:
            # 没有 fields 数据，直接创建简单 DataFrame
            return pd.DataFrame({
                'id': ids,
                'distance': dists
            })

        # 处理 fields - 可能是嵌套 list
        if isinstance(self.fields[0], list):
            flat_fields = [f for sub in self.fields for f in sub]
        else:
            flat_fields = self.fields

        # 首先创建 id 和 distance 列
        data = {
            'id': ids,
            'distance': dists
        }

        # 如果有 fields，将每个字段展开为独立的列
        if flat_fields and len(flat_fields) == len(ids):
            # 获取字段键的有序列表（从第一个非空记录获取顺序）
            all_keys = []
            seen = set()
            for fld in flat_fields:
                if fld:
                    for key in fld.keys():
                        if key not in seen:
                            all_keys.append(key)
                            seen.add(key)

            # 为每个字段键创建一列
            for key in all_keys:
                data[key] = [fld.get(key) if fld else None for fld in flat_fields]

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # 与 QueryView 对齐的其它格式化方法
    # ------------------------------------------------------------------

    def to_df(self):  # alias of to_pandas
        """`to_pandas` 的别名，与 `QueryView` 行为保持一致。"""
        return self.to_pandas()

    def to_polars(self):  # type: ignore[valid-type]
        """转换为 **polars** DataFrame。若未安装则抛出 ``ImportError``。

        列顺序: id, distance, 第一个字段, 第二个字段, ...
        """
        try:
            import polars as pl  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError("`polars` is required for `to_polars()`") from exc

        # 直接从原始数据构建 Polars DataFrame，避免先转换为 pandas
        ids, dists = self._flatten()

        if self.fields is None or not self.fields:
            return pl.DataFrame({
                'id': ids,
                'distance': dists
            })

        # 处理 fields
        if isinstance(self.fields[0], list):
            flat_fields = [f for sub in self.fields for f in sub]
        else:
            flat_fields = self.fields

        data = {
            'id': ids,
            'distance': dists
        }

        if flat_fields and len(flat_fields) == len(ids):
            # 获取字段键的有序列表
            all_keys = []
            seen = set()
            for fld in flat_fields:
                if fld:
                    for key in fld.keys():
                        if key not in seen:
                            all_keys.append(key)
                            seen.add(key)

            for key in all_keys:
                data[key] = [fld.get(key) if fld else None for fld in flat_fields]

        return pl.DataFrame(data)

    def to_arrow(self):  # type: ignore[valid-type]
        """转换为 **pyarrow** Table。若未安装则抛出 ``ImportError``。

        列顺序: id, distance, 第一个字段, 第二个字段, ...
        """
        try:
            import pyarrow as pa  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError("`pyarrow` is required for `to_arrow()`") from exc

        # 直接从原始数据构建 PyArrow Table，避免先转换为 pandas
        ids, dists = self._flatten()

        arrays = [pa.array(ids), pa.array(dists)]
        field_names = ['id', 'distance']

        if self.fields and len(self.fields) > 0:
            # 处理 fields
            if isinstance(self.fields[0], list):
                flat_fields = [f for sub in self.fields for f in sub]
            else:
                flat_fields = self.fields

            if flat_fields and len(flat_fields) == len(ids):
                # 获取字段键的有序列表
                all_keys = []
                seen = set()
                for fld in flat_fields:
                    if fld:
                        for key in fld.keys():
                            if key not in seen:
                                all_keys.append(key)
                                seen.add(key)

                for key in all_keys:
                    arrays.append(pa.array([fld.get(key) if fld else None for fld in flat_fields]))
                    field_names.append(key)

        return pa.table(arrays, names=field_names)

    # tuple / values ------------------------------------------------------
    def to_tuple(self):
        """返回 ``(indices, distances, fields)`` 元组。"""
        return (self.indices, self.distances, self.fields)

    @property
    def values(self):
        """与 `QueryView.values` 对齐。"""
        return self.to_tuple()
