#!/usr/bin/env python3
"""
SimSIMD性能测试脚本

该脚本用于对比使用SimSIMD优化前后的性能差异。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import logging
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lynse.computational_layer import engines
from lynse.configs.config import config
from lynse.index.flat import FlatIndex

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimSIMDPerformanceTester:
    """SimSIMD性能测试器"""

    def __init__(self):
        self.results = {}

    def generate_test_data(self, n_vectors: int, dimensions: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成测试数据

        参数:
            n_vectors: 向量数量
            dimensions: 向量维度
            dtype: 数据类型

        返回:
            (query_vector, database_vectors): 查询向量和数据库向量
        """
        np.random.seed(42)  # 确保结果可复现

        if dtype == np.int8:
            database_vectors = np.random.randint(-127, 128, (n_vectors, dimensions), dtype=dtype)
            query_vector = np.random.randint(-127, 128, dimensions, dtype=dtype)
        elif dtype == np.uint8:
            database_vectors = np.random.randint(0, 256, (n_vectors, dimensions), dtype=dtype)
            query_vector = np.random.randint(0, 256, dimensions, dtype=dtype)
        elif dtype == np.bool_:
            database_vectors = np.random.choice([True, False], (n_vectors, dimensions)).astype(dtype)
            query_vector = np.random.choice([True, False], dimensions).astype(dtype)
        else:  # float types
            database_vectors = np.random.randn(n_vectors, dimensions).astype(dtype)
            query_vector = np.random.randn(dimensions).astype(dtype)
            # 标准化
            database_vectors = database_vectors / np.linalg.norm(database_vectors, axis=1, keepdims=True)
            query_vector = query_vector / np.linalg.norm(query_vector)

        return query_vector, database_vectors

    def benchmark_distance_function(self, func, query_vec, db_vecs, k, use_simd, runs=10):
        """
        对距离函数进行基准测试

        参数:
            func: 距离函数
            query_vec: 查询向量
            db_vecs: 数据库向量
            k: 返回的近邻数量
            use_simd: 是否使用SIMD
            runs: 运行次数

        返回:
            平均执行时间
        """
        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            try:
                func(query_vec.reshape(1, -1), db_vecs, k, use_simd=use_simd)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed with use_simd={use_simd}: {e}")
                return None
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return np.mean(times)

    def benchmark_index_search(self, index, query_vec, k, runs=10):
        """
        对索引搜索进行基准测试

        参数:
            index: 索引实例
            query_vec: 查询向量
            k: 返回的近邻数量
            runs: 运行次数

        返回:
            平均执行时间
        """
        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            try:
                index.search(query_vec, k=k)
            except Exception as e:
                logger.warning(f"Index search failed: {e}")
                return None
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return np.mean(times)

    def test_engines_performance(self, vector_sizes: List[int], dimensions: int = 1536):
        """
        测试engines模块的性能

        参数:
            vector_sizes: 测试的向量数量列表
            dimensions: 向量维度
        """
        logger.info("Testing engines performance...")

        # 测试不同的数据类型和距离函数
        test_configs = [
            {'dtype': np.float32, 'func': engines.cosine, 'name': 'cosine_f32'},
            {'dtype': np.float16, 'func': engines.cosine, 'name': 'cosine_f16'},
            {'dtype': np.int8, 'func': engines.cosine, 'name': 'cosine_i8'},
            {'dtype': np.float32, 'func': engines.l2sq, 'name': 'l2sq_f32'},
            {'dtype': np.float16, 'func': engines.l2sq, 'name': 'l2sq_f16'},
            {'dtype': np.int8, 'func': engines.l2sq, 'name': 'l2sq_i8'},
            {'dtype': np.float32, 'func': engines.ip, 'name': 'inner_product_f32'},
            {'dtype': np.bool_, 'func': engines.hamming, 'name': 'hamming_bool'},
            {'dtype': np.bool_, 'func': engines.jaccard, 'name': 'jaccard_bool'},
        ]

        for config_item in test_configs:
            dtype = config_item['dtype']
            func = config_item['func']
            name = config_item['name']

            logger.info(f"Testing {name}...")

            simd_times = []
            no_simd_times = []
            sizes = []

            for n_vectors in vector_sizes:
                try:
                    query_vec, db_vecs = self.generate_test_data(n_vectors, dimensions, dtype)
                    k = min(10, n_vectors)

                    # 测试SIMD版本
                    simd_time = self.benchmark_distance_function(func, query_vec, db_vecs, k, True)

                    # 测试非SIMD版本
                    no_simd_time = self.benchmark_distance_function(func, query_vec, db_vecs, k, False)

                    if simd_time is not None and no_simd_time is not None:
                        simd_times.append(simd_time)
                        no_simd_times.append(no_simd_time)
                        sizes.append(n_vectors)

                        speedup = no_simd_time / simd_time if simd_time > 0 else 0
                        logger.info(f"  {n_vectors} vectors: SIMD={simd_time:.6f}s, No-SIMD={no_simd_time:.6f}s, Speedup={speedup:.2f}x")

                except Exception as e:
                    logger.error(f"Error testing {name} with {n_vectors} vectors: {e}")
                    continue

            if simd_times and no_simd_times:
                self.results[name] = {
                    'sizes': sizes,
                    'simd_times': simd_times,
                    'no_simd_times': no_simd_times,
                    'dtype': str(dtype)
                }

    def test_index_performance(self, vector_sizes: List[int], dimensions: int = 1536):
        """
        测试索引性能

        参数:
            vector_sizes: 测试的向量数量列表
            dimensions: 向量维度
        """
        logger.info("Testing index performance...")

        # 测试不同的索引配置
        index_configs = [
            {'distance': 'cosine', 'dtype': np.float32, 'use_simd': True, 'name': 'flat_cosine_simd'},
            {'distance': 'cosine', 'dtype': np.float32, 'use_simd': False, 'name': 'flat_cosine_no_simd'},
            {'distance': 'l2', 'dtype': np.float32, 'use_simd': True, 'name': 'flat_l2_simd'},
            {'distance': 'l2', 'dtype': np.float32, 'use_simd': False, 'name': 'flat_l2_no_simd'},
        ]

        for config_item in index_configs:
            distance = config_item['distance']
            dtype = config_item['dtype']
            use_simd = config_item['use_simd']
            name = config_item['name']

            logger.info(f"Testing {name}...")

            times = []
            sizes = []

            for n_vectors in vector_sizes:
                try:
                    # 生成数据
                    query_vec, db_vecs = self.generate_test_data(n_vectors, dimensions, dtype)

                    # 创建索引
                    index = FlatIndex(distance_metric=distance, use_simd=use_simd)
                    index.fit_transform(db_vecs)

                    # 基准测试
                    k = min(10, n_vectors)
                    time_taken = self.benchmark_index_search(index, query_vec, k)

                    if time_taken is not None:
                        times.append(time_taken)
                        sizes.append(n_vectors)
                        logger.info(f"  {n_vectors} vectors: {time_taken:.6f}s")

                except Exception as e:
                    logger.error(f"Error testing {name} with {n_vectors} vectors: {e}")
                    continue

            if times:
                self.results[name] = {
                    'sizes': sizes,
                    'times': times,
                    'dtype': str(dtype)
                }

    def test_cpu_capabilities(self):
        """测试CPU能力"""
        logger.info("Testing CPU capabilities...")

        capabilities = engines.get_simsimd_capabilities()
        logger.info(f"SimSIMD capabilities: {capabilities}")

        # 测试每种数据类型的支持情况
        test_dtypes = [np.float32, np.float16, np.int8, np.uint8, np.bool_]

        for dtype in test_dtypes:
            try:
                query_vec, db_vecs = self.generate_test_data(100, 128, dtype)

                # 测试cosine距离
                try:
                    engines.cosine(query_vec.reshape(1, -1), db_vecs, 5, use_simd=True)
                    logger.info(f"  {dtype}: SimSIMD cosine - SUPPORTED")
                except Exception as e:
                    logger.warning(f"  {dtype}: SimSIMD cosine - NOT SUPPORTED ({e})")

            except Exception as e:
                logger.error(f"Error testing capability for {dtype}: {e}")

    def plot_results(self, save_path: str = "simsimd_performance_results.png"):
        """
        绘制性能结果图表

        参数:
            save_path: 保存路径
        """
        if not self.results:
            logger.warning("No results to plot")
            return

        # 计算需要的子图数量
        n_plots = len([k for k in self.results.keys() if 'simd_times' in self.results[k]])
        if n_plots == 0:
            logger.warning("No timing comparison results to plot")
            return

        # 设置图表
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        plot_idx = 0
        for name, data in self.results.items():
            if 'simd_times' not in data:
                continue

            ax = axes[plot_idx]

            sizes = data['sizes']
            simd_times = data['simd_times']
            no_simd_times = data['no_simd_times']

            ax.plot(sizes, simd_times, 'b-o', label='SimSIMD', linewidth=2, markersize=6)
            ax.plot(sizes, no_simd_times, 'r-s', label='No SIMD', linewidth=2, markersize=6)

            ax.set_xlabel('Number of Vectors')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{name.replace("_", " ").title()} Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')

            plot_idx += 1

        # 隐藏多余的子图
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance plot saved to {save_path}")
        plt.close()

    def save_results(self, save_path: str = "simsimd_performance_results.txt"):
        """
        保存性能测试结果

        参数:
            save_path: 保存路径
        """
        with open(save_path, 'w') as f:
            f.write("SimSIMD Performance Test Results\n")
            f.write("=" * 50 + "\n\n")

            # CPU能力信息
            f.write("CPU Capabilities:\n")
            capabilities = engines.get_simsimd_capabilities()
            for key, value in capabilities.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # 配置信息
            f.write("Configuration:\n")
            f.write(f"  LYNSE_USE_SIMSIMD: {config.LYNSE_USE_SIMSIMD}\n")
            f.write(f"  LYNSE_SIMSIMD_AUTO_FALLBACK: {config.LYNSE_SIMSIMD_AUTO_FALLBACK}\n")
            f.write(f"  LYNSE_SIMSIMD_LOG_FALLBACK: {config.LYNSE_SIMSIMD_LOG_FALLBACK}\n")
            f.write("\n")

            # 性能结果
            for name, data in self.results.items():
                f.write(f"Test: {name}\n")
                f.write("-" * 30 + "\n")

                if 'simd_times' in data:
                    f.write("Vector Count | SIMD Time | No-SIMD Time | Speedup\n")
                    f.write("-" * 50 + "\n")

                    for i, size in enumerate(data['sizes']):
                        simd_time = data['simd_times'][i]
                        no_simd_time = data['no_simd_times'][i]
                        speedup = no_simd_time / simd_time if simd_time > 0 else 0
                        f.write(f"{size:11d} | {simd_time:9.6f} | {no_simd_time:12.6f} | {speedup:6.2f}x\n")

                    # 计算平均加速比
                    if data['simd_times'] and data['no_simd_times']:
                        try:
                            speedup_ratios = []
                            for i in range(len(data['simd_times'])):
                                simd_time = data['simd_times'][i]
                                no_simd_time = data['no_simd_times'][i]
                                if simd_time > 0:
                                    speedup_ratios.append(no_simd_time / simd_time)

                            if speedup_ratios:
                                avg_speedup = np.mean(speedup_ratios)
                                f.write(f"Average Speedup: {avg_speedup:.2f}x\n")
                            else:
                                f.write("Average Speedup: N/A\n")
                        except Exception as e:
                            f.write(f"Average Speedup: Error calculating ({e})\n")
                    else:
                        f.write("Average Speedup: N/A\n")

                elif 'times' in data:
                    f.write("Vector Count | Time\n")
                    f.write("-" * 20 + "\n")
                    for i, size in enumerate(data['sizes']):
                        f.write(f"{size:11d} | {data['times'][i]:9.6f}\n")

                f.write("\n")

        logger.info(f"Results saved to {save_path}")


def main():
    """主函数"""
    logger.info("Starting SimSIMD performance testing...")

    # 创建测试器
    tester = SimSIMDPerformanceTester()

    # 测试CPU能力
    tester.test_cpu_capabilities()

    # 定义测试参数
    vector_sizes = [100, 500, 1000, 2000, 5000]  # 使用较小的测试规模以加快测试速度
    dimensions = 1536  # OpenAI Ada embeddings维度

    # 测试engines性能
    tester.test_engines_performance(vector_sizes, dimensions)

    # 测试索引性能
    tester.test_index_performance(vector_sizes, dimensions)

    # 绘制结果
    tester.plot_results()

    # 保存结果
    tester.save_results()

    logger.info("Performance testing completed!")


if __name__ == "__main__":
    main()
