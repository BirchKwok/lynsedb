#!/usr/bin/env python3
"""
Comprehensive benchmark for SimSIMD vs NumPy vs USearch
Automatically selects the fastest implementation for each distance metric and data type
"""

import time
import numpy as np
import json
import sys
import os
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import logging

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simsimd
from usearch.compiled import exact_search
from usearch.index import MetricKind
from lynse.core_components.fast_sort import FastSort

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Store benchmark results for each implementation"""
    method: str
    distance_metric: str
    data_type: str
    vector_size: int
    time_mean: float
    time_std: float
    success: bool
    error: str = ""

class ComprehensiveBenchmark:
    """Comprehensive benchmark comparing different similarity computation methods"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.optimal_choices: Dict[str, Dict[str, str]] = {}

    def generate_test_data(self, n_vectors: int, dimensions: int, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test data for benchmarking"""
        np.random.seed(42)  # Reproducible results

        if dtype == np.int8:
            database_vectors = np.random.randint(-127, 128, (n_vectors, dimensions), dtype=dtype)
            query_vector = np.random.randint(-127, 128, (1, dimensions), dtype=dtype)
        elif dtype == np.uint8:
            database_vectors = np.random.randint(0, 256, (n_vectors, dimensions), dtype=dtype)
            query_vector = np.random.randint(0, 256, (1, dimensions), dtype=dtype)
        elif dtype == np.bool_:
            database_vectors = np.random.choice([True, False], (n_vectors, dimensions)).astype(dtype)
            query_vector = np.random.choice([True, False], (1, dimensions)).astype(dtype)
        else:  # float types
            database_vectors = np.random.randn(n_vectors, dimensions).astype(dtype)
            query_vector = np.random.randn(1, dimensions).astype(dtype)
            # Normalize for better cosine similarity results
            database_vectors = database_vectors / np.linalg.norm(database_vectors, axis=1, keepdims=True)
            query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

        return query_vector, database_vectors

    def benchmark_simsimd_cosine(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """SimSIMD cosine distance implementation"""
        if query_vec.shape[0] == 1:
            distances = simsimd.cosine(query_vec.squeeze(), db_vecs)
            topk = FastSort(distances)
            ids, distance = topk.topk(k, ascending=True)
            return ids, distance
        else:
            distances = simsimd.cdist(query_vec, db_vecs, metric="cosine")
            topk = FastSort(distances.squeeze())
            ids, distance = topk.topk(k, ascending=True)
            return ids, distance

    def benchmark_simsimd_l2sq(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """SimSIMD L2 squared distance implementation"""
        if query_vec.shape[0] == 1:
            distances = simsimd.sqeuclidean(query_vec.squeeze(), db_vecs)
            topk = FastSort(distances)
            ids, distance = topk.topk(k, ascending=True)
            return ids, distance
        else:
            distances = simsimd.cdist(query_vec, db_vecs, metric="sqeuclidean")
            topk = FastSort(distances.squeeze())
            ids, distance = topk.topk(k, ascending=True)
            return ids, distance

    def benchmark_simsimd_inner(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """SimSIMD inner product implementation"""
        if query_vec.shape[0] == 1:
            distances = simsimd.inner(query_vec.squeeze(), db_vecs)
            topk = FastSort(-distances)  # Negative for descending order
            ids, distance = topk.topk(k, ascending=False)
            return ids, -distance
        else:
            distances = simsimd.cdist(query_vec, db_vecs, metric="inner")
            topk = FastSort(-distances.squeeze())
            ids, distance = topk.topk(k, ascending=False)
            return ids, -distance

    def benchmark_numpy_cosine(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy cosine distance implementation"""
        # Cosine distance = 1 - cosine similarity
        cosine_sim = np.dot(query_vec, db_vecs.T) / (np.linalg.norm(query_vec, axis=1, keepdims=True) * np.linalg.norm(db_vecs, axis=1))
        distances = 1 - cosine_sim.squeeze()
        topk = FastSort(distances)
        ids, distance = topk.topk(k, ascending=True)
        return ids, distance

    def benchmark_numpy_l2sq(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy L2 squared distance implementation"""
        diff = db_vecs - query_vec
        distances = np.sum(diff * diff, axis=1)
        topk = FastSort(distances)
        ids, distance = topk.topk(k, ascending=True)
        return ids, distance

    def benchmark_numpy_inner(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy inner product implementation"""
        distances = np.dot(query_vec, db_vecs.T).squeeze()
        topk = FastSort(-distances)  # Negative for descending order
        ids, distance = topk.topk(k, ascending=False)
        return ids, -distance

    def benchmark_usearch_cosine(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """USearch cosine distance implementation"""
        ids, distances, _ = exact_search(db_vecs, query_vec, k, metric_kind=MetricKind.Cos)
        return ids, distances

    def benchmark_usearch_l2sq(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """USearch L2 squared distance implementation"""
        ids, distances, _ = exact_search(db_vecs, query_vec, k, metric_kind=MetricKind.L2sq)
        return ids, distances

    def benchmark_usearch_inner(self, query_vec: np.ndarray, db_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """USearch inner product implementation"""
        ids, distances, _ = exact_search(db_vecs, query_vec, k, metric_kind=MetricKind.IP)
        distances = -distances  # USearch returns negative inner product
        return ids, distances

    def run_single_benchmark(self, func: Callable, query_vec: np.ndarray, db_vecs: np.ndarray,
                           k: int, runs: int = 10) -> Tuple[float, float, bool, str]:
        """Run a single benchmark function multiple times and return statistics"""
        times = []
        error_msg = ""

        for _ in range(runs):
            try:
                start_time = time.perf_counter()
                func(query_vec, db_vecs, k)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                error_msg = str(e)
                return 0.0, 0.0, False, error_msg

        if times:
            return np.mean(times), np.std(times), True, ""
        else:
            return 0.0, 0.0, False, error_msg

    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks across different configurations"""
        logger.info("🚀 Starting comprehensive benchmark...")

        # Test configurations
        test_configs = [
            # (distance_metric, data_types, vector_sizes)
            ('cosine', [np.float32, np.float16, np.int8], [100, 500, 1000, 2000]),
            ('l2sq', [np.float32, np.float16, np.int8], [100, 500, 1000, 2000]),
            ('inner', [np.float32, np.float16, np.int8], [100, 500, 1000, 2000]),
        ]

        dimensions = 512  # Fixed dimension for consistency
        k = 10
        runs = 5  # Number of runs per test

        for distance_metric, data_types, vector_sizes in test_configs:
            logger.info(f"📊 Testing {distance_metric} distance...")

            for dtype in data_types:
                logger.info(f"  📈 Data type: {dtype}")

                for n_vectors in vector_sizes:
                    logger.info(f"    🔍 Vector count: {n_vectors}")

                    # Generate test data
                    query_vec, db_vecs = self.generate_test_data(n_vectors, dimensions, dtype)

                    # Define benchmark functions for this distance metric
                    if distance_metric == 'cosine':
                        methods = {
                            'simsimd': self.benchmark_simsimd_cosine,
                            'numpy': self.benchmark_numpy_cosine,
                            'usearch': self.benchmark_usearch_cosine,
                        }
                    elif distance_metric == 'l2sq':
                        methods = {
                            'simsimd': self.benchmark_simsimd_l2sq,
                            'numpy': self.benchmark_numpy_l2sq,
                            'usearch': self.benchmark_usearch_l2sq,
                        }
                    elif distance_metric == 'inner':
                        methods = {
                            'simsimd': self.benchmark_simsimd_inner,
                            'numpy': self.benchmark_numpy_inner,
                            'usearch': self.benchmark_usearch_inner,
                        }

                    # Benchmark each method
                    for method_name, method_func in methods.items():
                        mean_time, std_time, success, error = self.run_single_benchmark(
                            method_func, query_vec, db_vecs, k, runs
                        )

                        result = BenchmarkResult(
                            method=method_name,
                            distance_metric=distance_metric,
                            data_type=str(dtype),
                            vector_size=n_vectors,
                            time_mean=mean_time,
                            time_std=std_time,
                            success=success,
                            error=error
                        )

                        self.results.append(result)

                        if success:
                            logger.info(f"      ✅ {method_name}: {mean_time:.6f}s ± {std_time:.6f}s")
                        else:
                            logger.info(f"      ❌ {method_name}: Failed - {error}")

    def analyze_results(self):
        """Analyze benchmark results and determine optimal choices"""
        logger.info("🔍 Analyzing benchmark results...")

        # Group results by distance metric and data type
        grouped_results = {}
        for result in self.results:
            if not result.success:
                continue

            key = (result.distance_metric, result.data_type)
            if key not in grouped_results:
                grouped_results[key] = {}

            if result.method not in grouped_results[key]:
                grouped_results[key][result.method] = []

            grouped_results[key][result.method].append(result.time_mean)

        # Find optimal choice for each combination
        self.optimal_choices = {}
        for (distance_metric, data_type), methods in grouped_results.items():
            # Calculate average time for each method
            avg_times = {}
            for method, times in methods.items():
                avg_times[method] = np.mean(times)

            # Find the fastest method
            if avg_times:
                fastest_method = min(avg_times.keys(), key=lambda m: avg_times[m])

                if distance_metric not in self.optimal_choices:
                    self.optimal_choices[distance_metric] = {}

                self.optimal_choices[distance_metric][data_type] = {
                    'fastest_method': fastest_method,
                    'avg_time': avg_times[fastest_method],
                    'all_methods': avg_times
                }

                logger.info(f"📈 {distance_metric} + {data_type}: {fastest_method} ({avg_times[fastest_method]:.6f}s)")

    def save_results(self, filename: str = "comprehensive_benchmark_results.json"):
        """Save benchmark results to JSON file"""
        results_data = {
            'results': [
                {
                    'method': r.method,
                    'distance_metric': r.distance_metric,
                    'data_type': r.data_type,
                    'vector_size': r.vector_size,
                    'time_mean': r.time_mean,
                    'time_std': r.time_std,
                    'success': r.success,
                    'error': r.error
                }
                for r in self.results
            ],
            'optimal_choices': self.optimal_choices
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"💾 Results saved to {filename}")

    def generate_optimal_engine_config(self):
        """Generate optimal engine configuration based on benchmark results"""
        logger.info("⚙️ Generating optimal engine configuration...")

        # Count votes for each method across all distance metrics and data types
        method_scores = {'simsimd': 0, 'numpy': 0, 'usearch': 0}

        for distance_metric, data_types in self.optimal_choices.items():
            for data_type, choice_info in data_types.items():
                fastest_method = choice_info['fastest_method']
                method_scores[fastest_method] += 1

        # Determine global winner
        global_winner = max(method_scores.keys(), key=lambda m: method_scores[m])

        logger.info("🏆 Benchmark Summary:")
        logger.info(f"  📊 Method scores: {method_scores}")
        logger.info(f"  🥇 Global winner: {global_winner}")

        return global_winner, method_scores

def main():
    """Main function to run comprehensive benchmark"""
    logger.info("🧪 Starting comprehensive SimSIMD vs NumPy vs USearch benchmark...")

    benchmark = ComprehensiveBenchmark()

    # Run benchmarks
    benchmark.run_comprehensive_benchmark()

    # Analyze results
    benchmark.analyze_results()

    # Save results
    benchmark.save_results()

    # Generate optimal configuration
    global_winner, method_scores = benchmark.generate_optimal_engine_config()

    logger.info("✅ Comprehensive benchmark completed!")
    logger.info(f"🎯 Recommended default engine: {global_winner}")

    return global_winner, benchmark.optimal_choices

if __name__ == "__main__":
    main()
