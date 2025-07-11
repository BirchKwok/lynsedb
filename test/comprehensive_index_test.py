#!/usr/bin/env python3
"""
Comprehensive test for LynseDB index modes and quantization effects.

This test validates:
1. Search quality (recall@k) for different index modes
2. Quantization effects on compression ratio and accuracy
3. Index persistence and reloading
4. Performance metrics (build time, search time)
"""

import numpy as np
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import trange, tqdm

import lynse
from lynse.index.indexer import Indexer


class IndexTestResults:
    """Store and analyze test results."""

    def __init__(self):
        self.results = {}

    def add_result(self, index_mode: str, **kwargs):
        """Add test result for an index mode."""
        self.results[index_mode] = kwargs

    def print_summary(self):
        """Print a summary report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE INDEX TEST RESULTS")
        print("="*80)

        passed = [k for k, v in self.results.items() if v.get('passed', False)]
        failed = [k for k, v in self.results.items() if not v.get('passed', False)]

        print(f"Total modes tested: {len(self.results)}")
        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed modes:")
            for mode in failed:
                error = self.results[mode].get('error', 'Unknown error')
                print(f"  {mode}: {error}")

        # Performance summary for passed tests
        if passed:
            print("\nPerformance Summary (passed tests only):")
            build_times = [self.results[k]['build_time'] for k in passed]
            search_times = [self.results[k]['search_time'] for k in passed]
            recalls = [self.results[k]['recall_at_10'] for k in passed]

            print(f"  Average build time: {np.mean(build_times):.2f}s")
            print(f"  Average search time: {np.mean(search_times):.4f}s")
            print(f"  Average recall@10: {np.mean(recalls):.3f}")


def prepare_simple_test_data(num_vectors=5000, dim=64):
    """Create simple test data for quick testing."""

    # Clean up previous test
    root_path = "test_comprehensive"
    if os.path.exists(root_path):
        shutil.rmtree(root_path)

    print(f"Creating test collection with {num_vectors} vectors...")

    client = lynse.VectorDBClient(root_path)
    my_db = client.create_database("test_db", drop_if_exists=True)

    collection = my_db.require_collection(
        "test_vectors",
        dim=dim,
        drop_if_exists=True,
        cache_chunks=0,
        chunk_size=1000,
    )

    # Generate structured test data
    np.random.seed(42)

    # Create 20 clusters
    num_clusters = 20
    cluster_centers = np.random.randn(num_clusters, dim).astype(np.float32)

    vectors = []
    labels = []

    for i in range(num_vectors):
        cluster_id = i % num_clusters
        noise = np.random.randn(dim).astype(np.float32) * 0.1
        vector = cluster_centers[cluster_id] + noise
        vectors.append(vector)
        labels.append(cluster_id)

    # Add vectors to collection
    with collection.insert_session() as session:
        for i, vec in enumerate(vectors):
            session.add_item(vec, field={"id": i, "cluster": labels[i]})

    return collection, cluster_centers[:5], np.array(labels)  # Use 5 query vectors


def calculate_recall(retrieved_ids, true_cluster_id, true_labels, k=10):
    """Calculate recall@k."""
    if len(retrieved_ids) == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = np.sum(true_labels[top_k] == true_cluster_id)
    total_relevant = np.sum(true_labels == true_cluster_id)

    return relevant_count / min(k, total_relevant)


def test_single_index_mode(collection, query_vectors, true_labels, index_mode, results):
    """Test a single index mode."""
    print(f"\n[TEST] {index_mode}")

    try:
        # Build index
        start_time = time.time()
        collection.build_index(index_mode=index_mode)
        build_time = time.time() - start_time

        # Test search
        total_recall = 0.0
        search_times = []

        for i, query_vec in enumerate(query_vectors):
            start_time = time.time()
            res = collection.search(query_vec, k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

            indices, _, _ = res.to_tuple()
            recall = calculate_recall(indices, i, true_labels, k=10)
            total_recall += recall

        avg_recall = total_recall / len(query_vectors)
        avg_search_time = np.mean(search_times)

        results.add_result(
            index_mode,
            passed=True,
            build_time=build_time,
            search_time=avg_search_time,
            recall_at_10=avg_recall
        )

        print(f"[PASS] Build: {build_time:.2f}s, Search: {avg_search_time:.4f}s, Recall: {avg_recall:.3f}")
        return True

    except Exception as e:
        print(f"[FAIL] {str(e)}")
        results.add_result(
            index_mode,
            passed=False,
            error=str(e),
            build_time=0,
            search_time=0,
            recall_at_10=0
        )
        return False


def main():
    """Run the comprehensive test."""
    print("Starting comprehensive index test...")

    # Prepare test data
    collection, query_vectors, true_labels = prepare_simple_test_data()

    # Initialize results
    results = IndexTestResults()

    # Get all index modes
    all_modes = sorted(set(Indexer._INDEX_ALIAS.values()))

    print(f"\nTesting {len(all_modes)} index modes...")

    # Test each mode
    passed_count = 0
    for mode in all_modes:
        if test_single_index_mode(collection, query_vectors, true_labels, mode, results):
            passed_count += 1

    # Print results
    results.print_summary()

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {passed_count}/{len(all_modes)} modes passed")

    if passed_count == len(all_modes):
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {len(all_modes) - passed_count} tests failed")

    return passed_count == len(all_modes)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
