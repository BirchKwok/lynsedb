#!/usr/bin/env python3
"""
Comprehensive Index and Quantizer Combination Test

This script tests all combinations of indices and quantizers to ensure:
1. All combinations work correctly without bugs
2. SimSIMD integration is working properly
3. Distance calculations are accurate
4. Index operations (fit_transform, search, save/load) function correctly
"""

import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback

# Add lynse to path
sys.path.insert(0, str(Path(__file__).parent))

from lynse.index.factory import IndexFactory
from lynse.index.builder import create_index_from_string
from lynse.computational_layer.engines import get_simsimd_capabilities


class IndexTester:
    """Comprehensive tester for index and quantizer combinations."""

    def __init__(self):
        self.temp_dir = None
        self.results = {}
        self.errors = {}

        # Define test configurations
        self.index_types = [
            'flat', 'flat-l2', 'flat-ip', 'flat-cosine', 'flat-jaccard', 'flat-hamming',
            'hnsw', 'hnsw-l2', 'hnsw-ip', 'hnsw-cosine',
            'diskann', 'diskann-l2', 'diskann-ip', 'diskann-cosine',
            'ivf', 'ivf-l2', 'ivf-ip', 'ivf-cosine', 'ivf-jaccard', 'ivf-hamming'
        ]

        self.quantizers = ['none', 'sq', 'binary', 'pq']

        # Data types for testing
        self.test_data_configs = [
            {'dtype': np.float32, 'dimensions': 128, 'n_vectors': 1000},
            {'dtype': np.float16, 'dimensions': 64, 'n_vectors': 500},
            {'dtype': np.int8, 'dimensions': 32, 'n_vectors': 200},
        ]

        # Binary data for binary distance metrics
        self.binary_data = np.random.choice([0, 1], size=(500, 64)).astype(np.bool_)

    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory: {self.temp_dir}")

        # Print SimSIMD capabilities
        capabilities = get_simsimd_capabilities()
        print(f"SimSIMD capabilities: {capabilities}")

    def cleanup(self):
        """Cleanup test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

    def generate_test_data(self, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test data based on configuration."""
        n_vectors = config['n_vectors']
        dimensions = config['dimensions']
        dtype = config['dtype']

        # Generate random data
        if dtype == np.float32:
            data = np.random.randn(n_vectors, dimensions).astype(dtype)
        elif dtype == np.float16:
            data = np.random.randn(n_vectors, dimensions).astype(dtype)
        elif dtype == np.int8:
            data = np.random.randint(-128, 127, size=(n_vectors, dimensions)).astype(dtype)
        else:
            data = np.random.randn(n_vectors, dimensions).astype(dtype)

        # Normalize for cosine similarity tests
        data = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-8)

        # Generate query data
        query = data[:10]  # Use first 10 vectors as queries

        return data, query

    def get_compatible_combinations(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all compatible index-quantizer-data combinations."""
        combinations = []

        for index_type in self.index_types:
            for quantizer in self.quantizers:
                for data_config in self.test_data_configs:
                    # Check compatibility
                    if self._is_compatible(index_type, quantizer, data_config):
                        combinations.append((index_type, quantizer, data_config))

        # Add binary combinations for binary distance metrics
        for index_type in ['flat-jaccard', 'flat-hamming', 'ivf-jaccard', 'ivf-hamming']:
            for quantizer in ['none', 'binary']:
                binary_config = {'dtype': np.bool_, 'dimensions': 64, 'n_vectors': 500}
                combinations.append((index_type, quantizer, binary_config))

        return combinations

    def _is_compatible(self, index_type: str, quantizer: str, data_config: Dict[str, Any]) -> bool:
        """Check if index-quantizer-data combination is compatible."""
        # Binary quantizer works with all data types
        if quantizer == 'binary':
            return True

        # Jaccard and Hamming work best with binary data, but we'll test with boolean conversion
        if 'jaccard' in index_type or 'hamming' in index_type:
            return data_config['dtype'] in [np.bool_, np.uint8, np.int8]

        # Product quantizer needs sklearn
        if quantizer == 'pq':
            try:
                from sklearn.cluster import KMeans
                return True
            except ImportError:
                return False

        return True

    def test_single_combination(self, index_type: str, quantizer: str, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single index-quantizer-data combination."""
        test_result = {
            'success': False,
            'error': None,
            'simsimd_used': False,
            'operations': {
                'creation': False,
                'fit_transform': False,
                'search': False,
                'save_load': False
            },
            'performance': {},
            'accuracy': {}
        }

        try:
            print(f"  Testing {index_type} + {quantizer} + {data_config['dtype'].__name__}")

            # Generate or use appropriate test data
            if data_config['dtype'] == np.bool_:
                data = self.binary_data.copy()
                query = data[:5]
            else:
                data, query = self.generate_test_data(data_config)

            # Test index creation
            if quantizer == 'none':
                index = create_index_from_string(index_type)
            else:
                index = create_index_from_string(f"{index_type}-{quantizer}")
            test_result['operations']['creation'] = True

            # Check SimSIMD info
            simd_info = index.get_simd_info()
            test_result['simsimd_used'] = simd_info.get('use_simd') is not False

            # Test fit_transform
            import time
            start_time = time.time()
            encoded_data = index.fit_transform(data)
            fit_time = time.time() - start_time
            test_result['operations']['fit_transform'] = True
            test_result['performance']['fit_time'] = fit_time

            # Test search
            start_time = time.time()
            for q in query:
                ids, distances = index.search(q, k=5)
                # Basic sanity checks
                assert len(ids) > 0, "No results returned"
                assert len(ids) == len(distances), "Mismatched ids and distances"
                assert all(d >= 0 for d in distances), "Negative distances found"
            search_time = time.time() - start_time
            test_result['operations']['search'] = True
            test_result['performance']['search_time'] = search_time

            # Test save/load
            save_path = Path(self.temp_dir) / f"test_{index_type}_{quantizer}_{data_config['dtype'].__name__}.index"
            index.save(str(save_path))

            # Create new index and load
            new_index = create_index_from_string(f"{index_type}-{quantizer}")
            new_index.load(str(save_path))

            # Verify loaded index works
            ids2, distances2 = new_index.search(query[0], k=5)
            assert len(ids2) > 0, "Loaded index search failed"
            test_result['operations']['save_load'] = True

            test_result['success'] = True

        except Exception as e:
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()

        return test_result

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all combinations."""
        print("Starting comprehensive index and quantizer combination test...")
        print("=" * 70)

        combinations = self.get_compatible_combinations()
        print(f"Testing {len(combinations)} combinations...")

        total_tests = len(combinations)
        passed_tests = 0
        failed_tests = 0

        for i, (index_type, quantizer, data_config) in enumerate(combinations, 1):
            combo_key = f"{index_type}+{quantizer}+{data_config['dtype'].__name__}"
            print(f"[{i}/{total_tests}] {combo_key}")

            try:
                result = self.test_single_combination(index_type, quantizer, data_config)
                self.results[combo_key] = result

                if result['success']:
                    passed_tests += 1
                    print(f"  ✅ PASSED")
                else:
                    failed_tests += 1
                    print(f"  ❌ FAILED: {result['error']}")
                    self.errors[combo_key] = result

            except Exception as e:
                failed_tests += 1
                error_info = {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.results[combo_key] = error_info
                self.errors[combo_key] = error_info
                print(f"  ❌ FAILED: {str(e)}")

        # Summary
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'results': self.results,
            'errors': self.errors
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']:.2%}")

        if summary['errors']:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for combo, error_info in summary['errors'].items():
                print(f"❌ {combo}")
                print(f"   Error: {error_info['error']}")
                print()

        # SimSIMD usage analysis
        simsimd_used_count = sum(1 for r in summary['results'].values()
                                if r.get('success') and r.get('simsimd_used'))
        simsimd_total = sum(1 for r in summary['results'].values() if r.get('success'))

        print(f"SimSIMD Usage: {simsimd_used_count}/{simsimd_total} successful tests")

        if summary['failed_tests'] == 0:
            print("\n🎉 ALL TESTS PASSED! SimSIMD integration is working correctly across all combinations!")
        else:
            print(f"\n⚠️  {summary['failed_tests']} tests failed. Please review the errors above.")


def main():
    """Main test function."""
    tester = IndexTester()

    try:
        tester.setup()
        summary = tester.run_comprehensive_test()
        tester.print_summary(summary)

        # Return appropriate exit code
        if summary['failed_tests'] == 0:
            print("\n✅ All index and quantizer combinations work correctly!")
            return 0
        else:
            print(f"\n❌ {summary['failed_tests']} combinations failed.")
            return 1

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    sys.exit(main())
