import os
import time
import psutil
from typing import List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import struct
import lz4.frame  # 用于快速压缩


class FSST:
    def __init__(self, thread_count: int = 1):
        self.thread_count = thread_count

    def _analyze_data(self, data: bytes, sample_size: int = 1024) -> str:
        """分析数据类型"""
        if not data:
            return 'empty'

        # 取样本进行分析
        sample = data[:min(sample_size, len(data))]
        sample_array = np.frombuffer(sample, dtype=np.uint8)

        # 计算基本统计信息
        unique_ratio = len(set(sample)) / len(sample)
        zero_ratio = np.count_nonzero(sample_array == 0) / len(sample)
        digit_ratio = len([x for x in sample if x >= 48 and x <= 57]) / len(sample)

        # 数据类型判断
        if unique_ratio > 0.9:  # 高度随机数据
            return 'random'
        elif digit_ratio > 0.5:  # 数字序列
            return 'numeric'
        elif unique_ratio < 0.1:  # 高度重复数据
            return 'repetitive'
        else:  # 普通数据
            return 'normal'

    def compress(self, data: bytes) -> bytes:
        """智能压缩"""
        if not data:
            return b''

        if len(data) < 1024:  # 小数据块直接返回
            return b'\x00' + data

        # 分析数据类型
        data_type = self._analyze_data(data)

        if data_type == 'random':
            # 随机数据几乎不可压缩，直接返回
            return b'\x00' + data

        elif data_type == 'numeric':
            # 数字序列使用特殊的数字压缩
            if self.thread_count == 1:
                return b'\x02' + self._compress_numeric(data)
            return self._parallel_compress_numeric(data)

        else:
            # 其他类型使用LZ4
            if self.thread_count == 1:
                return b'\x01' + lz4.frame.compress(data)
            return self._parallel_compress_lz4(data)

    def _compress_numeric(self, data: bytes) -> bytes:
        """针对数字序列的压缩"""
        if not data:
            return b''

        # 添加原始长度信息
        compressed = bytearray(struct.pack('<I', len(data)))

        # 按4字节处理数字
        i = 0
        prev_num = 0
        while i < len(data):
            chunk = data[i:i+4]
            if len(chunk) == 4 and chunk.isdigit():  # 是数字字符
                try:
                    num = int(chunk)
                    diff = num - prev_num

                    # 使用变长编码存储差值
                    if -64 <= diff <= 63:  # 1字节范围
                        compressed.append((diff & 0x7F) | 0x80)
                    elif -8192 <= diff <= 8191:  # 2字节范围
                        compressed.append((diff >> 8) & 0x3F)
                        compressed.append(diff & 0xFF)
                    else:  # 4字节完整存储
                        compressed.append(0x40)  # 标记为4字节存储
                        compressed.extend(struct.pack('<i', diff))

                    prev_num = num

                except ValueError:
                    compressed.append(0xFF)  # 非数字标记
                    compressed.extend(chunk)
            else:
                compressed.append(0xFF)  # 非数字标记
                compressed.extend(chunk)
            i += 4

        return bytes(compressed)

    def _parallel_compress_numeric(self, data: bytes) -> bytes:
        """并行数字压缩"""
        if len(data) < 1024 * self.thread_count:
            # 数据太小，使用单线程压缩
            return b'\x02' + self._compress_numeric(data)

        # 确保每个块都是完整的数字序列
        chunk_size = (len(data) // (self.thread_count * 4)) * 4
        chunks = []

        # 分割数据
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i+chunk_size])

        # 并行压缩
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            compressed_chunks = list(executor.map(self._compress_numeric, chunks))

        # 构建结果
        header = struct.pack('<I', len(compressed_chunks))
        sizes = struct.pack('<' + 'I' * len(compressed_chunks),
                          *[len(chunk) for chunk in compressed_chunks])

        return b'\x02' + header + sizes + b''.join(compressed_chunks)

    def _parallel_compress_lz4(self, data: bytes) -> bytes:
        """并行LZ4压缩"""
        chunk_size = len(data) // self.thread_count

        # 使用numpy快速分割数据
        data_array = np.frombuffer(data, dtype=np.uint8)
        chunks = np.array_split(data_array, self.thread_count)
        chunks = [chunk.tobytes() for chunk in chunks]

        # 并行压缩
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            compressed_chunks = list(executor.map(lz4.frame.compress, chunks))

        # 构建结果
        header = struct.pack('<I', len(compressed_chunks))
        sizes = struct.pack('<' + 'I' * len(compressed_chunks),
                          *[len(chunk) for chunk in compressed_chunks])

        # 添加压缩方法标识符
        return b'\x01' + header + sizes + b''.join(compressed_chunks)

    def decompress(self, data: bytes) -> bytes:
        """智能解压"""
        if not data:
            return b''

        # 读取压缩方法
        method = data[0]
        data = data[1:]  # 移除方法标识符

        if method == 0:  # 未压缩
            return data
        elif method == 1:  # LZ4
            if self.thread_count == 1:
                return lz4.frame.decompress(data)
            try:
                return self._parallel_decompress_lz4(data)
            except Exception:
                # 如果并行解压失败，回退到单线程
                return lz4.frame.decompress(data)
        elif method == 2:  # 数字压缩
            if self.thread_count == 1:
                return self._decompress_numeric(data)
            try:
                return self._parallel_decompress_numeric(data)
            except Exception:
                # 如果并行解压失败，回退到单线程
                return self._decompress_numeric(data)
        else:
            raise ValueError(f"Unknown compression method: {method}")

    def _decompress_numeric(self, data: bytes) -> bytes:
        """数字序列解压"""
        if len(data) < 4:
            return data

        # 读取原始长度
        original_length = struct.unpack('<I', data[:4])[0]
        pos = 4
        result = bytearray()
        prev_num = 0

        while pos < len(data):
            if data[pos] == 0xFF:  # 非数字数据
                pos += 1
                chunk_len = min(4, len(data) - pos)
                chunk = data[pos:pos+chunk_len]
                result.extend(chunk)
                pos += chunk_len
                continue

            # 解析差值
            marker = data[pos]
            pos += 1

            if marker & 0x80:  # 1字节差值
                diff = ((marker & 0x7F) ^ 0x40) - 0x40  # 有符号转换
            elif marker == 0x40:  # 4字节完整存储
                if pos + 4 > len(data):
                    break
                diff = struct.unpack('<i', data[pos:pos+4])[0]
                pos += 4
            else:  # 2字节差值
                if pos + 1 > len(data):
                    break
                diff = ((marker & 0x3F) << 8) | data[pos]
                if diff & 0x2000:  # 负数处理
                    diff |= -0x4000
                pos += 1

            # 计算实际数字并格式化
            num = prev_num + diff
            result.extend(str(num).zfill(4).encode('ascii'))
            prev_num = num

        # 截断到原始长度
        return bytes(result[:original_length])

    def _parallel_decompress_numeric(self, data: bytes) -> bytes:
        """并行数字解压"""
        try:
            # 读取头部信息
            chunk_count = struct.unpack('<I', data[:4])[0]
            pos = 4

            # 安全检查
            if chunk_count > 100 or chunk_count < 1:
                return self._decompress_numeric(data)

            # 读取块大小
            size_data_len = 4 * chunk_count
            if pos + size_data_len > len(data):
                return self._decompress_numeric(data)

            chunk_sizes = struct.unpack('<' + 'I' * chunk_count,
                                      data[pos:pos + size_data_len])
            pos += size_data_len

            # 提取压缩块
            chunks = []
            for size in chunk_sizes:
                if pos + size > len(data):
                    return self._decompress_numeric(data)
                chunks.append(data[pos:pos + size])
                pos += size

            # 并行解压
            with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                decompressed_chunks = list(executor.map(self._decompress_numeric, chunks))

            return b''.join(decompressed_chunks)

        except Exception:
            # 如果出现任何错误，回退到单线程解压
            return self._decompress_numeric(data)

    def _parallel_decompress_lz4(self, data: bytes) -> bytes:
        """并行LZ4解压"""
        # 读取头部信息
        chunk_count = struct.unpack('<I', data[:4])[0]
        pos = 4

        # 读取块大小
        chunk_sizes = struct.unpack('<' + 'I' * chunk_count,
                                  data[pos:pos + 4 * chunk_count])
        pos += 4 * chunk_count

        # 提取压缩块
        chunks = []
        for size in chunk_sizes:
            chunks.append(data[pos:pos + size])
            pos += size

        # 并行解压
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            decompressed_chunks = list(executor.map(lz4.frame.decompress, chunks))

        return b''.join(decompressed_chunks)

# 简化的接口
def compress(data: bytes, thread_count: int = 4) -> bytes:
    compressor = FSST(thread_count=thread_count)
    return compressor.compress(data)

def decompress(data: bytes, thread_count: int = 4) -> bytes:
    compressor = FSST(thread_count=thread_count)
    return compressor.decompress(data)

def get_memory_usage() -> float:
    """获取当前进程的内存使用量（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_benchmark(data: bytes, name: str, thread_count: int = 4):
    """运行性能基准测试"""
    print(f"\n测试 {name}:")
    print(f"原始大小: {len(data):,} 字节")

    # 记录初始内存
    initial_memory = get_memory_usage()

    # 压缩测试
    start_time = time.time()
    compressed = compress(data, thread_count)
    compress_time = time.time() - start_time
    compress_memory = get_memory_usage() - initial_memory

    print(f"压缩后大小: {len(compressed):,} 字节")
    print(f"压缩率: {len(compressed) / len(data):.2%}")
    print(f"压缩时间: {compress_time:.3f} 秒")
    print(f"压缩内存: {compress_memory:.2f} MB")

    # 重置内存基准
    initial_memory = get_memory_usage()

    # 解压测试
    start_time = time.time()
    decompressed = decompress(compressed, thread_count)
    decompress_time = time.time() - start_time
    decompress_memory = get_memory_usage() - initial_memory

    print(f"解压时间: {decompress_time:.3f} 秒")
    print(f"解压内存: {decompress_memory:.2f} MB")
    print(f"数据完整性: {decompressed == data}")

    return {
        'name': name,
        'original_size': len(data),
        'compressed_size': len(compressed),
        'compression_ratio': len(compressed) / len(data),
        'compress_time': compress_time,
        'decompress_time': decompress_time,
        'compress_memory': compress_memory,
        'decompress_memory': decompress_memory,
        'is_valid': decompressed == data
    }

def run_comprehensive_benchmark(thread_counts: List[int] = [1, 2, 4, 8]):
    """运行综合性能测试"""
    # 准备测试数据
    test_cases = [
        (b"Hello" * 100000, "重复文本"),
        (os.urandom(100000), "随机二进制"),
        (b"".join([str(i).encode() for i in range(10000)]), "数字序列"),
        (b"abcdefghijklmnopqrstuvwxyz" * 10000, "重复字母"),
        (open(__file__, 'rb').read() * 100, "Python源代码")
    ]

    all_results = []

    # 对每个线程数进行测试
    for thread_count in thread_counts:
        print(f"\n使用 {thread_count} 个线程进行测试")
        print("=" * 80)

        results = []
        for data, name in test_cases:
            result = run_benchmark(data, name, thread_count)
            results.append(result)

        all_results.extend(results)

        # 打印当前线程数的汇总报告
        print(f"\n{thread_count} 线程测试汇总:")
        print("-" * 80)
        print(f"{'测试类型':<15} {'原始大小':>10} {'压缩大小':>10} {'压缩率':>8} "
              f"{'压缩时间':>8} {'解压时间':>8} {'压缩内存':>8} {'解压内存':>8}")
        print("-" * 80)

        for result in results:
            print(f"{result['name']:<15} {result['original_size']:>10,} "
                  f"{result['compressed_size']:>10,} {result['compression_ratio']:>7.2%} "
                  f"{result['compress_time']:>7.3f}s {result['decompress_time']:>7.3f}s "
                  f"{result['compress_memory']:>7.1f}M {result['decompress_memory']:>7.1f}M")

    return all_results

if __name__ == "__main__":
    # 运行综合基准测试
    results = run_comprehensive_benchmark([1, 2, 4, 8])
