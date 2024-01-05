from bitarray import bitarray
import mmh3
import numpy as np


class CountingBloomFilter:
    def __init__(self, size, hash_count, dtype=np.int32):
        self.size = size
        self.hash_count = hash_count
        self.count_array = np.zeros(size, dtype=dtype)
        self.hash_functions = [self._generate_hash_function(i) for i in range(hash_count)]

    def _generate_hash_function(self, seed):
        def hash_function(item):
            return (hash(item) + seed) % self.size
        return hash_function

    def add(self, item):
        for hash_function in self.hash_functions:
            index = hash_function(item)
            self.count_array[index] += 1

    def remove(self, item):
        for hash_function in self.hash_functions:
            index = hash_function(item)
            if self.count_array[index] > 0:
                self.count_array[index] -= 1

    def __contains__(self, item):
        return all(self.count_array[hash_function(item)] > 0 for hash_function in self.hash_functions)


class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def add(self, item):
        item = str(item)
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            self.bit_array[index] = 1

    def __contains__(self, item):
        item = str(item)
        for i in range(self.hash_count):
            index = mmh3.hash(item, i) % self.size
            if self.bit_array[index] == 0:
                return False
        return True

    def to_file(self, path):
        with open(path, 'wb') as f:
            self.bit_array.tofile(f)

    def from_file(self, path):
        self.bit_array = bitarray()
        with open(path, 'rb') as f:
            self.bit_array.fromfile(f)

        if len(self.bit_array) != self.size:
            raise ValueError('The size of the bit array is not equal to the size of the Bloom filter, '
                             f'excepted {self.size} but got {len(self.bit_array)}. '
                             f'Perhaps you want to delete the file {path},  to recreate a new Bloom filter?')
