from bitarray import bitarray
import mmh3
import struct


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


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


class BloomTrie:
    def __init__(self, bloom_size, bloom_hash_count):
        self.bloom_filter = BloomFilter(bloom_size, bloom_hash_count)
        self.trie = Trie()

    def add(self, item):
        str_item = str(item)
        self.bloom_filter.add(str_item)
        self.trie.insert(str_item)

    def __contains__(self, item):
        str_item = str(item)
        if str_item in self.bloom_filter:
            return self.trie.search(str_item)
        return False

    def _save_trie_node(self, node, f):
        f.write(struct.pack('B', len(node.children)))
        f.write(struct.pack('?', node.is_end_of_word))
        for char, child in node.children.items():
            f.write(struct.pack('c', char.encode('utf-8')))
            self._save_trie_node(child, f)

    def to_file(self, path):
        path = str(path)
        self.bloom_filter.to_file(path + '.bloom.mvdb')
        with open(path + '.trie.mvdb', 'wb') as f:
            self._save_trie_node(self.trie.root, f)

    def _load_trie_node(self, f):
        node = TrieNode()
        children_count = struct.unpack('B', f.read(1))[0]
        node.is_end_of_word = struct.unpack('?', f.read(1))[0]
        for _ in range(children_count):
            char = struct.unpack('c', f.read(1))[0].decode('utf-8')
            child = self._load_trie_node(f)
            node.children[char] = child
        return node

    def from_file(self, path):
        path = str(path)
        self.bloom_filter.from_file(path + '.bloom.mvdb')
        with open(path + '.trie.mvdb', 'rb') as f:
            self.trie.root = self._load_trie_node(f)
