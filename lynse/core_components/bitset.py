import pickle
from bitarray import bitarray


class BitSet:
    __slots__ = ['_bits']

    def __init__(self, size=0, initial_bits=None, fill=0):
        """
        Initialize a new BitSet.

        Parameters:
            size(int): The initial size of the BitSet (in bits).
            initial_bits(bitarray or iterable): An optional bitarray or iterable to initialize the BitSet with.
            fill(int): The value to fill the bits with if no initial_bits is provided (0 or 1).
        """
        if initial_bits is not None:
            if isinstance(initial_bits, bitarray):
                self._bits = initial_bits.copy()
            else:
                self._bits = bitarray(initial_bits)
        else:
            self._bits = bitarray(size)
            self._bits.setall(fill)

    def __len__(self):
        """
        Return the size of the BitSet.

        Returns:
            int: The size of the BitSet (in bits).
        """
        return len(self._bits)

    def _check_index(self, index):
        """
        Check if the given index is within the bounds of the BitSet.

        Parameters:
            index(int): The index to check.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if index < 0 or index >= len(self._bits):
            raise IndexError("Bit index out of range")

    def resize(self, new_size, fill=0):
        """
        Resize the BitSet to a new size.

        Parameters:
            new_size(int): The new size of the BitSet (in bits).
            fill(int): The value to fill the new bits with.
        """
        current_size = len(self._bits)
        if new_size < current_size:
            self._bits = self._bits[:new_size]
        elif new_size > current_size:
            extension_size = new_size - current_size
            self._bits.extend([fill] * extension_size)

    def set_bit(self, index):
        """
        Set the bit at the given index to 1.

        Parameters:
            index(int): The index of the bit to set.
        """
        if index >= len(self._bits):
            self.resize(index + 1)
        self._bits[index] = 1

    def clear_bit(self, index):
        """
        Clear the bit at the given index.

        Parameters:
            index(int): The index of the bit to clear.
        """
        self._check_index(index)
        self._bits[index] = 0

    def get_bit(self, index):
        """
        Get the value of the bit at the given index.

        Parameters:
            index(int): The index of the bit to get the value of.

        Returns:
            int: The value of the bit at the given index.
        """
        if index < 0 or index >= len(self._bits):
            raise IndexError(f"Bit index {index} out of range (0, {len(self._bits)-1})")
        return self._bits[index]

    def toggle_bit(self, index):
        """
        Toggle the value of the bit at the given index.

        Parameters:
            index(int): The index of the bit to toggle the value of.
        """
        if index >= len(self._bits):
            self.resize(index + 1)
        self._bits[index] = not self._bits[index]

    def count(self):
        """
        Count the number of set bits in the BitSet.

        Returns:
            int: The number of set bits in the BitSet.
        """
        return self._bits.count()

    def __and__(self, other):
        """
        Execute a bitwise AND operation with another BitSet.

        Parameters:
            other(BitSet): The other BitSet to perform the AND operation with.

        Returns:
            BitSet: The result of the AND operation.
        """
        self._check_compatibility(other)
        result = BitSet()
        result._bits = self._bits & other._bits
        return result

    def __or__(self, other):
        """
        Execute a bitwise OR operation with another BitSet.

        Parameters:
            other(BitSet): The other BitSet to perform the OR operation with.

        Returns:
            BitSet: The result of the OR operation.
        """
        self._check_compatibility(other)
        result = BitSet()
        result._bits = self._bits | other._bits
        return result

    def __xor__(self, other):
        """
        Execute a bitwise XOR operation with another BitSet.

        Parameters:
            other(BitSet): The other BitSet to perform the XOR operation with.

        Returns:
            BitSet: The result of the XOR operation.
        """
        self._check_compatibility(other)
        result = BitSet()
        result._bits = self._bits ^ other._bits
        return result

    def __invert__(self):
        """
        Execute a bitwise NOT operation on the BitSet.

        Returns:
            BitSet: The result of the NOT operation.
        """
        result = BitSet()
        result._bits = ~self._bits
        return result

    def __eq__(self, other):
        """
        Check if two BitSet objects are equal.

        Parameters:
            other(BitSet): The other BitSet to compare with.

        Returns:
            bool: True if the two BitSet objects are equal, False otherwise.
        """
        if not isinstance(other, BitSet):
            return False
        return self._bits == other._bits

    def __repr__(self):
        return f"BitSet(size={len(self)}, bits={self._bits.to01()})"

    def iter_set_bits(self):
        """
        Iterate over the set bits in the BitSet.

        Returns:
            Iterator: An iterator over the set bits in the BitSet.
        """
        return self._bits.search(bitarray('1'))

    def save_to_file(self, filename):
        """
        Save the BitSet to a file.

        Parameters:
            filename(str): The name of the file to save the BitSet to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self._bits, f)

    @classmethod
    def load_from_file(cls, filename):
        """
        Load a BitSet from a file.

        Parameters:
            filename(str or Pathlike): The name of the file to load the BitSet from.

        Returns:
            BitSet: The loaded BitSet.
        """
        with open(filename, 'rb') as f:
            bits = pickle.load(f)
            return cls(initial_bits=bits)

    def to_bytes(self):
        """
        Transform the BitSet into a byte array.

        Returns:
            bytes: The byte array representation of the BitSet.
        """
        return self._bits.tobytes()

    @classmethod
    def from_bytes(cls, byte_data, size=None):
        """
        Create a BitSet from a byte array.

        Parameters:
            byte_data(bytes): The byte array to create the BitSet from.
            size(int): The size of the BitSet.

        Returns:
            BitSet: The created BitSet.
        """
        bits = bitarray()
        bits.frombytes(byte_data)
        if size is not None:
            bits = bits[:size]
        return cls(initial_bits=bits)

    def clear_all(self):
        """
        Clear all bits to 0.
        """
        self._bits.setall(0)

    def set_all(self):
        """
        Set all bits to 1.
        """
        self._bits.setall(1)

    def _check_compatibility(self, other):
        """
        Check if two BitSet objects are compatible.

        Parameters:
            other(BitSet): The other BitSet to check compatibility with.

        Raises:
            TypeError: If the other object is not a BitSet.
            ValueError: If the two BitSet objects are not of the same size.
        """
        if not isinstance(other, BitSet):
            raise TypeError("Operand must be a BitSet")
        if len(self) != len(other):
            raise ValueError("BitSets must be of the same size")

    def size(self):
        return len(self._bits)
