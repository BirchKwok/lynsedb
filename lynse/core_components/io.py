import struct
from functools import lru_cache

import numpy as np
import filelock
import os


_HEADER_STRUCT = struct.Struct('<I30sI')
_HEADER_STRUCT_SIZE = _HEADER_STRUCT.size


@lru_cache(maxsize=None)
def get_dtype(dtype_str):
    """
    get the numpy dtype from the string

    Parameters:
        dtype_str (str): the string representation of the dtype

    Returns:
        np.dtype: the numpy dtype
    """
    return np.dtype(dtype_str)


def load_nnp_header(filename):
    """
    load the header of the nnp file

    Parameters:
        filename (str): the path to the nnp file

    Returns:
        (np.dtype, tuple): the data type and the shape of the data
    """
    global _HEADER_STRUCT, _HEADER_STRUCT_SIZE

    fd = os.open(filename, os.O_RDONLY)
    try:
        header_bytes = os.read(fd, _HEADER_STRUCT_SIZE)
    finally:
        os.close(fd)

    current_rows, dtype_bytes, data_shape = _HEADER_STRUCT.unpack(header_bytes)
    dtype_str = dtype_bytes.decode('utf-8').strip()
    dtype = get_dtype(dtype_str)

    return dtype, (current_rows, data_shape)


def save_nnp(filename, data, append=False):
    """
    save data to nnp file

    Parameters:
        filename (str): the path to the nnp file
        data (np.ndarray): the data to be saved
        append (bool): if True, append the data to the existing file

    Returns:
        None
    """
    lock = filelock.FileLock(f"{filename}.lock")
    with lock:
        # preset header size
        header_size = 100
        dtype_str = str(data.dtype)
        data_shape = data.shape[1]  # Assuming the data is two-dimensional, get the number of columns

        if append and os.path.exists(filename):
            with open(filename, 'r+b') as f:
                # Read file header
                f.seek(0)
                header = f.read(header_size)
                current_rows = int.from_bytes(header[0:4], 'little')
                stored_dtype_str = header[4:34].decode('utf-8').strip()
                stored_data_shape = int.from_bytes(header[34:38], 'little')

                # Check if data type and shape match
                if dtype_str != stored_dtype_str or data_shape != stored_data_shape:
                    raise TypeError(
                        "The appended data does not match the data type or shape of the existing data in the file."
                    )

                # check if the number of rows exceeds the limit
                if current_rows + data.shape[0] > 10000000:
                    raise Exception("Exceeds the maximum number of rows limit.")

                # update the number of rows in the file header
                new_rows = current_rows + data.shape[0]
                f.seek(0)
                f.write(new_rows.to_bytes(4, 'little'))

                # write data
                f.seek(header_size + current_rows * data_shape * data.dtype.itemsize)
                f.write(data.tobytes())
        else:
            # create new file or overwrite existing file
            with open(filename, 'w+b') as f:
                # initialize the number of rows
                current_rows = data.shape[0]

                # check if the number of rows exceeds the limit
                if current_rows > 10000000:
                    raise Exception("Exceeds the maximum number of rows limit.")

                # build file header
                header = bytearray(header_size)
                header[0:4] = current_rows.to_bytes(4, 'little')
                dtype_bytes = dtype_str.encode('utf-8')
                header[4:34] = dtype_bytes.ljust(30, b' ')  # ensure fixed length
                header[34:38] = data_shape.to_bytes(4, 'little')

                # write file header
                f.seek(0)
                f.write(header)

                # write data
                f.seek(header_size)
                f.write(data.tobytes())


def load_nnp(filename, mmap_mode=False):
    """
    load data from nnp file

    Parameters:
        filename (str): the path to the nnp file
        mmap_mode (bool): if True, use numpy.memmap to load the data

    Returns:
        (np.ndarray or np.memmap): the data loaded from the nnp file
    """
    data_offset = 100  # preset header size

    # read file header information
    dtype, (current_rows, data_shape) = load_nnp_header(filename)

    if mmap_mode:
        return np.memmap(filename, mode='r', dtype=dtype,
                         shape=(current_rows, data_shape), offset=data_offset)
    else:
        return np.fromfile(filename, dtype=dtype, count=current_rows * data_shape,
                           offset=data_offset).reshape(-1, data_shape)
