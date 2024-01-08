import shutil
from pathlib import Path

import numpy as np
from spinesUtils.utils import Timer
from tqdm import trange

from min_vec.model import MinVectorDB


timer = Timer()


def get_database(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, dtypes=np.float32, device='cpu'):
    if Path('.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])).exists():
        shutil.rmtree(Path('.mvdb'.join(Path(database_path).name.split('.mvdb')[:-1])))

    database = MinVectorDB(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes, device=device)
    if database._database_chunk_path:
        database.delete()
    database._database_chunk_path = []
    return database


def get_database_for_query(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000,
                           dtypes=np.float32, device='cpu', **kwargs):
    database = get_database(dim=dim, database_path=database_path, chunk_size=chunk_size, dtypes=dtypes, device=device)
    np.random.seed(2023)

    def get_test_vectors(shape):
        for i in trange(shape[0], total=shape[0], unit='lines'):
            yield np.random.random(shape[1])

    with database.insert_session():
        id = 0
        if 'lines' in kwargs:
            lines = kwargs['lines']
        else:
            lines = 100000
        for t in get_test_vectors((lines, 1024)):
            database.add_item(t, id, normalize=True)
            id += 1

    return database


# 测试MinVectorDB在不同设备上的运行时间
# CPU
def query_test_cpu(lines=100000):
    timer.start()
    database = get_database_for_query(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, dtypes=np.float32,
                                      device='cpu', lines=lines)
    print(f"\n* [CPU Insert] Time cost {timer.last_timestamp_diff():>.4f} s.")
    vec = np.random.random(1024)
    timer.middle_point()

    n, d = database.query(vec, k=12)

    print(f"\n* [CPU Query] Time cost {timer.last_timestamp_diff():>.4f} s.")

    return database


# GPU
def query_test_gpu(lines=100000, device='mps'):
    timer.start()

    database = get_database_for_query(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, dtypes=np.float32,
                                      device=device, lines=lines)
    print(f"\n* [GPU Insert] Time cost {timer.last_timestamp_diff():>.4f} s.")
    vec = np.random.random(1024)
    timer.middle_point()

    n, d = database.query(vec, k=12)

    print(f"\n* [GPU Query] Time cost {timer.last_timestamp_diff():>.4f} s.")

    return database


if __name__ == '__main__':
    database = query_test_gpu(lines=1000000, device='mps')
    database.delete()
    database = query_test_cpu(lines=1000000)
    database.delete()
