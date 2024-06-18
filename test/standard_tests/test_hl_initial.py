import os.path
from pathlib import Path

import numpy as np

import lynse


client = lynse.VectorDBClient()


def test_create_collection():
    my_vec_db = client.create_database("my_vec_db", drop_if_exists=False)
    collection = my_vec_db.require_collection('test_collection', dim=4, drop_if_exists=True)

    assert Path(collection._database_path).stem == 'test_collection'
    assert collection._matrix_serializer.dim == 4
    assert collection._matrix_serializer.chunk_size == 100000
    assert collection._matrix_serializer.dtypes == np.float32
    assert Path(os.path.expanduser("~/.LynseDB/databases/my_vec_db/test_collection")).exists()

    my_vec_db.drop_collection('test_collection')
    assert not Path(os.path.expanduser("~/.LynseDB/databases/my_vec_db/test_collection")).exists()

    my_vec_db.drop_database()
    assert not Path(os.path.expanduser("~/.LynseDB/databases/my_vec_db/test_collection/my_vec_db")).exists()


def test_show_collections():
    my_vec_db = client.create_database("my_vec_db", drop_if_exists=False)
    collection = my_vec_db.require_collection('test_collection', dim=4, drop_if_exists=True)

    collections = my_vec_db.show_collections()
    assert collections == ['test_collection']

    my_vec_db.drop_collection('test_collection')
    my_vec_db.drop_database()
    assert not Path(os.path.expanduser("~/.LynseDB/databases/my_vec_db/test_collection/my_vec_db")).exists()


def test_get_an_exists_collection():
    my_vec_db = client.create_database("my_vec_db", drop_if_exists=False)
    collection = my_vec_db.require_collection('test_collection', dim=4, drop_if_exists=True)

    collection = my_vec_db.get_collection('test_collection')
    assert Path(collection._database_path).stem == 'test_collection'

    my_vec_db.drop_collection('test_collection')
    my_vec_db.drop_database()
    assert not Path(os.path.expanduser("~/.LynseDB/databases/my_vec_db/test_collection/my_vec_db")).exists()
