from test import MinVectorDB
import numpy as np


def test_modify_exists_database_parameter():
    db = MinVectorDB(dim=1024, database_path='test_mvdb.mvdb', chunk_size=10000, dtypes='float32')

    # insert data
    with db.insert_session():
        for i in range(10):
            db.add_item(np.random.rand(1024), index=i)

    # modify exists database
    db = MinVectorDB(dim=1023, database_path='test_mvdb.mvdb', chunk_size=10002, dtypes='float16')

    assert db._matrix_serializer.dim == 1024  # not modified
    assert db._matrix_serializer.chunk_size == 10000  # not modified
    assert db._matrix_serializer.dtypes == np.float32  # not modified

    # delete database
    db.delete()


def test_using_api_after_database_deleted(capfd):
    db = MinVectorDB(dim=1024, database_path='test_mvdb.mvdb', chunk_size=10000, dtypes='float32')

    # insert data
    with db.insert_session():
        for i in range(10):
            db.add_item(np.random.rand(1024), index=i)

    # delete database
    db.delete()

    # using add_item function after database deleted, it should print "The database has been deleted,
    # and the operation is invalid."
    db.add_item(np.random.rand(1024), index=11)
    out, err = capfd.readouterr()
    assert out == "The database has been deleted, and the operation is invalid.\n"

    # bulk_add_items
    db.bulk_add_items([(np.random.rand(1024), 11)])
    out, err = capfd.readouterr()
    assert out == "The database has been deleted, and the operation is invalid.\n"

    # query
    db.query(np.random.rand(1024))
    out, err = capfd.readouterr()
    assert out == "The database has been deleted, and the operation is invalid.\n"

    # delete database
    db.delete()
    out, err = capfd.readouterr()
    assert out == "The database has been deleted, and the operation is invalid.\n"
