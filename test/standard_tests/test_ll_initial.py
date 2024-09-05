import os
import shutil

import pytest

from lynse.utils.utils import OpsError
from test import ExclusiveDB
import numpy as np


def test_modify_exists_database_parameter():
    if os.path.exists('test_mvdb'):
        shutil.rmtree('test_mvdb')

    db = ExclusiveDB(dim=1024, database_path='test_mvdb', chunk_size=10000, dtypes='float32')

    # insert data
    with db.insert_session():
        for i in range(10):
            db.add_item(np.random.rand(1024), id=i)

    # modify exists database
    db = ExclusiveDB(dim=1023, database_path='test_mvdb', chunk_size=10002, dtypes='float16')

    assert db._matrix_serializer.dim == 1024  # not modified
    assert db._matrix_serializer.chunk_size == 10000  # not modified
    assert db._matrix_serializer.dtypes == np.float32  # not modified

    # delete database
    db.delete()

    del db


def test_using_api_after_database_deleted(capfd):
    if os.path.exists('test_mvdb'):
        shutil.rmtree('test_mvdb')

    db = ExclusiveDB(dim=1024, database_path='test_mvdb', chunk_size=10000, dtypes='float32')

    # insert data
    with db.insert_session():
        for i in range(10):
            db.add_item(np.random.rand(1024), id=i)

    # delete database
    db.delete()

    # using add_item function after database deleted, it should raise ValueError
    with pytest.raises(OpsError):
        db.add_item(np.random.rand(1024), id=11)

    # using bulk_add_items function after database deleted, it should raise ValueError
    with pytest.raises(OpsError):
        db.bulk_add_items([(np.random.rand(1024), 11)])

    # using query function after database deleted, it should raise ValueError
    with pytest.raises(OpsError):
        db.search(np.random.rand(1024))

    # delete database
    db.delete()

    del db
