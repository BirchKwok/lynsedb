import pytest

from lynse.utils.utils import OpsError

import lynse
import numpy as np

client = lynse.VectorDBClient()
database = client.create_database(database_name='test_local_db', drop_if_exists=True)


def test_modify_exists_database_parameter():
    collection = database.require_collection("test_collection", dim=1024, chunk_size=10000,
                                             dtypes='float32', drop_if_exists=True)

    # insert data
    with collection.insert_session():
        for i in range(10):
            collection.add_item(np.random.rand(1024), id=i)

    collection = database.require_collection("test_collection", dim=1024, chunk_size=10000,
                                             dtypes='float32', drop_if_exists=False)

    assert collection._matrix_serializer.dim == 1024  # not modified
    assert collection._matrix_serializer.chunk_size == 10000  # not modified
    assert collection._matrix_serializer.dtypes == np.float32  # not modified


def test_using_api_after_database_deleted(capfd):
    collection = database.require_collection("test_collection2", dim=1024, chunk_size=10000, dtypes='float32', drop_if_exists=True)

    # insert data
    with collection.insert_session():
        for i in range(10):
            collection.add_item(np.random.rand(1024), id=i)

    # delete database
    collection.delete()

    # using add_item function after database deleted, it should raise ValueError
    with pytest.raises(OpsError):
        collection.add_item(np.random.rand(1024), id=11)

    # using bulk_add_items function after database deleted, it should raise ValueError
    with pytest.raises(OpsError):
        collection.bulk_add_items([(np.random.rand(1024), 11)])

    # using query function after database deleted, it should raise ValueError
    with pytest.raises(OpsError):
        collection.search(np.random.rand(1024))
