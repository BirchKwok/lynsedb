"""Shared pytest fixtures for LynseDB standard tests."""
import numpy as np
import pytest

import lynse

DIM = 8
N = 20


@pytest.fixture(scope="function")
def tmp_root(tmp_path):
    """A fresh temporary root directory per test."""
    yield str(tmp_path)


@pytest.fixture(scope="function")
def client(tmp_root):
    """VectorDBClient in local mode pointing at a temp dir."""
    c = lynse.VectorDBClient(uri=tmp_root)
    yield c


@pytest.fixture(scope="function")
def db(client):
    """A fresh database named 'test_db'."""
    db = client.create_database("test_db", drop_if_exists=True)
    yield db


@pytest.fixture(scope="function")
def collection(db):
    """An empty collection of dimension DIM."""
    coll = db.require_collection("test_col", dim=DIM, drop_if_exists=True)
    yield coll


@pytest.fixture(scope="function")
def populated_collection(collection):
    """Collection pre-loaded with N vectors and fields."""
    np.random.seed(42)
    items = [
        (
            np.random.rand(DIM).astype(np.float32),
            i,
            {"tag": f"item_{i}", "group": i % 3},
        )
        for i in range(N)
    ]
    with collection.insert_session() as session:
        session.bulk_add_items(items, enable_progress_bar=False)
    yield collection


@pytest.fixture(scope="function")
def query_vec():
    """A single deterministic query vector."""
    np.random.seed(0)
    return np.random.rand(DIM).astype(np.float32)
