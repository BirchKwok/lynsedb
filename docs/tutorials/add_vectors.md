# Tutorial: Add vectors

First, import the LynseDB client:

```python linenums="1"
import lynse
import numpy as np
```

Create a local client (no server needed):

```python linenums="1"
client = lynse.VectorDBClient()
my_db = client.create_database("my_vec_db", drop_if_exists=False)
```

## Add vectors one at a time

Create or truncate a collection:

```python linenums="1"
collection = my_db.require_collection("test_add_vectors", dim=128, drop_if_exists=True)
```

Use `insert_session` to automatically commit on exit. Pass a `field` dict to store
arbitrary metadata alongside each vector.

```python linenums="1"
with collection.insert_session() as session:
    ret_id = session.add_item(
        vector=np.random.rand(128).astype(np.float32),
        id=1,
        field={'category': 'A', 'score': 0.9},
    )

# Without insert_session, call commit() manually:
# collection.commit()
```

```python linenums="1"
print(ret_id)       # 1
print(collection)
```

    1
    LocalCollectionInstance(
        database="my_vec_db",
        collection="test_add_vectors",
        shape=(1, 128)
    )

Check whether an ID exists and query the highest stored ID:

```python linenums="1"
print(collection.is_id_exists(1))   # True
print(collection.max_id)            # 1
```

## Add vectors in bulk

```python linenums="1"
collection = my_db.require_collection("test_bulk_add", dim=128, drop_if_exists=True)
```

Build a list of `(vector, id, field)` tuples and pass them to `bulk_add_items`:

```python linenums="1"
items = [
    (np.random.rand(128).astype(np.float32), i, {'category': f'cat_{i % 3}'})
    for i in range(10)
]

with collection.insert_session() as session:
    ids = session.bulk_add_items(items)

print(ids)          # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(collection)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    LocalCollectionInstance(
        database="my_vec_db",
        collection="test_bulk_add",
        shape=(10, 128)
    )

## High-throughput binary bulk add

When you have large volumes of vectors without per-vector metadata, use
`bulk_add_binary` for maximum write throughput. IDs are assigned automatically
starting from `max_id + 1`.

```python linenums="1"
collection = my_db.require_collection("test_binary_add", dim=128, drop_if_exists=True)

vecs = np.random.rand(10_000, 128).astype(np.float32)
n_added = collection.bulk_add_binary(vecs, batch_size=5000)
collection.commit()

print(f"Added {n_added} vectors")
print(collection.shape)   # (10000, 128)
```

## Check collection info

```python linenums="1"
print(collection.shape)    # (n_vectors, dim)
print(collection.max_id)   # highest user ID
print(collection.stats())
# {'n_vectors': 10000, 'n_live': 10000, 'n_tombstoned': 0,
#  'dimension': 128, 'index_mode': 'FLAT', 'max_id': ...}
```
