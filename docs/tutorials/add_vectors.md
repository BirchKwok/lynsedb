# Tutorial: Add vectors

First, import the LynseDB client:

```python linenums="1"
import lynse
```


And then, Start the server, you can specify the root path directly
```python linenums="1"
client = lynse.VectorDBClient(uri="demo")
my_db = client.create_database("my_vec_db", drop_if_exists=False)
```

## Sequentially add vectors

create or truncate a collection:
```python linenums="1"
collection = my_db.require_collection("test_add_vectors", dim=128, drop_if_exists=True)
```

The `add_item` API can be used to add vectors to the database one by one, and it is recommended to always submit data within the `insert_session` context manager to ensure the highest level of data security:
```python linenums="1"
import numpy as np

# If the field is not passed, a blank string field will be used.
# The add_item function returns the id of the data by default.
with collection.insert_session() as session:
    id = session.add_item(vector=np.random.random(128), id=1, field={'test': 'test', 'test1': 'test2'})

# If you do not use the insert_session function, you need to manually call the commit function to submit the data
# collection.commit()
```

    
    2024-09-13 10:44:57 - LynseDB - INFO - Saving data...
    2024-09-13 10:44:57 - LynseDB - INFO - Writing chunk to storage...
    Data persisting: 100%|██████████| 1/1 [00:00<00:00, 264.76chunk/s]
    
    2024-09-13 10:44:57 - LynseDB - INFO - Writing chunk to storage done.
    2024-09-13 10:44:57 - LynseDB - INFO - Pre-building the index...
    2024-09-13 10:44:57 - LynseDB - INFO - Building an index using the `Flat-IP` index mode...
    2024-09-13 10:44:57 - LynseDB - INFO - Index built.

Now we can see what content the `id` returns

```python linenums="1"
print(id)
```

    1


Print `collection` to get some information about it
```python linenums="1"
print(collection)
```

    LocalCollectionInstance(
        database="my_vec_db", 
        collection="test_add_vectors", 
        shape=(1, 128)
    )


## Add vectors in bulk

Now let's try to add vectors in bulk:
```python linenums="1"
collection = my_db.require_collection("test_min_vec", dim=128, drop_if_exists=True)
```


Similarly, we use the `bulk_add_items` method within the `insert_session` context manager to submit all the data at once:
```python linenums="1"
import numpy as np

vectors = []

with collection.insert_session() as session:
    for i in range(10):
        # The order is vector, id, field
        vectors.append((np.random.random(128), i, {'test': f'test_{i}'}))


    ids = session.bulk_add_items(vectors)

print(ids)
```

    
    2024-09-13 10:45:14 - LynseDB - INFO - Saving data...
    2024-09-13 10:45:14 - LynseDB - INFO - Writing chunk to storage...
    Data persisting: 100%|██████████| 1/1 [00:00<00:00, 460.86chunk/s]
    
    2024-09-13 10:45:14 - LynseDB - INFO - Writing chunk to storage done.
    2024-09-13 10:45:14 - LynseDB - INFO - Pre-building the index...
    2024-09-13 10:45:14 - LynseDB - INFO - Building an index using the `Flat-IP` index mode...
    2024-09-13 10:45:14 - LynseDB - INFO - Index built.

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


Isn't it simple? At this point, we can check if the shape of the `collection` is correct:
```python linenums="1"
print(collection)
```

    LocalCollectionInstance(
        database="my_vec_db", 
        collection="test_min_vec", 
        shape=(10, 128)
    )

