<div align="center">
  <a href="https://github.com/BirchKwok/MinVectorDB"><img src="https://github.com/BirchKwok/MinVectorDB/blob/main/pic/logo.png" alt="MinVectorDB" style="max-width: 20%; height: auto;"></a>
  <h3>A pure Python-implemented, lightweight, serverless, locally deployed vector database.</h3>
  <p>
    <a href="https://badge.fury.io/py/MinVectorDB"><img src="https://badge.fury.io/py/MinVectorDB.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/pyversions/MinVectorDB" alt="PyPI - Python Version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/l/MinVectorDB" alt="PyPI - License"></a>
    <a href="https://github.com/BirchKwok/MinVectorDB/actions/workflows/python-package.yml"><img src="https://github.com/BirchKwok/MinVectorDB/actions/workflows/python-package.yml/badge.svg" alt="Python package"></a>
  </p>
</div>

⚡ **Serverless, simple parameters, simple API.**

⚡ **Fast, memory-efficient, easily scales to millions of vectors.**

⚡ **Supports cosine similarity and L2 distance, uses FLAT for exhaustive search or IVF-FLAT for inverted indexing.**

⚡ **Friendly caching technology stores recently queried vectors for accelerated access.**

⚡ **Based on a generic Python software stack, platform-independent, highly versatile.**

> **WARNING**: MinVectorDB is actively being updated, and API backward compatibility is not guaranteed. You should use version numbers as a strong constraint during deployment to avoid unnecessary feature conflicts and errors.
> **Although our goal is to enable brute force search or inverted indexing on billion-scale vectors, we currently still recommend using it on a scale of millions of vectors or less for the best experience.**

*MinVectorDB* is a vector database implemented purely in Python, designed to be lightweight, serverless, and easy to deploy locally. It offers straightforward and clear Python APIs, aiming to lower the entry barrier for using vector databases. In response to user needs and to enhance its practicality, we are planning to introduce new features, including but not limited to:

- **Optimizing Global Search Performance**: We are focusing on algorithm and data structure enhancements to speed up searches across the database, enabling faster retrieval of vector data.
- **Enhancing Cluster Search with Inverted Indexes**: Utilizing inverted index technology, we aim to refine the cluster search process for better search efficiency and precision.
- **Refining Clustering Algorithms**: By improving our clustering algorithms, we intend to offer more precise and efficient data clustering to support complex queries.
- **Facilitating Vector Modifications and Deletions**: We will introduce features to modify and delete vectors, allowing for more flexible data management.
- **Implementing Rollback Strategies**: To increase database robustness and data security, rollback strategies will be added, helping users recover from incorrect operations or system failures easily.

MinVectorDB focuses on achieving 100% recall, prioritizing recall accuracy over high-speed search performance. This approach ensures that users can reliably retrieve all relevant vector data, making MinVectorDB particularly suitable for applications that require responses within hundreds of milliseconds.

While the project has not yet been benchmarked against other systems, we believe these planned features will significantly enhance MinVectorDB's capabilities in managing and retrieving vector data, addressing a wide range of user needs.

## Install

```shell
pip install MinVectorDB
```

## Qucik Start

### Environment setup (optional, Each instance can only be set once, and needs to be set before instantiation)


```python
import os

# logger settings
# logger level: DEBUG, INFO, WARNING, ERROR, CRITICAL
os.environ['MVDB_LOG_LEVEL'] = 'INFO'  # default: INFO, Options are 'DEBUG'/'INFO'/'WARNING'/'ERROR'/'CRITICAL'

# log path
os.environ['MVDB_LOG_PATH'] = './min_vec_db.log'  # default: None

# whether to truncate log file
os.environ['MVDB_TRUNCATE_LOG'] = 'True'  # default: True

# whether to add time to log
os.environ['MVDB_LOG_WITH_TIME'] = 'False'  # default: False

# clustering settings
# kmeans epochs
os.environ['MVDB_KMEANS_EPOCHS'] = '500'  # default: 100

# query cache size
os.environ['MVDB_QUERY_CACHE_SIZE'] = '10000'  # default: 10000

# specify the number of chunks in the memory cache
os.environ['MVDB_DATALOADER_BUFFER_SIZE'] = '20'  # default to '40', must be integer-like string
```


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
print("MinVectorDB all configs: ", '\n - ' + '\n - '.join([f'{k}: {v}' for k, v in min_vec.get_all_configs().items()]))
```

    MinVectorDB version is:  0.3.0
    MinVectorDB all configs:  
     - MVDB_LOG_LEVEL: INFO
     - MVDB_LOG_PATH: ./min_vec_db.log
     - MVDB_TRUNCATE_LOG: True
     - MVDB_LOG_WITH_TIME: False
     - MVDB_KMEANS_EPOCHS: 500
     - MVDB_QUERY_CACHE_SIZE: 10000
     - MVDB_DATALOADER_BUFFER_SIZE: 20


### create a collection


```python
from min_vec import MinVectorDB

# Specify database root directory
my_db = MinVectorDB(root_path='my_vec_db')
```

    MinVectorDB - INFO - Successful initialization of MinVectorDB in root_path: /projects/MinVectorDB/my_vec_db



```python
collection = my_db.require_collection("test_collection", 4, drop_if_exists=True)
```

    MinVectorDB - INFO - Creating collection test_collection with: 
    //    dim=4, collection='test_collection', 
    //    n_clusters=16, chunk_size=100000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float32', use_cache=True, 
    //    scaler_bits=8, n_threads=10


### Add vectors


```python
with collection.insert_session():
    id = collection.add_item(vector=[0.01, 0.34, 0.74, 0.31], id=1, field={'field': 'test_1', 'order': 0})
    id = collection.add_item(vector=[0.36, 0.43, 0.56, 0.12], id=2, field={'field': 'test_1', 'order': 1})
    id = collection.add_item(vector=[0.03, 0.04, 0.10, 0.51], id=3, field={'field': 'test_2', 'order': 2})
    id = collection.add_item(vector=[0.11, 0.44, 0.23, 0.24], id=4, field={'field': 'test_2', 'order': 3})
    id = collection.add_item(vector=[0.91, 0.43, 0.44, 0.67], id=5, field={'field': 'test_2', 'order': 4})
    id = collection.add_item(vector=[0.92, 0.12, 0.56, 0.19], id=6, field={'field': 'test_3', 'order': 5})
    id = collection.add_item(vector=[0.18, 0.34, 0.56, 0.71], id=7, field={'field': 'test_1', 'order': 6})
    id = collection.add_item(vector=[0.01, 0.33, 0.14, 0.31], id=8, field={'field': 'test_2', 'order': 7})
    id = collection.add_item(vector=[0.71, 0.75, 0.91, 0.82], id=9, field={'field': 'test_3', 'order': 8})
    id = collection.add_item(vector=[0.75, 0.44, 0.38, 0.75], id=10, field={'field': 'test_1', 'order': 9})

# If you do not use the insert_session function, you need to manually call the commit function to submit the data
# collection.commit()
```


```python
print(id)
```

    10


### Query


```python
collection.query(vector=[0.36, 0.43, 0.56, 0.12], k=3)
```




    (array([2, 9, 1]), Array([0.99822044, 0.9201999 , 0.8585187 ], dtype=float32))




```python
print(collection.query_report_)
```

    
    * - MOST RECENT QUERY REPORT -
    | - Database Shape: (10, 4)
    | - Query Time: 0.00125 s
    | - Query Distance: cosine
    | - Query K: 3
    | - Top 3 Results ID: [2 9 1]
    | - Top 3 Results Similarity: [0.99822  0.9202   0.858519]
    * - END OF REPORT -
    



```python
collection.status_report_['DATABASE STATUS REPORT']
```




    {'Database shape': (10, 4),
     'Database last_commit_time': datetime.datetime(2024, 4, 23, 21, 16, 38, 764711),
     'Database commit status': True,
     'Database index_mode': 'IVF-FLAT',
     'Database distance': 'cosine',
     'Database use_cache': True,
     'Database status': 'ACTIVE'}



### Use Filter


```python
import operator

from min_vec.structures.filter import Filter, FieldCondition, MatchField, IDCondition, MatchID


collection.query(
    vector=[0.36, 0.43, 0.56, 0.12], 
    k=10, 
    query_filter=Filter(
        must=[
            FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
        ], 
        any=[

            FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
            IDCondition(MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
        ]
    )
)

print(collection.query_report_)
```

    
    * - MOST RECENT QUERY REPORT -
    | - Database Shape: (10, 4)
    | - Query Time: 0.00237 s
    | - Query Distance: cosine
    | - Query K: 10
    | - Top 10 Results ID: [ 2  1  4  5 10  3]
    | - Top 10 Results Similarity: [0.99822    0.858519   0.85362    0.812733   0.783597   0.34614798]
    * - END OF REPORT -
    


### Drop a collection


```python
print("Collection list before dropping:", my_db.show_collections())
my_db.drop_collection("test_collection")
print("Collection list after dropped:", my_db.show_collections())
```

    Collection list before dropping: ['test_collection']
    Collection list after dropped: []


## Drop the database


```python
my_db.drop_database()
my_db
```




    DELETED MinVectorDB(root_path='/projects/MinVectorDB/my_vec_db')



## What's Next

- [Collection's operations](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/collections.ipynb)
- [Add vectors to collection](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/add_vectors.ipynb)
- [Using different indexing methods](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/index_mode.ipynb)
- [Using different distance metric functions](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/distance.ipynb)
- [Diversified queries](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/queries.ipynb)
- [Benchmarks](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/Benchmarks.ipynb)
