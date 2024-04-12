<div align="center">
  <a href="https://github.com/BirchKwok/MinVectorDB"><img src="https://github.com/BirchKwok/MinVectorDB/blob/main/pic/logo.png" alt="MinVectorDB" style="max-width: 20%; height: auto;"></a>
  <h3>A pure Python-implemented, lightweight, stateless, locally deployed vector database.</h3>
  <p>
    <a href="https://badge.fury.io/py/MinVectorDB"><img src="https://badge.fury.io/py/MinVectorDB.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/pyversions/MinVectorDB" alt="PyPI - Python Version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/l/MinVectorDB" alt="PyPI - License"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/dm/MinVectorDB" alt="PyPI - Downloads"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/implementation/MinVectorDB" alt="PyPI - Implementation"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/wheel/MinVectorDB" alt="PyPI - Wheel"></a>
    <a href="https://github.com/BirchKwok/MinVectorDB/actions/workflows/python-package.yml"><img src="https://github.com/BirchKwok/MinVectorDB/actions/workflows/python-package.yml/badge.svg" alt="Python package"></a>
  </p>
</div>

<div align="center">
  <a href="https://github.com/BirchKwok/MinVectorDB"><img src="https://github.com/BirchKwok/MinVectorDB/blob/main/pic/terminal-demo-show.gif" alt="Demo"></a>
</div>

> **WARNING**: MinVectorDB is actively being updated, and API backward compatibility is not guaranteed. You should use version numbers as a strong constraint during deployment to avoid unnecessary feature conflicts and errors.
> **Although our goal is to enable brute force search or inverted indexing on billion-scale vectors, we currently still recommend using it on a scale of millions of vectors or less for the best experience.**

*MinVectorDB* is a vector database implemented purely in Python, designed to be lightweight, stateless, and easy to deploy locally. It offers straightforward and clear Python APIs, aiming to lower the entry barrier for using vector databases. In response to user needs and to enhance its practicality, we are planning to introduce new features, including but not limited to:

- **Optimizing Global Search Performance**: We are focusing on algorithm and data structure enhancements to speed up searches across the database, enabling faster retrieval of vector data.
- **Enhancing Cluster Search with Inverted Indexes**: Utilizing inverted index technology, we aim to refine the cluster search process for better search efficiency and precision.
- **Refining Clustering Algorithms**: By improving our clustering algorithms, we intend to offer more precise and efficient data clustering to support complex queries.
- **Facilitating Vector Modifications and Deletions**: We will introduce features to modify and delete vectors, allowing for more flexible data management.
- **Implementing Rollback Strategies**: To increase database robustness and data security, rollback strategies will be added, helping users recover from incorrect operations or system failures easily.

Additionally, we are introducing a query caching feature, with a default cache for the most recent 10,000 query results. In cases where a query does not hit the cache, the system will calculate the cosine similarity between the given vector and cached vectors. If the similarity is greater than 0.85, it will return the result of the closest cached vector directly.

MinVectorDB focuses on achieving a 100% recall rate, prioritizing recall accuracy over high-speed search performance. This approach ensures that users can reliably retrieve all relevant vector data, making MinVectorDB particularly suitable for applications requiring responses within 100 milliseconds.

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

# cosine similarity threshold for cache result matching 
os.environ['MVDB_COSINE_SIMILARITY_THRESHOLD'] = '0.9'  # 'None' for disable this feature, default to 0.85

# specify the number of chunks in the memory cache
os.environ['MVDB_DATALOADER_BUFFER_SIZE'] = '20'  # default to '20', must be integer-like string
```


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
print("MinVectorDB all configs: ", '\n - ' + '\n - '.join([f'{k}: {v}' for k, v in min_vec.get_all_configs().items()]))
```

    MinVectorDB version is:  0.2.5
    MinVectorDB all configs:  
     - MVDB_LOG_LEVEL: INFO
     - MVDB_LOG_PATH: ./min_vec_db.log
     - MVDB_TRUNCATE_LOG: True
     - MVDB_LOG_WITH_TIME: False
     - MVDB_KMEANS_EPOCHS: 500
     - MVDB_QUERY_CACHE_SIZE: 10000
     - MVDB_COSINE_SIMILARITY_THRESHOLD: 0.9
     - MVDB_DATALOADER_BUFFER_SIZE: 20


### Sequentially add vectors.


```python
import numpy as np
from tqdm import tqdm

from min_vec import MinVectorDB


# Generate vectors that need to be saved, this code is only for this demo
np.random.seed(23)
def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

# distance can be 'L2' or 'cosine'
# index_mode can be 'FLAT' or 'IVF-FLAT', default is 'IVF-FLAT'
db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', index_mode='FLAT', chunk_size=100000, use_cache=False, scaler_bits=8)

# ========== Use automatic commit statements. Recommended. =============
# You can perform this operation multiple times, and the data will be appended to the database.
with db.insert_session():
    # Define the initial ID.
    id = 0
    for t in tqdm(get_test_vectors((5000000, 1024)), total=5000000, unit="vector"):
        if id == 0:
            query = t
            query_id = 0
            query_field = None
        # Vectors will be normalized after writing to the database.
        db.add_item(t, index=id)

        id += 1

res = db.query(query, k=10, return_similarity=True)

print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```

    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=16, chunk_size=100000,
    //    distance='cosine', index_mode='FLAT', 
    //    dtypes='float32', use_cache=False, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'
    100%|██████████| 5000000/5000000 [01:01<00:00, 81902.68vector/s] 


      - Query sample id:  0
      - Query sample field:  None
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (5000000, 1024)
    | - Query time: 1.12399 s
    | - Query K: 10
    | - Top 10 results index: [      0  126163 2995566 1455136 3285759 3671500 2498399 4372617 2141370
     3401650]
    | - Top 10 results similarity: [0.9967977  0.78328276 0.78147084 0.7807232  0.7804502  0.78013766
     0.77972347 0.77943265 0.7793954  0.779209  ]
    * - END OF REPORT -
    


### Bulk add vectors


```python
import numpy as np

from min_vec import MinVectorDB

# Generate vectors that need to be saved, this code is only for this demo
np.random.seed(23)
def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])
        
db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, index_mode='FLAT')

# You can perform this operation multiple times, and the data will be appended to the database.
with db.insert_session():  
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100000, 1024)):
        if id == 0:
            query = t 
            query_id = id
            query_field = None
        vectors.append((t, id))
        id += 1
        
    # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
    db.bulk_add_items(vectors)

res = db.query(query, k=10, return_similarity=True)
print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```

    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=16, chunk_size=10000,
    //    distance='cosine', index_mode='FLAT', 
    //    dtypes='float32', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  0
      - Query sample field:  None
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.04958 s
    | - Query K: 10
    | - Top 10 results index: [    0 67927 53447 47665 13859  5788 41949 64134 38082 18507]
    | - Top 10 results similarity: [0.9974107  0.7780206  0.77481455 0.7742663  0.7730598  0.77304006
     0.772899   0.772897   0.7720787  0.7714467 ]
    * - END OF REPORT -
    


### Use field to improve Searching Recall


```python
import numpy as np

from min_vec import MinVectorDB


# Generate vectors that need to be saved, this code is only for this demo
np.random.seed(23)
def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, index_mode='IVF-FLAT')

with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100000, 1024)):
        if id == 0:
            query = t
            query_id = id
            query_field = 'test_'+str(id // 100)
        vectors.append((t, id, 'test_'+str(id // 100)))
        id += 1
        
    db.bulk_add_items(vectors)


res = db.query(query, k=10, fields=[query_field])

print("  - Query sample id: ", query_id)

print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```

    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=16, chunk_size=10000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float32', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  0
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.00236 s
    | - Query K: 10
    | - Top 10 results index: [ 0 60 76 63 52 14 27 61 83 79]
    | - Top 10 results similarity: [0.9974107  0.7515197  0.7489892  0.7483421  0.7466648  0.7452562
     0.74377525 0.7390184  0.7307658  0.7282359 ]
    * - END OF REPORT -
    


### Use subset_indices to narrow down the search range


```python
import numpy as np

from min_vec import MinVectorDB


# Generate vectors that need to be saved, this code is only for this demo
np.random.seed(23)
def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, index_mode='IVF-FLAT')

with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100000, 1024)):
        if id == 0:
            query = t
            query_id = id
            query_field = 'test_'+str(id // 100)

        vectors.append((t, id, 'test_'+str(id // 100)))
        id += 1
        
    db.bulk_add_items(vectors)

# You may define both 'subset_indices' and 'fields'
res = db.query(query, k=10, subset_indices=list(range(query_id - 20, query_id + 20)))
print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```

    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=16, chunk_size=10000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float32', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  0
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.00384 s
    | - Query K: 10
    | - Top 10 results index: [ 0  9 14  7  3  1 19 18  8 17]
    | - Top 10 results similarity: [0.9974107  0.75857055 0.7452562  0.7420311  0.7413465  0.73768425
     0.7370884  0.73495173 0.73355234 0.7306047 ]
    * - END OF REPORT -
    


### Conduct searches by specifying both subset_indices and fields simultaneously.


```python
import numpy as np

from min_vec import MinVectorDB

# Generate vectors that need to be saved, this code is only for this demo
np.random.seed(23)
def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, index_mode='IVF-FLAT')


with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    last_field = None
    for t in get_test_vectors((100000, 1024)):
        if id == 0:
            query = t
            query_id = id
            query_field = 'test_'+str(id // 100)
        vectors.append((t, id, 'test_'+str(id // 100)))
        id += 1
        
    db.bulk_add_items(vectors)

# You may define both 'subset_indices' and 'fields'
# If there is no intersection between subset_indices and fields, there will be no result. 
# If there is an intersection, the query results within the intersection will be returned.
res = db.query(query, k=10, subset_indices=list(range(query_id-20, query_id + 20)), fields=[query_field])
print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```

    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=16, chunk_size=10000,
    //    distance='L2', index_mode='IVF-FLAT', 
    //    dtypes='float32', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  0
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.05352 s
    | - Query K: 10
    | - Top 10 results index: [ 0  9 14 13  7  3  1 19 10  8]
    | - Top 10 results similarity: [0.00545208 0.6913169  0.71014524 0.7103149  0.71466976 0.71513784
     0.7208866  0.7213987  0.72268754 0.7265912 ]
    * - END OF REPORT -
    

