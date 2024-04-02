<div align="center">
  <h1><a href="https://github.com/BirchKwok/MinVectorDB"><img src="https://github.com/BirchKwok/MinVectorDB/blob/main/pic/logo.png" alt="MinVectorDB"></a></h1>
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

# computing platform, can be set to platforms supported by torch.device 
os.environ['MVDB_COMPUTE_DEVICE'] = 'mps' # default to 'cpu', torch.device

# specify the number of chunks in the memory cache
os.environ['MVDB_DATALOADER_BUFFER_SIZE'] = '20'  # default to '20', must be integer-like string
```


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
print("MinVectorDB all configs: ", '\n - ' + '\n - '.join([f'{k}: {v}' for k, v in min_vec.get_all_configs().items()]))
```

    MinVectorDB version is:  0.2.3
    MinVectorDB all configs:  
     - MVDB_LOG_LEVEL: INFO
     - MVDB_LOG_PATH: ./min_vec_db.log
     - MVDB_TRUNCATE_LOG: True
     - MVDB_LOG_WITH_TIME: False
     - MVDB_KMEANS_EPOCHS: 500
     - MVDB_QUERY_CACHE_SIZE: 10000
     - MVDB_COSINE_SIMILARITY_THRESHOLD: 0.9
     - MVDB_COMPUTE_DEVICE: cpu
     - MVDB_USER_MESSAGE_PATH: None
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
db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', distance='cosine',
                 index_mode='FLAT', chunk_size=100000, use_cache=False, scaler_bits=8)

# ========== Use automatic commit statements. Recommended. =============
# You can perform this operation multiple times, and the data will be appended to the database.
with db.insert_session():
    # Define the initial ID.
    id = 0
    for t in tqdm(get_test_vectors((1000000, 1024)), total=1000000, unit="vector"):
        if id == 0:
            query = t / np.linalg.norm(t)
            query_id = 0
            query_field = None
        # Vectors need to be normalized before writing to the database.
        db.add_item(t, index=id, normalize=True, save_immediately=False)

        id += 1

res = db.query(query, k=10)

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
    100%|██████████| 1000000/1000000 [00:15<00:00, 65961.24vector/s]


      - Query sample id:  0
      - Query sample field:  None
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (1000000, 1024)
    | - Query time: 0.34045 s
    | - Query K: 10
    | - Query normalize: False
    | - Top 10 results index: [     0 126163 934623 376250 136782  67927 927723 454821 909201 283657]
    | - Top 10 results similarity: [1.         0.7866893  0.7823357  0.78192943 0.78141373 0.78101647
     0.78069043 0.7806051  0.78056455 0.78041667]
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
        # Vectors need to be normalized before writing to the database.
        vectors.append((t, id))
        id += 1
        
    # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
    db.bulk_add_items(vectors, normalize=True, save_immediately=False)


query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

res = db.query(query, k=10)
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
    //    reindex_if_conflict=False, scaler_bits=None
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  0
      - Query sample field:  
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.03720 s
    | - Query K: 10
    | - Query normalize: False
    | - Top 10 results index: [    0 67927 53447 47665 64134 13859 41949  5788 38082 18507]
    | - Top 10 results similarity: [1.0000002  0.78101647 0.77775997 0.7771763  0.77591014 0.77581763
     0.77578723 0.77570754 0.77500904 0.77420104]
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

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000)

with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        vectors.append((t, id, 'test_'+str(id // 100)))
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

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
    //    reindex_if_conflict=False, scaler_bits=None
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  11
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.01311 s
    | - Query K: 10
    | - Query normalize: False
    | - Top 10 results index: [11 11  2 58 71 71 88 88 81 81]
    | - Top 10 results similarity: [1.         1.         0.7702445  0.76463264 0.764104   0.764104
     0.7637025  0.7637025  0.7620634  0.7620634 ]
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
        # Vectors need to be normalized before writing to the database.
        vectors.append((t, id, 'test_'+str(id // 100)))
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

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
    //    reindex_if_conflict=False, scaler_bits=None
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  11
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.01752 s
    | - Query K: 10
    | - Query normalize: False
    | - Top 10 results index: [11 11  2 25  4 19 21 29 30 13]
    | - Top 10 results similarity: [1.         1.         0.7702445  0.7628149  0.7586509  0.75613594
     0.7559968  0.7528333  0.74891967 0.7427169 ]
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

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, distance='L2')


with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    last_field = None
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        vectors.append((t, id, 'test_'+str(id // 100)))
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

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
    //    reindex_if_conflict=False, scaler_bits=None
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/'


      - Query sample id:  11
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.02081 s
    | - Query K: 10
    | - Query normalize: False
    | - Top 10 results index: [11 11  2 25  4 19 21 29 30  1]
    | - Top 10 results similarity: [0.         0.         0.6778725  0.68874544 0.69476485 0.6983754
     0.69857436 0.7030885  0.70863307 0.70987064]
    * - END OF REPORT -
    

