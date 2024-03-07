<div align="center">
  <h1>MinVectorDB</h1>
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

Additionally, we are introducing a query caching feature, with a default cache for the most recent 10,000 query results. In cases where a query does not hit the cache, the system will calculate the cosine similarity between the given vector and cached vectors. If the similarity is greater than 0.8, it will return the result of the closest cached vector directly.

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

# bulk add batch size
os.environ['MVDB_BULK_ADD_BATCH_SIZE'] = '100000'  # default: 100000

# cache size
os.environ['MVDB_CACHE_SIZE'] = '10000'  # default: 10000

# cosine similarity threshold for cache result matching 
os.environ['MVDB_COSINE_SIMILARITY_THRESHOLD'] = '0.9'  # 'None' for disable this feature, default to 0.8

# computing platform, can be set to platforms supported by torch.device 
os.environ['MVDB_COMPUTE_DEVICE'] = 'cpu' # default to 'cpu', torch.device
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
     - MVDB_BULK_ADD_BATCH_SIZE: 100000
     - MVDB_CACHE_SIZE: 10000
     - MVDB_COSINE_SIMILARITY_THRESHOLD: 0.9
     - MVDB_COMPUTE_DEVICE: cpu



```python
# define the display function, this is optional in actual use.
try:
    from IPython.display import display_markdown
except ImportError:
    def display_markdown(text, raw=True):
        print(text)
```

### Sequentially add vectors.


```python
import numpy as np
from tqdm import tqdm

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

vectors = 10_0000

display_markdown("*Demo 1* -- **Sequentially add vectors**", raw=True)

# distance can be 'L2' or 'cosine'
# index_mode can be 'FLAT' or 'IVF-FLAT', default is 'IVF-FLAT'
db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', distance='cosine',
                 index_mode='IVF-FLAT', chunk_size=100000, use_cache=False, scaler_bits=8)

np.random.seed(23)

def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

timer.start()
# ========== Use automatic commit statements. Recommended. =============
# You can perform this operation multiple times, and the data will be appended to the database.
with db.insert_session():
    # Define the initial ID.
    id = 0
    for t in tqdm(get_test_vectors((vectors, 1024)), total=vectors, unit="vector"):
        if id == 0:
            query = t / np.linalg.norm(t)
            query_id = 0
            query_field = None
            # break
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        # Here, normalization can be directly specified, achieving the same effect as the previous sentence.
        db.add_item(t, index=id, normalize=True, save_immediately=False)
        
        # ID increments by 1 with each loop iteration.
        id += 1

# ============== Or use manual commit statements. =================
# id = 0
# for t in get_test_vectors((vectors, 1024)):
#     # Vectors need to be normalized before writing to the database.
#     # t = t / np.linalg.norm(t) 
#     # Here, normalization can be directly specified, achieving the same effect as the previous sentence.
#     db.add_item(t, index=id, normalize=True)
    
#     # ID increments by 1 with each loop iteration.
#     id += 1
# db.commit()

# 45:40 -> 36:06
print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

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


*Demo 1* -- **Sequentially add vectors**


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=100000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float64', use_cache=False, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/', database file path: 'test_min_vec/test_min_vec.mvdb'
    100%|██████████| 100000/100000 [00:02<00:00, 39194.40vector/s]
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Saving scalar quantization model...
    MinVectorDB - INFO - Saving fields mapper...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 5.0081 s.
      - Query sample id:  0
      - Query sample field:  None
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.02220 s
    | - Query vector: [0.02898663 0.05306277 0.04289231 ... 0.0143056  0.01658325 0.04808333]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [    0 82013 72370 69565 59036 98630 75948 27016  6658 38440]
    | - Top 10 results similarity: [0.9910508  0.76508979 0.7644585  0.76421338 0.76404578 0.76316689
     0.76315354 0.76276408 0.7627276  0.76228611]
    * - END OF REPORT -


    MinVectorDB - INFO - Saving ann model...


### Bulk add vectors


```python
import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

display_markdown("*Demo 2* -- **Bulk add vectors**", raw=True)

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, n_cluster=8, index_mode='FLAT')

np.random.seed(23)

def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

timer.start()

# You can perform this operation multiple times, and the data will be appended to the database.
with db.insert_session():  
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100003, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id))
        # ID increments by 1 with each loop iteration.
        id += 1
        
    # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
    db.bulk_add_items(vectors, normalize=True, save_immediately=False)

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[2]
query_id = db.head(10, returns='indices')[2]
query_field = db.head(10, returns='fields')[2]

res = db.query(query, k=10)
print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)


timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 2* -- **Bulk add vectors**


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', index_mode='FLAT', 
    //    dtypes='float64', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/', database file path: 'test_min_vec/test_min_vec.mvdb'


    
    * [Insert data] Time cost 4.8161 s.
      - Query sample id:  2
      - Query sample field:  
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100003, 1024)
    | - Query time: 0.13812 s
    | - Query vector: [0.04471495 0.00244354 0.03863426 ... 0.02068288 0.05212677 0.03638766]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [    2 91745 34952 73172 56017 16234 21556  3534   440 20005]
    | - Top 10 results similarity: [0.99395035 0.7834744  0.77941584 0.77786743 0.77772982 0.77767259
     0.77766295 0.77743496 0.77707587 0.77678738]
    * - END OF REPORT -


    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Saving scalar quantization model...
    MinVectorDB - INFO - Saving fields mapper...


### Use field to improve Searching Recall


```python
import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

display_markdown("*Demo 3* -- **Use field to improve Searching Recall**", raw=True)

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000)

np.random.seed(23)


def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

timer.start()
with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id, 'test_'+str(id // 100)))
        # ID increments by 1 with each loop iteration.
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

res = db.query(query, k=10, fields=[query_field])

print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 3* -- **Use field to improve Searching Recall**


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float64', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/', database file path: 'test_min_vec/test_min_vec.mvdb'
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Saving scalar quantization model...
    MinVectorDB - INFO - Saving fields mapper...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 5.9627 s.
      - Query sample id:  77
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.03002 s
    | - Query vector: [0.04427223 0.05085017 0.05055556 ... 0.03278401 0.00088751 0.03528501]
    | - Query K: 10
    | - Query fields: ['test_0']
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [77 24 69 71 97 94 81 48 13 67]
    | - Top 10 results similarity: [0.98169604 0.75300974 0.74258977 0.74204745 0.74182057 0.74018082
     0.74005663 0.74002714 0.73858353 0.7351924 ]
    * - END OF REPORT -


    MinVectorDB - INFO - Saving ann model...


### Use subset_indices to narrow down the search range


```python
import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

display_markdown("*Demo 4* -- **Use subset_indices to narrow down the search range**", raw=True)

timer.start()

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, index_mode='IVF-FLAT')

np.random.seed(23)


def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id, 'test_'+str(id // 100)))
        # ID increments by 1 with each loop iteration.
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

# You may define both 'subset_indices' and 'fields'
res = db.query(query, k=10, subset_indices=list(range(query_id - 20, query_id + 20)))
print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 4* -- **Use subset_indices to narrow down the search range**


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float64', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/', database file path: 'test_min_vec/test_min_vec.mvdb'
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Saving scalar quantization model...
    MinVectorDB - INFO - Saving fields mapper...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 5.9423 s.
      - Query sample id:  77
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.02989 s
    | - Query vector: [0.04427223 0.05085017 0.05055556 ... 0.03278401 0.00088751 0.03528501]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
    | - Top 10 results index: [77 69 71 94 81 67 65 62 88 59]
    | - Top 10 results similarity: [0.98169604 0.74258977 0.74204745 0.74018082 0.74005663 0.7351924
     0.73376908 0.73177266 0.73129947 0.72831082]
    * - END OF REPORT -


    MinVectorDB - INFO - Saving ann model...


### Conduct searches by specifying both subset_indices and fields simultaneously.


```python
import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

display_markdown("*Demo 5* -- **Conduct searches by specifying both subset_indices and fields simultaneously**", raw=True)

timer.start()

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000)

np.random.seed(23)


def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    last_field = None
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id, 'test_'+str(id // 100)))
        # if 'test_'+str(id // 100) != last_field:
        #     print('test_'+str(id // 100))
        #     last_field = 'test_'+str(id // 100)
        # ID increments by 1 with each loop iteration.
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

# You may define both 'subset_indices' and 'fields'
# If there is no intersection between subset_indices and fields, there will be no result. 
# If there is an intersection, the query results within the intersection will be returned.
res = db.query(query, k=10, subset_indices=list(range(query_id-20, query_id + 20)), fields=['test_0', 'test_1'])
print("  - Query sample id: ", query_id)
print("  - Query sample field: ", query_field)

# Query report
print(db.query_report_)

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 5* -- **Conduct searches by specifying both subset_indices and fields simultaneously**


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', index_mode='IVF-FLAT', 
    //    dtypes='float64', use_cache=True, 
    //    reindex_if_conflict=False, scaler_bits=8
    
    MinVectorDB - INFO - Initializing database folder path: 'test_min_vec/', database file path: 'test_min_vec/test_min_vec.mvdb'
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Saving scalar quantization model...
    MinVectorDB - INFO - Saving fields mapper...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 5.9817 s.
      - Query sample id:  77
      - Query sample field:  test_0
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.02898 s
    | - Query vector: [0.04427223 0.05085017 0.05055556 ... 0.03278401 0.00088751 0.03528501]
    | - Query K: 10
    | - Query fields: ['test_0', 'test_1']
    | - Query normalize: False
    | - Query subset_indices: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
    | - Top 10 results index: [77 69 71 94 81 67 65 62 88 59]
    | - Top 10 results similarity: [0.98169604 0.74258977 0.74204745 0.74018082 0.74005663 0.7351924
     0.73376908 0.73177266 0.73129947 0.72831082]
    * - END OF REPORT -


    MinVectorDB - INFO - Saving ann model...

