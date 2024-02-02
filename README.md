# MinVectorDB

<b>MinVectorDB</b> is a simple vector storage and query database implementation that offers clear and concise Python APIs, aimed at lowering the barrier to using vector databases. More practical features will be added in the future.
<br>
It is important to note that MinVectorDB is not designed for efficiency, and therefore does not include built-in algorithms like approximate nearest neighbors for efficient searching.
<br>
Originally created to demonstrate large language model demos with a goal of 100% recall.
<br>
<br>
<b>MinVectorDB</b> 是简易实现的向量存储和查询数据库，提供简洁明了的python API，旨在降低向量数据库的使用门槛。未来将添加更多实用功能。
<br>
需要注意的是，MinVectorDB并非为追求效率而生，因此，并没有内置近似最近邻等高效查找算法。
<br>
它起源于作者需要演示大语言模型Demo的契机，为了追求100%召回率而设计。

## TODO
- [x] Sequentially add vectors.
- [x] Bulk add vectors.
- [x] Use field to improve Searching Recall.
- [x] Use subset_indices to narrow down the search range.
- [ ] Add rollback functionality.
- [ ] Add multi-threaded writing functionality.
- [ ] Add multi-threaded query functionality.

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
os.environ['MVDB_LOG_LEVEL'] = 'INFO'  # default: INFO

# log path
os.environ['MVDB_LOG_PATH'] = './min_vec_db.log'  # default: None

# whether to truncate log file
os.environ['MVDB_TRUNCATE_LOG'] = 'True'  # default: True

# whether to add time to log
os.environ['MVDB_LOG_WITH_TIME'] = 'False'  # default: False

# clustering settings
# kmeans epochs
os.environ['MVDB_KMEANS_EPOCHS'] = '500'  # default: 500

# clustering quality threshold
os.environ['MVDB_CLUSTERING_QUALITY_THRESHOLD'] = '0.3'  # default: 0.3

# bulk add batch size
os.environ['MVDB_BULK_ADD_BATCH_SIZE'] = '10000'  # default: 10000

```


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
```

    MinVectorDB version is:  0.1.5


### Sequentially add vectors.


```python
try:
    from IPython.display import display_markdown
except ImportError:
    def display_markdown(text, raw=True):
        print(text)

import numpy as np
from tqdm import tqdm

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

vectors = 100000


# ===================================================================
# ========================= DEMO 1 ==================================
# ===================================================================
# Demo 1 -- Sequentially add vectors.
# Create a MinVectorDB instance.
display_markdown("*Demo 1* -- **Sequentially add vectors**", raw=True)

# distance can be 'L2' or 'cosine'
# storage_mode can be 'disk' or 'memory'
db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=vectors // 10, device='cpu', distance='cosine', storage_mode='disk')

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
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        # Here, normalization can be directly specified, achieving the same effect as the previous sentence.
        # print("id: ", id)
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

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]

timer.middle_point()
res = db.query(query, k=10)

query_time_cost = timer.last_timestamp_diff()
print("  - Database shape: ", db.shape)
print("  - Query vector: ", query)
print("  - Query id: ", query_id)
print("  - Database index of top 10 results: ", res[0][:10])
print("  - Similarity of top 10 results: ", res[1][:10])
print(f"\n* [Query data] Time cost {query_time_cost :>.4f} s.")
timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 1* -- **Sequentially add vectors**


    2024-01-29 18:50:40 - MinVectorDB - INFO - Initializing MinVectorDB with dim=1024, database_path='test_min_vec.mvdb', n_cluster=8, chunk_size=10000, dtypes=<class 'numpy.float32'>, distance='cosine', bloom_filter_size=100000000, device='cpu', storage_mode='disk'
    100%|██████████| 100000/100000 [00:01<00:00, 54008.05vector/s]
    2024-01-29 18:50:45 - MinVectorDB - INFO - The clustering quality is: -0.022005783393979073
    2024-01-29 18:50:45 - MinVectorDB - INFO - The clustering quality is not good, reindexing...


    
    * [Insert data] Time cost 7.3625 s.
      - Database shape:  (100000, 1024)
      - Query vector:  [0.02898663 0.05306277 0.04289231 ... 0.0143056  0.01658326 0.04808333]
      - Query id:  0
      - Database index of top 10 results:  [    0 67927 53447 64134 13859 41949  5788 38082 18507 82013]
      - Similarity of top 10 results:  [1.0000002  0.78101647 0.77775997 0.77591014 0.77581763 0.77578723
     0.77570754 0.77500904 0.77420104 0.77413327]
    
    * [Query data] Time cost 0.0035 s.


### Bulk add vectors


```python
try:
    from IPython.display import display_markdown
except ImportError:
    def display_markdown(text, raw=True):
        print(text)

import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

# ===================================================================
# ========================= DEMO 2 ==================================
# ===================================================================
# Demo 2 -- Bulk add vectors.
display_markdown("*Demo 2* -- **Bulk add vectors**", raw=True)

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, device='cpu')

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
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id))
        # ID increments by 1 with each loop iteration.
        id += 1
        
#     # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
    db.bulk_add_items(vectors, normalize=True, save_immediately=False)

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[0]
query_id = db.head(10, returns='indices')[0]

timer.middle_point()
res = db.query(query, k=10)
query_time_cost = timer.last_timestamp_diff()
print("  - Database shape: ", db.shape)
print("  - Query vector: ", query)
print("  - Query id: ", query_id)
print("  - Database index of top 10 results: ", res[0][:10])
print("  - Similarity of top 10 results: ", res[1][:10])
print(f"\n* [Query data] Time cost {query_time_cost :>.4f} s.")


timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 2* -- **Bulk add vectors**


    2024-01-29 18:50:54 - MinVectorDB - INFO - Initializing MinVectorDB with dim=1024, database_path='test_min_vec.mvdb', n_cluster=8, chunk_size=10000, dtypes=<class 'numpy.float32'>, distance='cosine', bloom_filter_size=100000000, device='cpu', storage_mode='disk'
    2024-01-29 18:51:00 - MinVectorDB - INFO - The clustering quality is: -0.022005783393979073
    2024-01-29 18:51:00 - MinVectorDB - INFO - The clustering quality is not good, reindexing...


    
    * [Insert data] Time cost 9.2694 s.
      - Database shape:  (100000, 1024)
      - Query vector:  [0.02898663 0.05306277 0.04289231 ... 0.0143056  0.01658326 0.04808333]
      - Query id:  0
      - Database index of top 10 results:  [    0 67927 53447 64134 13859 41949  5788 38082 18507 82013]
      - Similarity of top 10 results:  [1.0000002  0.78101647 0.77775997 0.77591014 0.77581763 0.77578723
     0.77570754 0.77500904 0.77420104 0.77413327]
    
    * [Query data] Time cost 0.0028 s.


### Use field to improve Searching Recall


```python
try:
    from IPython.display import display_markdown
except ImportError:
    def display_markdown(text, raw=True):
        print(text)

import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

# ===================================================================
# ========================= DEMO 3 ==================================
# ===================================================================
# Demo 3 -- Use field to improve Searching Recall
display_markdown("*Demo 3* -- **Use field to improve Searching Recall**", raw=True)

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, device='cpu')

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

timer.middle_point()

res = db.query(query, k=10, fields=[query_field])
query_time_cost = timer.last_timestamp_diff()
print("  - Database shape: ", db.shape)
print("  - Query vector: ", query)
print("  - Query id: ", query_id)
print("  - Query field: ", query_field)
print("  - Database index of top 10 results: ", res[0][:10])
print("  - Similarity of top 10 results: ", res[1][:10])
print(f"\n* [Query data] Time cost {query_time_cost :>.4f} s.")

timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 3* -- **Use field to improve Searching Recall**


    2024-01-29 18:51:03 - MinVectorDB - INFO - Initializing MinVectorDB with dim=1024, database_path='test_min_vec.mvdb', n_cluster=8, chunk_size=10000, dtypes=<class 'numpy.float32'>, distance='cosine', bloom_filter_size=100000000, device='cpu', storage_mode='disk'
    2024-01-29 18:51:10 - MinVectorDB - INFO - The clustering quality is: -0.022005783393979073
    2024-01-29 18:51:10 - MinVectorDB - INFO - The clustering quality is not good, reindexing...


    
    * [Insert data] Time cost 9.4631 s.
      - Database shape:  (100000, 1024)
      - Query vector:  [0.02898663 0.05306277 0.04289231 ... 0.0143056  0.01658326 0.04808333]
      - Query id:  0
      - Query field:  test_0
      - Database index of top 10 results:  [ 0 66 21 60 91 43 84 52 14 28]
      - Similarity of top 10 results:  [1.         0.75745714 0.75445515 0.75418174 0.75279343 0.7514601
     0.75065786 0.7492904  0.7480291  0.7465518 ]
    
    * [Query data] Time cost 0.0349 s.


### Use subset_indices to narrow down the search range


```python
try:
    from IPython.display import display_markdown
except ImportError:
    def display_markdown(text, raw=True):
        print(text)

import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

# ===================================================================
# ========================= DEMO 4 ==================================
# ===================================================================
# Demo 4 -- Use subset_indices to narrow down the search range
display_markdown("*Demo 4* -- **Use subset_indices to narrow down the search range**", raw=True)

timer.start()

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, device='cpu')

np.random.seed(23)


def get_test_vectors(shape):
    for i in range(shape[0]):
        yield np.random.random(shape[1])

with db.insert_session():     
    # Define the initial ID.
    id = 0
    vectors = []
    for t in get_test_vectors((100001, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id, 'test_'+str(id // 100)))
        # ID increments by 1 with each loop iteration.
        id += 1
        
    db.bulk_add_items(vectors, normalize=True)

print(f"\n* [Insert data] Time cost {timer.last_timestamp_diff():>.4f} s.")

query = db.head(10)[1]
query_id = db.head(10, returns='indices')[0]
query_field = db.head(10, returns='fields')[0]

timer.middle_point()

# You may define both 'subset_indices' and 'fields'
res = db.query(query, k=10, subset_indices=list(range(0, query_id + 1000)))
query_time_cost = timer.last_timestamp_diff()
print("  - Database shape: ", db.shape)
print("  - Query vector: ", query)
print("  - Query id: ", query_id)
print("  - Query field: ", query_field)
print("  - Database index of top 10 results: ", res[0][:10])
print("  - Similarity of top 10 results: ", res[1][:10])
print(f"\n* [Query data] Time cost {query_time_cost :>.4f} s.")

timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 4* -- **Use subset_indices to narrow down the search range**


    2024-01-29 18:51:13 - MinVectorDB - INFO - Initializing MinVectorDB with dim=1024, database_path='test_min_vec.mvdb', n_cluster=8, chunk_size=10000, dtypes=<class 'numpy.float32'>, distance='cosine', bloom_filter_size=100000000, device='cpu', storage_mode='disk'
    2024-01-29 18:51:20 - MinVectorDB - INFO - The clustering quality is: -0.020339230075478554
    2024-01-29 18:51:20 - MinVectorDB - INFO - The clustering quality is not good, reindexing...


    
    * [Insert data] Time cost 9.5953 s.
      - Database shape:  (100001, 1024)
      - Query vector:  [0.02089841 0.01518081 0.03332409 ... 0.01748978 0.01588354 0.00640359]
      - Query id:  0
      - Query field:  test_0
      - Database index of top 10 results:  [431   0 842  66 764 788 130 313 555 863]
      - Similarity of top 10 results:  [0.99999994 0.7651854  0.75583184 0.75225186 0.75125563 0.750734
     0.7507172  0.748871   0.7422094  0.7416878 ]
    
    * [Query data] Time cost 0.0022 s.


### Conduct searches by specifying both subset_indices and fields simultaneously.


```python
try:
    from IPython.display import display_markdown
except ImportError:
    def display_markdown(text, raw=True):
        print(text)

import numpy as np

from spinesUtils.timer import Timer
from min_vec import MinVectorDB

timer = Timer()

# ===================================================================
# ========================= DEMO 5 ==================================
# ===================================================================
display_markdown("*Demo 5* -- **Conduct searches by specifying both subset_indices and fields simultaneously**", raw=True)

timer.start()

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000, device='cpu')

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

timer.middle_point()

# You may define both 'subset_indices' and 'fields'
res = db.query(query, k=10, subset_indices=list(range(0, query_id + 10000)), fields=['test_0', 'test_2'])
query_time_cost = timer.last_timestamp_diff()
print("  - Database shape: ", db.shape)
print("  - Query vector: ", query)
print("  - Query id: ", query_id)
print("  - Query field: ", query_field)
print("  - Database index of top 10 results: ", res[0][:10])
print("  - Similarity of top 10 results: ", res[1][:10])
print(f"\n* [Query data] Time cost {query_time_cost :>.4f} s.")

timer.end()

# This sentence is for demo demonstration purposes, 
# to clear the currently created .mvdb files from the database, 
# but this is optional in actual use.
db.delete()
```


*Demo 5* -- **Conduct searches by specifying both subset_indices and fields simultaneously**


    2024-01-29 18:51:22 - MinVectorDB - INFO - Initializing MinVectorDB with dim=1024, database_path='test_min_vec.mvdb', n_cluster=8, chunk_size=10000, dtypes=<class 'numpy.float32'>, distance='cosine', bloom_filter_size=100000000, device='cpu', storage_mode='disk'
    2024-01-29 18:51:29 - MinVectorDB - INFO - The clustering quality is: -0.022005783393979073
    2024-01-29 18:51:29 - MinVectorDB - INFO - The clustering quality is not good, reindexing...


    
    * [Insert data] Time cost 9.5274 s.
      - Database shape:  (100000, 1024)
      - Query vector:  [0.02898663 0.05306277 0.04289231 ... 0.0143056  0.01658326 0.04808333]
      - Query id:  0
      - Query field:  test_0
      - Database index of top 10 results:  [  0  66 293 281  84 224 235 262  28  68]
      - Similarity of top 10 results:  [1.         0.75745714 0.7557113  0.7524824  0.75065786 0.74784184
     0.7472197  0.7467411  0.7465518  0.7465167 ]
    
    * [Query data] Time cost 0.0217 s.

