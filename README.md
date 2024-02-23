<div align="center">
  <h1>MinVectorDB</h1>
  <h3>A pure Python-implemented, lightweight, stateless, locally deployed vector database.</h3>
  <p>
    <a href="https://badge.fury.io/py/MinVectorDB"><img src="https://badge.fury.io/py/MinVectorDB.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/pyversions/MinVectorDB" alt="PyPI - Python Version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/l/MinVectorDB" alt="PyPI - License"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/dm/MinVectorDB" alt="PyPI - Downloads"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/format/MinVectorDB" alt="PyPI - Format"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/implementation/MinVectorDB" alt="PyPI - Implementation"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/wheel/MinVectorDB" alt="PyPI - Wheel"></a>
  </p>
</div>


> **WARNING**: MinVectorDB is actively being updated, and API backward compatibility is not guaranteed. You should use version numbers as a strong constraint during deployment to avoid unnecessary feature conflicts and errors.

*MinVectorDB* is a pure Python-implemented, lightweight, stateless vector, locally deployed database that offers clear and concise Python APIs, aimed at lowering the barrier to the use of vector databases. To make it more practical, we plan to add a range of features in the future, including but not limited to:

- **Improving global search performance**: By optimizing algorithms and data structures, we aim to accelerate the operation of searching across the entire database, allowing users to retrieve the vector data they need more quickly.
- **Enhancing cluster search performance with inverted indexes**: With the help of inverted index technology, we plan to optimize the cluster search process, thereby improving the efficiency and accuracy of searches.
- **Improving clustering algorithms**: By enhancing existing clustering algorithms, we hope to provide more accurate and efficient data clustering capabilities to support complex query requirements.
- **Supporting vector modifications and deletions**: To allow users to manage their data more flexibly, we will provide functionalities for vector modifications and deletions, making data updates and maintenance more convenient.
- **Implementing rollback strategies**: To enhance the robustness of the database and the security of the data, we will introduce rollback strategies, enabling users to easily revert to a previous state in case of erroneous operations or system failures.

The design focus of MinVectorDB is on achieving a 100% recall rate, meaning we prioritize ensuring recall accuracy, allowing users to accurately and reliably retrieve all relevant vector data. 
This design philosophy makes MinVectorDB particularly suitable for scenarios requiring latency to be completed within 100 milliseconds, rather than pursuing high-speed performance searches. 
We believe that with these upcoming features, MinVectorDB will be able to better meet users' diverse needs in vector data management and retrieval.

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

# clustering quality threshold
os.environ['MVDB_CLUSTERING_QUALITY_THRESHOLD'] = '0.3'  # default: 0.3

# bulk add batch size
os.environ['MVDB_BULK_ADD_BATCH_SIZE'] = '100000'  # default: 100000

# threshold for the amount of data to trigger reindexing each time new data is added.
os.environ['MVDB_REINDEX_CHECKING_SAMPLES'] = '10000' # default 10000

```


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
```

    MinVectorDB version is:  0.2.0


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

vectors = 5400_0000
vectors = 10_0000

# ===================================================================
# ========================= DEMO 1 ==================================
# ===================================================================
# Demo 1 -- Sequentially add vectors.
# Create a MinVectorDB instance.
display_markdown("*Demo 1* -- **Sequentially add vectors**", raw=True)

# distance can be 'L2' or 'cosine'
# index_mode can be 'FLAT' or 'IVF-FLAT', default is 'IVF-FLAT'
db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', distance='cosine', index_mode='IVF-FLAT', chunk_size=10000, use_cache=True)

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

query = db.head()[0]
query_id = db.head(returns='indices')[0]
query_field = db.head(returns='fields')[0]

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


    2024-02-23 17:12:43 - MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    100%|██████████| 100000/100000 [00:01<00:00, 51415.99vector/s]
    2024-02-23 17:12:45 - MinVectorDB - INFO - Saving chunk immediately...
    2024-02-23 17:12:45 - MinVectorDB - INFO - Saving id filter...
    2024-02-23 17:12:45 - MinVectorDB - INFO - Building index...
    2024-02-23 17:12:48 - MinVectorDB - INFO - The clustering quality is: -0.022148462012410164
    2024-02-23 17:12:48 - MinVectorDB - INFO - The clustering quality is not good, reindexing...
    2024-02-23 17:12:49 - MinVectorDB - INFO - Saving ann model...
    2024-02-23 17:12:49 - MinVectorDB - INFO - Saving ivf index...


    
    * [Insert data] Time cost 5.6480 s.
      - Query sample id:  6849
      - Query sample field:  
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.17522 s
    | - Query vector: [0.04436826 0.02498281 0.00122129 ... 0.01926835 0.04222433 0.03094637]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [ 6849 59690 55161 81104 65673 66663 53914 88507 21582 81396]
    | - Top 10 results similarity: [1.         0.78499    0.7844205  0.78379464 0.7836677  0.7836211
     0.7835463  0.78324085 0.78320646 0.78291345]
    * - END OF REPORT -
    


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
    for t in get_test_vectors((100000, 1024)):
        # Vectors need to be normalized before writing to the database.
        # t = t / np.linalg.norm(t) 
        vectors.append((t, id))
        # ID increments by 1 with each loop iteration.
        id += 1
        
#     # Here, normalization can be directly specified, achieving the same effect as `t = t / np.linalg.norm(t) `.
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


    2024-02-23 17:12:49 - MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    2024-02-23 17:12:54 - MinVectorDB - INFO - Saving chunk immediately...
    2024-02-23 17:12:54 - MinVectorDB - INFO - Saving id filter...


    
    * [Insert data] Time cost 4.2270 s.
      - Query sample id:  2
      - Query sample field:  
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.16281 s
    | - Query vector: [0.04493065 0.00245387 0.03883836 ... 0.02070636 0.05214242 0.03655052]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [    2 91745 34952 73172 16234 21556 56017  3534   440 36253]
    | - Top 10 results similarity: [0.99999994 0.7895216  0.78557634 0.7839494  0.78385794 0.78378147
     0.78375924 0.78356993 0.7831306  0.78296286]
    * - END OF REPORT -
    


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


    2024-02-23 17:12:54 - MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    2024-02-23 17:12:58 - MinVectorDB - INFO - Saving chunk immediately...
    2024-02-23 17:12:58 - MinVectorDB - INFO - Saving id filter...
    2024-02-23 17:12:58 - MinVectorDB - INFO - Building index...
    2024-02-23 17:13:00 - MinVectorDB - INFO - The clustering quality is: -0.022148462012410164
    2024-02-23 17:13:00 - MinVectorDB - INFO - The clustering quality is not good, reindexing...
    2024-02-23 17:13:02 - MinVectorDB - INFO - Saving ann model...
    2024-02-23 17:13:02 - MinVectorDB - INFO - Saving ivf index...


    
    * [Insert data] Time cost 7.8755 s.
      - Query sample id:  6849
      - Query sample field:  test_68
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.12755 s
    | - Query vector: [0.04436826 0.02498281 0.00122129 ... 0.01926835 0.04222433 0.03094637]
    | - Query K: 10
    | - Query fields: ['test_68']
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [6849 6837 6828 6884 6870 6822 6838 6898 6840 6883]
    | - Top 10 results similarity: [0.9999999  0.771406   0.7689647  0.7684603  0.76826066 0.767176
     0.7662252  0.7654011  0.7649375  0.76457   ]
    * - END OF REPORT -
    


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


    2024-02-23 17:13:02 - MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    2024-02-23 17:13:06 - MinVectorDB - INFO - Saving chunk immediately...
    2024-02-23 17:13:06 - MinVectorDB - INFO - Saving id filter...
    2024-02-23 17:13:06 - MinVectorDB - INFO - Building index...
    2024-02-23 17:13:08 - MinVectorDB - INFO - The clustering quality is: -0.022148462012410164
    2024-02-23 17:13:08 - MinVectorDB - INFO - The clustering quality is not good, reindexing...
    2024-02-23 17:13:10 - MinVectorDB - INFO - Saving ann model...
    2024-02-23 17:13:10 - MinVectorDB - INFO - Saving ivf index...


    
    * [Insert data] Time cost 8.0041 s.
      - Query sample id:  6849
      - Query sample field:  test_68
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.10887 s
    | - Query vector: [0.04436826 0.02498281 0.00122129 ... 0.01926835 0.04222433 0.03094637]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: [6829, 6830, 6831, 6832, 6833, 6834, 6835, 6836, 6837, 6838, 6839, 6840, 6841, 6842, 6843, 6844, 6845, 6846, 6847, 6848, 6849, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865, 6866, 6867, 6868]
    | - Top 10 results index: [6849 6837 6838 6840 6852 6868 6866 6829 6862 6850]
    | - Top 10 results similarity: [0.9999999  0.771406   0.7662252  0.7649375  0.7639904  0.7635061
     0.76331264 0.75842524 0.75825465 0.75766325]
    * - END OF REPORT -
    


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

db = MinVectorDB(dim=1024, database_path='test_min_vec.mvdb', chunk_size=10000)

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
# If there is no intersection between subset_indices and fields, there will be no result. 
# If there is an intersection, the query results within the intersection will be returned.
res = db.query(query, k=10, subset_indices=list(range(query_id-20, query_id + 20)), fields=['test_0', 'test_2'])
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


    2024-02-23 17:13:10 - MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    2024-02-23 17:13:14 - MinVectorDB - INFO - Saving chunk immediately...
    2024-02-23 17:13:14 - MinVectorDB - INFO - Saving id filter...
    2024-02-23 17:13:14 - MinVectorDB - INFO - Building index...
    2024-02-23 17:13:16 - MinVectorDB - INFO - The clustering quality is: -0.022148462012410164
    2024-02-23 17:13:16 - MinVectorDB - INFO - The clustering quality is not good, reindexing...
    2024-02-23 17:13:18 - MinVectorDB - INFO - Saving ann model...
    2024-02-23 17:13:18 - MinVectorDB - INFO - Saving ivf index...


    
    * [Insert data] Time cost 7.7911 s.
      - Query sample id:  6849
      - Query sample field:  test_68
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.12052 s
    | - Query vector: [0.04436826 0.02498281 0.00122129 ... 0.01926835 0.04222433 0.03094637]
    | - Query K: 10
    | - Query fields: ['test_0', 'test_2']
    | - Query normalize: False
    | - Query subset_indices: [6829, 6830, 6831, 6832, 6833, 6834, 6835, 6836, 6837, 6838, 6839, 6840, 6841, 6842, 6843, 6844, 6845, 6846, 6847, 6848, 6849, 6850, 6851, 6852, 6853, 6854, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865, 6866, 6867, 6868]
    | - Top 10 results index: []
    | - Top 10 results similarity: []
    * - END OF REPORT -
    

