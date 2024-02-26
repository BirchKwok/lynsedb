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

    MinVectorDB version is:  0.2.2
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
db = MinVectorDB(dim=1024, database_path='/Volumes/西数SSD/test_min_vec.mvdb', distance='cosine', index_mode='FLAT', chunk_size=100000, use_cache=True)

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


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='/Volumes/西数SSD/test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=100000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    100%|██████████| 100000/100000 [00:02<00:00, 42802.86vector/s]
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...


    
    * [Insert data] Time cost 2.4119 s.
      - Query sample id:  0
      - Query sample field:  
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.14636 s
    | - Query vector: [0.02898663 0.05306277 0.04289231 ... 0.0143056  0.01658326 0.04808333]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [    0 67927 53447 47665 64134 13859 41949  5788 38082 18507]
    | - Top 10 results similarity: [1.0000002  0.78101647 0.77775997 0.7771763  0.77591014 0.77581763
     0.77578723 0.77570754 0.77500904 0.77420104]
    * - END OF REPORT -
    


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
    for t in get_test_vectors((100000, 1024)):
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
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...


    
    * [Insert data] Time cost 4.1953 s.
      - Query sample id:  2
      - Query sample field:  
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.15137 s
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
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 6.3419 s.
      - Query sample id:  150
      - Query sample field:  test_1
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.12271 s
    | - Query vector: [0.03762956 0.05180147 0.04209524 ... 0.04615058 0.05285349 0.05330994]
    | - Query K: 10
    | - Query fields: ['test_1']
    | - Query normalize: False
    | - Query subset_indices: None
    | - Top 10 results index: [150 122 118 199 177 130 175 194 126 168]
    | - Top 10 results similarity: [1.         0.7638782  0.76372206 0.7636392  0.7589303  0.75863093
     0.7581293  0.7578685  0.75781167 0.75737965]
    * - END OF REPORT -
    


    MinVectorDB - INFO - Saving ann model...
    MinVectorDB - INFO - Saving ivf index...


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
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 6.2138 s.
      - Query sample id:  150
      - Query sample field:  test_1
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.10610 s
    | - Query vector: [0.03762956 0.05180147 0.04209524 ... 0.04615058 0.05285349 0.05330994]
    | - Query K: 10
    | - Query fields: None
    | - Query normalize: False
    | - Query subset_indices: [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169]
    | - Top 10 results index: [150 130 168 167 141 144 161 166 162 157]
    | - Top 10 results similarity: [1.         0.75863093 0.75737965 0.75729394 0.75688565 0.7568122
     0.75455916 0.7535342  0.7531498  0.7517347 ]
    * - END OF REPORT -
    


    MinVectorDB - INFO - Saving ann model...
    MinVectorDB - INFO - Saving ivf index...


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


    MinVectorDB - INFO - Initializing MinVectorDB with: 
    //    dim=1024, database_path='test_min_vec.mvdb', 
    //    n_cluster=8, chunk_size=10000,
    //    distance='cosine', bloom_filter_size=100000000, 
    //    index_mode='IVF-FLAT', dtypes='float32',
    //    use_cache=True, reindex_if_conflict=False
    
    MinVectorDB - INFO - Saving chunk immediately...
    MinVectorDB - INFO - Saving id filter...
    MinVectorDB - INFO - Building index...


    
    * [Insert data] Time cost 6.1384 s.
      - Query sample id:  150
      - Query sample field:  test_1
    
    * - MOST RECENT QUERY REPORT -
    | - Database shape: (100000, 1024)
    | - Query time: 0.11414 s
    | - Query vector: [0.03762956 0.05180147 0.04209524 ... 0.04615058 0.05285349 0.05330994]
    | - Query K: 10
    | - Query fields: ['test_0', 'test_2']
    | - Query normalize: False
    | - Query subset_indices: [130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169]
    | - Top 10 results index: []
    | - Top 10 results similarity: []
    * - END OF REPORT -
    


    MinVectorDB - INFO - Saving ann model...
    MinVectorDB - INFO - Saving ivf index...

