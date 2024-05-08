<div align="center">
  <a href="https://github.com/BirchKwok/MinVectorDB"><img src="https://github.com/BirchKwok/MinVectorDB/blob/main/pic/logo.png" alt="MinVectorDB" style="max-width: 20%; height: auto;"></a>
  <h3>A pure Python-implemented, lightweight, server-optional, multi-end compatible, vector database deployable locally or remotely.</h3>
  <p>
    <a href="https://badge.fury.io/py/MinVectorDB"><img src="https://badge.fury.io/py/MinVectorDB.svg" alt="PyPI version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/pyversions/MinVectorDB" alt="PyPI - Python Version"></a>
    <a href="https://pypi.org/project/MinVectorDB/"><img src="https://img.shields.io/pypi/l/MinVectorDB" alt="PyPI - License"></a>
    <a href="https://github.com/BirchKwok/MinVectorDB/actions/workflows/python-tests.yml"><img src="https://github.com/BirchKwok/MinVectorDB/actions/workflows/python-tests.yml/badge.svg" alt="Python testing"></a>
    <a href="https://github.com/BirchKwok/MinVectorDB/actions/workflows/docker-tests.yml"><img src="https://github.com/BirchKwok/MinVectorDB/actions/workflows/docker-tests.yml/badge.svg" alt="Docker build"></a>
  </p>
</div>

⚡ **Server-optional, simple parameters, simple API.**

⚡ **Fast, memory-efficient, easily scales to millions of vectors.**

⚡ **Supports cosine similarity and L2 distance, uses FLAT for exhaustive search or IVF-FLAT for inverted indexing.**

⚡ **Friendly caching technology stores recently queried vectors for accelerated access.**

⚡ **Based on a generic Python software stack, platform-independent, highly versatile.**

> **WARNING**: MinVectorDB is actively being updated, and API backward compatibility is not guaranteed. You should use version numbers as a strong constraint during deployment to avoid unnecessary feature conflicts and errors.
> **Although our goal is to enable brute force search or inverted indexing on billion-scale vectors, we currently still recommend using it on a scale of millions of vectors or less for the best experience.**

*MinVectorDB* is a vector database implemented purely in Python, designed to be lightweight, server-optional, and easy to deploy locally or remotely. It offers straightforward and clear Python APIs, aiming to lower the entry barrier for using vector databases. In response to user needs and to enhance its practicality, we are planning to introduce new features, including but not limited to:

- **Optimizing Global Search Performance**: We are focusing on algorithm and data structure enhancements to speed up searches across the database, enabling faster retrieval of vector data.
- **Enhancing Cluster Search with Inverted Indexes**: Utilizing inverted index technology, we aim to refine the cluster search process for better search efficiency and precision.
- **Refining Clustering Algorithms**: By improving our clustering algorithms, we intend to offer more precise and efficient data clustering to support complex queries.
- **Facilitating Vector Modifications and Deletions**: We will introduce features to modify and delete vectors, allowing for more flexible data management.

MinVectorDB focuses on achieving 100% recall, prioritizing recall accuracy over high-speed search performance. This approach ensures that users can reliably retrieve all relevant vector data, making MinVectorDB particularly suitable for applications that require responses within hundreds of milliseconds.

- [x] **Now supports HTTP API and Python local code API.**
- [X] **Now supports Docker deployment.**
- [X] **Now supports vector id and field filtering.**
- [X] **Now supports transaction management; if a commit fails, it will automatically roll back.**

## Install Client API package (Mandatory)

```shell
pip install MinVectorDB
```

## If you wish to use Docker (Optional)

```shell
docker pull birchkwok/minvectordb:latest
```

## Qucik Start


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
```

    MinVectorDB version is:  0.3.3


## Initialize Database

MinVectorDB now supports HTTP API and Python local code API. 


The HTTP API mode requires starting an HTTP server beforehand. You have two options: 
- start directly.
  
  For direct startup, the default port is 7637. You can run the following command in the terminal to start the service:
```shell
min_vec run --host localhost --port 7637
```

- within Docker
  
  In Docker, You can run the following command in the terminal to start the service:
```shell
docker run -p 7637:7637 birchkwok/minvectordb:latest
```

```python
from min_vec import MinVectorDB

# This method is for the Python local code API, recommended only for CI/CD testing or single-user local use.
# Specify database root directory
my_db = MinVectorDB('my_vec_db')  # Judgment condition, root_path does not start with http or https
# or
# Use the HTTP API mode, it is suitable for use in production environments.
my_db = MinVectorDB("http://localhost:7637")
```


```python
from min_vec import MinVectorDB

# For direct startup
my_db = MinVectorDB("http://localhost:7637")
```

### create a collection


```python
collection = my_db.require_collection("test_collection", 4, drop_if_exists=True, scaler_bits=8)
```

### Add vectors

When inserting vectors, collection requires manually running the `commit` function or inserting within the `insert_session` function context manager, which will run the `commit` function in the background.


```python
with collection.insert_session():
    id = collection.add_item(vector=[0.01, 0.34, 0.74, 0.31], id=1, field={'field': 'test_1', 'order': 0})   # id = 0
    id = collection.add_item(vector=[0.36, 0.43, 0.56, 0.12], id=2, field={'field': 'test_1', 'order': 1})   # id = 1
    id = collection.add_item(vector=[0.03, 0.04, 0.10, 0.51], id=3, field={'field': 'test_2', 'order': 2})   # id = 2
    id = collection.add_item(vector=[0.11, 0.44, 0.23, 0.24], id=4, field={'field': 'test_2', 'order': 3})   # id = 3
    id = collection.add_item(vector=[0.91, 0.43, 0.44, 0.67], id=5, field={'field': 'test_2', 'order': 4})   # id = 4
    id = collection.add_item(vector=[0.92, 0.12, 0.56, 0.19], id=6, field={'field': 'test_3', 'order': 5})   # id = 5
    id = collection.add_item(vector=[0.18, 0.34, 0.56, 0.71], id=7, field={'field': 'test_1', 'order': 6})   # id = 6
    id = collection.add_item(vector=[0.01, 0.33, 0.14, 0.31], id=8, field={'field': 'test_2', 'order': 7})   # id = 7
    id = collection.add_item(vector=[0.71, 0.75, 0.91, 0.82], id=9, field={'field': 'test_3', 'order': 8})   # id = 8
    id = collection.add_item(vector=[0.75, 0.44, 0.38, 0.75], id=10, field={'field': 'test_1', 'order': 9})  # id = 9

# If you do not use the insert_session function, you need to manually call the commit function to submit the data
# collection.commit()
```


```python
# or use the bulk_add_items function
# with collection.insert_session():
#     ids = collection.bulk_add_items([([0.01, 0.34, 0.74, 0.31], 0, {'field': 'test_1', 'order': 0}), 
#                                      ([0.36, 0.43, 0.56, 0.12], 1, {'field': 'test_1', 'order': 1}), 
#                                      ([0.03, 0.04, 0.10, 0.51], 2, {'field': 'test_2', 'order': 2}),
#                                      ([0.11, 0.44, 0.23, 0.24], 3, {'field': 'test_2', 'order': 3}), 
#                                      ([0.91, 0.43, 0.44, 0.67], 4, {'field': 'test_2', 'order': 4}), 
#                                      ([0.92, 0.12, 0.56, 0.19], 5, {'field': 'test_3', 'order': 5}),
#                                      ([0.18, 0.34, 0.56, 0.71], 6, {'field': 'test_1', 'order': 6}), 
#                                      ([0.01, 0.33, 0.14, 0.31], 7, {'field': 'test_2', 'order': 7}), 
#                                      ([0.71, 0.75, 0.91, 0.82], 8, {'field': 'test_3', 'order': 8}),
#                                      ([0.75, 0.44, 0.38, 0.75], 9, {'field': 'test_1', 'order': 9})])
# print(ids)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Query


```python
collection.query(vector=[0.36, 0.43, 0.56, 0.12], k=10)
```




    (array([ 2,  9,  1,  4,  6,  5, 10,  7,  8,  3]),
     array([1.        , 0.92355633, 0.86097705, 0.85727406, 0.81551266,
            0.813797  , 0.78595245, 0.7741583 , 0.6871773 , 0.34695023]))




```python
print(collection.query_report_)
```

    
    * - MOST RECENT QUERY REPORT -
    | - Collection Shape: (10, 4)
    | - Query Time: 0.15716 s
    | - Query Distance: cosine
    | - Query K: 10
    | - Top 10 Results ID: [ 2  9  1  4  6  5 10  7  8  3]
    | - Top 10 Results Similarity: [1.         0.92355633 0.86097705 0.85727406 0.81551266 0.813797
     0.78595245 0.7741583  0.6871773  0.34695023]
    * - END OF REPORT -
    


### Use Filter


```python
import operator

from min_vec.core_components.filter import Filter, FieldCondition, MatchField, IDCondition, MatchID


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
        ],
        must_not=[
            IDCondition(MatchID([8])), 
            FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
        ]
    )
)

print(collection.query_report_)
```

    
    * - MOST RECENT QUERY REPORT -
    | - Collection Shape: (10, 4)
    | - Query Time: 0.09065 s
    | - Query Distance: cosine
    | - Query K: 10
    | - Top 10 Results ID: [2 1]
    | - Top 10 Results Similarity: [1.         0.86097705]
    * - END OF REPORT -
    


### Drop a collection


```python
print("Collection list before dropping:", my_db.show_collections())
status = my_db.drop_collection("test_collection")
print("Collection list after dropped:", my_db.show_collections())
```

    Collection list before dropping: ['test_collection']
    {'status': 'success', 'params': {'collection_name': 'test_collection', 'exists': False}}
    Collection list after dropped: []


## Drop the database


```python
my_db.drop_database()
my_db
```




    MinVectorDB remote server at http://localhost:7637 does not exist.




```python
my_db.database_exists()
```




    {'status': 'success', 'params': {'exists': False}}



## What's Next

- [Collection's operations](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/collections.ipynb)
- [Add vectors to collection](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/add_vectors.ipynb)
- [Using different indexing methods](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/index_mode.ipynb)
- [Using different distance metric functions](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/distance.ipynb)
- [Diversified queries](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/queries.ipynb)
- [Benchmarks](https://github.com/BirchKwok/MinVectorDB/blob/main/tutorials/Benchmarks.ipynb)
