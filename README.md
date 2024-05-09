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

## Prerequisite

- [x] python version >= 3.9
- [x] Owns one of the operating systems: Windows, macOS, or Ubuntu (or other Linux distributions). The recommendation is for the latest version of the system, but non-latest versions should also be installable, although they have not been tested.
- [x] Memory >= 4GB, Free Disk >= 4GB.

## Install Client API package (Mandatory)

```shell
pip install MinVectorDB
```

## If you wish to use Docker (Optional)

**You must first [install Docker](https://docs.docker.com/engine/install/) on the host machine.**

```shell
docker pull birchkwok/minvectordb:latest
```

## Qucik Start


```python
import min_vec
print("MinVectorDB version is: ", min_vec.__version__)
```

    MinVectorDB version is:  0.3.4


## Initialize Database

MinVectorDB now supports HTTP API and Python native code API. 


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
- Remote deploy

  If you want to deploy remotely, you can bind the image to port 80 of the remote host, or allow the host to open access to port 7637.
  such as:
```shell
docker run -p 80:7637 birchkwok/minvectordb:latest
```

- test if api available

  You can directly request in the browser http://localhost:7637
  
  For port 80, you can use this url: http://localhost
  
  If the image is bound to port 80 of the host in remote deployment, you can directly access it http://your_host_ip
    


```python
from min_vec import MinVectorDB

# Use the HTTP API mode, it is suitable for use in production environments.
my_db = MinVectorDB("http://localhost:7637")
# Or use the Python native code API by specifying the database root directory.
# my_db = MinVectorDB('my_vec_db')  # Judgment condition, root_path does not start with http or https
# The Python native code API is recommended only for CI/CD testing or single-user local use.
```

### create a collection

**`WARNING`**

When using the `require_collection` method to request a collection, if the `drop_if_exists` parameter is set to True, it will delete all content of the collection if it already exists. 

A safer method is to use the `get_collection` method. It is recommended to use the `require_collection` method only when you need to reinitialize a collection or create a new one.


```python
collection = my_db.require_collection("test_collection", dim=4, drop_if_exists=True, scaler_bits=8, description="demo collection")
```

#### show database collections


```python
my_db.show_collections_details()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_size</th>
      <th>description</th>
      <th>dim</th>
      <th>distance</th>
      <th>dtypes</th>
      <th>index_mode</th>
      <th>initialize_as_collection</th>
      <th>n_clusters</th>
      <th>n_threads</th>
      <th>scaler_bits</th>
      <th>use_cache</th>
      <th>warm_up</th>
    </tr>
    <tr>
      <th>collections</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_collection</th>
      <td>100000</td>
      <td>demo collection</td>
      <td>4</td>
      <td>cosine</td>
      <td>float32</td>
      <td>IVF-FLAT</td>
      <td>True</td>
      <td>16</td>
      <td>10</td>
      <td>8</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



#### update description


```python
collection.update_description("test2")
my_db.show_collections_details()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chunk_size</th>
      <th>description</th>
      <th>dim</th>
      <th>distance</th>
      <th>dtypes</th>
      <th>index_mode</th>
      <th>initialize_as_collection</th>
      <th>n_clusters</th>
      <th>n_threads</th>
      <th>scaler_bits</th>
      <th>use_cache</th>
      <th>warm_up</th>
    </tr>
    <tr>
      <th>collections</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_collection</th>
      <td>100000</td>
      <td>test2</td>
      <td>4</td>
      <td>cosine</td>
      <td>float32</td>
      <td>IVF-FLAT</td>
      <td>True</td>
      <td>16</td>
      <td>10</td>
      <td>8</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



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


The default similarity measure for query is cosine. You can specify cosine or L2 to obtain the similarity measure you need.


```python
collection.query(vector=[0.36, 0.43, 0.56, 0.12], k=10)
```




    (array([ 2,  9,  1,  4,  6,  5, 10,  7,  8,  3]),
     array([1.        , 0.92355633, 0.86097705, 0.85727406, 0.81551266,
            0.813797  , 0.78595245, 0.7741583 , 0.6871773 , 0.34695023]))



The `query_report_` attribute is the report of the most recent query. When multiple queries are conducted simultaneously, this attribute will only save the report of the last completed query result.


```python
print(collection.query_report_)
```

    
    * - MOST RECENT QUERY REPORT -
    | - Collection Shape: (10, 4)
    | - Query Time: 0.13898 s
    | - Query Distance: cosine
    | - Query K: 10
    | - Top 10 Results ID: [ 2  9  1  4  6  5 10  7  8  3]
    | - Top 10 Results Similarity: [1.         0.92355633 0.86097705 0.85727406 0.81551266 0.813797
     0.78595245 0.7741583  0.6871773  0.34695023]
    * - END OF REPORT -
    


### Use Filter

Using the Filter class for result filtering can maximize Recall. 

The Filter class now supports `must`, `any`, and `must_not` parameters, all of which only accept list-type argument values. 

The filtering conditions in `must` must be met, those in `must_not` must not be met. 

After filtering with `must` and `must_not` conditions, the conditions in `any` will be considered, and at least one of the conditions in `any` must be met. 

If there is a conflict between the conditions in `any` and those in `must` or `must_not`, the conditions in `any` will be ignored.


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
    | - Query Time: 0.09066 s
    | - Query Distance: cosine
    | - Query K: 10
    | - Top 10 Results ID: [2 1]
    | - Top 10 Results Similarity: [1.         0.86097705]
    * - END OF REPORT -
    


### Drop a collection

`WARNING: This operation cannot be undone`


```python
print("Collection list before dropping:", my_db.show_collections())
status = my_db.drop_collection("test_collection")
print("Collection list after dropped:", my_db.show_collections())
```

    Collection list before dropping: ['test_collection']
    {'status': 'success', 'params': {'collection_name': 'test_collection', 'exists': False}}
    Collection list after dropped: []


## Drop the database

`WARNING: This operation cannot be undone`


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
