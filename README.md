<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png">
    <img alt="LynseDB logo" src="https://github.com/BirchKwok/LynseDB/blob/main/logo/logo.png" height="100">
  </picture>
</div>
<br>

<p align="center">
  <a href="https://discord.gg/u7DrH565XZ"><img src="https://img.shields.io/badge/Discord-Online-brightgreen" alt="Discord"></a>
  <a href="https://badge.fury.io/py/LynseDB"><img src="https://badge.fury.io/py/LynseDB.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/LynseDB/"><img src="https://img.shields.io/pypi/pyversions/LynseDB" alt="PyPI - Python Version"></a>
  <a href="https://github.com/BirchKwok/LynseDB/actions/workflows/python-tests.yml"><img src="https://github.com/BirchKwok/LynseDB/actions/workflows/python-tests.yml/badge.svg" alt="Python testing"></a>
  <a href="https://github.com/BirchKwok/LynseDB/actions/workflows/docker-tests.yml"><img src="https://github.com/BirchKwok/LynseDB/actions/workflows/docker-tests.yml/badge.svg" alt="Docker build"></a>
</p>


⚡ **Server-optional, simple parameters, simple API.**

⚡ **Fast, memory-efficient, easily scales to millions of vectors.**

⚡ **Friendly caching technology stores recently queried vectors for accelerated access.**

⚡ **Based on a generic Python software stack, platform-independent, highly versatile.**

*LynseDB* is a vector database implemented purely in Python, designed to be lightweight, server-optional, and easy to deploy locally or remotely. It offers straightforward and clear Python APIs, aiming to lower the entry barrier for using vector databases. 

LynseDB focuses on achieving 100% recall, prioritizing recall accuracy over high-speed search performance. This approach ensures that users can reliably retrieve all relevant vector data, making LynseDB particularly suitable for applications that require responses within hundreds of milliseconds.

- [x] **Now supports HTTP API and Python local code API and Docker deployment.**
- [X] **Now supports transaction management; if a commit fails, it will automatically roll back.**

<br>

:warning: **WARNING**

**Not yet backward compatible** 

LynseDB is actively being updated, and API backward compatibility is not guaranteed. You should use version numbers as a strong constraint during deployment to avoid unnecessary feature conflicts and errors. 

**Data size constraints**

Although our goal is to enable brute force search or inverted indexing on billion-scale vectors, we currently still recommend using it on a scale of millions of vectors or less for the best experience.

**python's native api is not process-safe**

The Python native API is recommended for use in single-process environments, whether single-threaded or multi-threaded; for ensuring process safety in multi-process environments, please use the HTTP API.



## Prerequisite

- [x] python version >= 3.9
- [x] Owns one of the operating systems: Windows, macOS, or Ubuntu (or other Linux distributions). The recommendation is for the latest version of the system, but non-latest versions should also be installable, although they have not been tested.
- [x] Memory >= 4GB, Free Disk >= 4GB.

## Install Client API package (Mandatory)

```shell
pip install LynseDB
```

## If you wish to use Docker (Optional)

**You must first [install Docker](https://docs.docker.com/engine/install/) on the host machine.**

```shell
docker pull birchkwok/LynseDB:latest
```

## Qucik Start


```python
import lynse

print("LynseDB version is: ", lynse.__version__)
```

    LynseDB version is:  0.0.1


### Initialize Database

LynseDB now supports HTTP API and Python native code API. 


The HTTP API mode requires starting an HTTP server beforehand. You have two options: 
- start directly.
  
  For direct startup, the default port is 7637. You can run the following command in the terminal to start the service:
```shell
lynse run --host localhost --port 7637
```

- within Docker
  
  In Docker, You can run the following command in the terminal to start the service:
```shell
docker run -p 7637:7637 birchkwok/LynseDB:latest
```
- Remote deploy

  If you want to deploy remotely, you can bind the image to port 80 of the remote host, or allow the host to open access to port 7637.
  such as:
```shell
docker run -p 80:7637 birchkwok/LynseDB:latest
```

- test if api available

  You can directly request in the browser http://localhost:7637
  
  For port 80, you can use this url: http://localhost
  
  If the image is bound to port 80 of the host in remote deployment, you can directly access it http://your_host_ip



```python
# If you are in a Jupyter environment, you can use this method to start the backend server
# Ignore this code if you are using docker
lynse.launch_in_jupyter()
```

    Server running at http://127.0.0.1:7637
    



```python
# Use the HTTP API mode, it is suitable for use in production environments.
client = lynse.VectorDBClient("http://127.0.0.1:7637")  # If no url is passed, the native api is used.
# Create a database named "test_db", if it already exists, delete it and rebuild it.
my_db = client.create_database("test_db", drop_if_exists=True)
```

### create a collection

**`WARNING`**

When using the `require_collection` method to request a collection, if the `drop_if_exists` parameter is set to True, it will delete all content of the collection if it already exists. 

A safer method is to use the `get_collection` method. It is recommended to use the `require_collection` method only when you need to reinitialize a collection or create a new one.


```python
collection = my_db.require_collection("test_collection", dim=4, drop_if_exists=True, scaler_bits=None, description="demo collection")
```

    2024-06-16 19:49:44 - LynseDB - INFO - Creating collection test_collection with: 
    //    dim=4, collection='test_collection', 
    //    chunk_size=100000, distance='cosine', 
    //    dtypes='float32', use_cache=True, 
    //    scaler_bits=None, n_threads=10, 
    //    warm_up=False, drop_if_exists=True, 
    //    description=demo collection, 
    


#### show database collections


```python
my_db.show_collections_details()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dim</th>
      <th>chunk_size</th>
      <th>dtypes</th>
      <th>distance</th>
      <th>use_cache</th>
      <th>scaler_bits</th>
      <th>n_threads</th>
      <th>warm_up</th>
      <th>initialize_as_collection</th>
      <th>description</th>
      <th>buffer_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_collection</th>
      <td>4</td>
      <td>100000</td>
      <td>float32</td>
      <td>cosine</td>
      <td>True</td>
      <td>None</td>
      <td>10</td>
      <td>False</td>
      <td>True</td>
      <td>demo collection</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



#### update description


```python
collection.update_description("Hello World")
my_db.show_collections_details()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dim</th>
      <th>chunk_size</th>
      <th>dtypes</th>
      <th>distance</th>
      <th>use_cache</th>
      <th>scaler_bits</th>
      <th>n_threads</th>
      <th>warm_up</th>
      <th>initialize_as_collection</th>
      <th>description</th>
      <th>buffer_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_collection</th>
      <td>4</td>
      <td>100000</td>
      <td>float32</td>
      <td>cosine</td>
      <td>True</td>
      <td>None</td>
      <td>10</td>
      <td>False</td>
      <td>True</td>
      <td>Hello World</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



### Add vectors

When inserting vectors, the collection requires manually running the `commit` function or inserting within the `insert_session` function context manager, which will run the `commit` function in the background.

It is strongly recommended to use the `insert_session` context manager for insertion, as this provides more comprehensive data security features during the insertion process.


```python
with collection.insert_session() as session:
    id = session.add_item(vector=[0.01, 0.34, 0.74, 0.31], id=1, field={'field': 'test_1', 'order': 0})   # id = 0
    id = session.add_item(vector=[0.36, 0.43, 0.56, 0.12], id=2, field={'field': 'test_1', 'order': 1})   # id = 1
    id = session.add_item(vector=[0.03, 0.04, 0.10, 0.51], id=3, field={'field': 'test_2', 'order': 2})   # id = 2
    id = session.add_item(vector=[0.11, 0.44, 0.23, 0.24], id=4, field={'field': 'test_2', 'order': 3})   # id = 3
    id = session.add_item(vector=[0.91, 0.43, 0.44, 0.67], id=5, field={'field': 'test_2', 'order': 4})   # id = 4
    id = session.add_item(vector=[0.92, 0.12, 0.56, 0.19], id=6, field={'field': 'test_3', 'order': 5})   # id = 5
    id = session.add_item(vector=[0.18, 0.34, 0.56, 0.71], id=7, field={'field': 'test_1', 'order': 6})   # id = 6
    id = session.add_item(vector=[0.01, 0.33, 0.14, 0.31], id=8, field={'field': 'test_2', 'order': 7})   # id = 7
    id = session.add_item(vector=[0.71, 0.75, 0.91, 0.82], id=9, field={'field': 'test_3', 'order': 8})   # id = 8
    id = session.add_item(vector=[0.75, 0.44, 0.38, 0.75], id=10, field={'field': 'test_1', 'order': 9})  # id = 9

# If you do not use the insert_session function, you need to manually call the commit function to submit the data
# collection.commit()

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

    2024-06-16 19:49:44 - LynseDB - INFO - Saving data...
    2024-06-16 19:49:44 - LynseDB - INFO - Writing chunk to storage...
    2024-06-16 19:49:44 - LynseDB - INFO - Task status: {'status': 'Processing'}
    2024-06-16 19:49:44 - LynseDB - INFO - Writing chunk to storage done.
    2024-06-16 19:49:46 - LynseDB - INFO - Task status: {'result': {'collection_name': 'test_collection', 'database_name': 'test_db'}, 'status': 'Success'}


### Find the nearest neighbors of a given vector

The default similarity measure for query is Inner Product (IP). You can specify cosine or L2 to obtain the similarity measure you need.


```python
ids, scores, fields = collection.search(vector=[0.36, 0.43, 0.56, 0.12], k=3, distance="cosine", return_fields=True)
print("ids: ", ids)
print("scores: ", scores)
print("fields: ", fields)
```

    ids:  [2 9 1]
    scores:  [1.         0.92355633 0.86097705]
    fields:  [{':id:': 2, 'field': 'test_1', 'order': 1}, {':id:': 9, 'field': 'test_3', 'order': 8}, {':id:': 1, 'field': 'test_1', 'order': 0}]


The `query_report_` attribute is the report of the most recent query. When multiple queries are conducted simultaneously, this attribute will only save the report of the last completed query result.


```python
print(collection.search_report_)
```

    
    * - MOST RECENT SEARCH REPORT -
    | - Collection Shape: (10, 4)
    | - Search Time: 0.01578 s
    | - Search Distance: cosine
    | - Search K: 3
    | - Top 3 Results ID: [2 9 1]
    | - Top 3 Results Similarity: [1.         0.92355633 0.86097705]
    


### List data


```python
collection.head(10)
```




    (array([[0.01      , 0.34      , 0.74000001, 0.31      ],
            [0.36000001, 0.43000001, 0.56      , 0.12      ],
            [0.03      , 0.04      , 0.1       , 0.50999999],
            [0.11      , 0.44      , 0.23      , 0.23999999],
            [0.91000003, 0.43000001, 0.44      , 0.67000002],
            [0.92000002, 0.12      , 0.56      , 0.19      ],
            [0.18000001, 0.34      , 0.56      , 0.70999998],
            [0.01      , 0.33000001, 0.14      , 0.31      ],
            [0.70999998, 0.75      , 0.91000003, 0.81999999],
            [0.75      , 0.44      , 0.38      , 0.75      ]]),
     array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]),
     [{':id:': 1, 'field': 'test_1', 'order': 0},
      {':id:': 2, 'field': 'test_1', 'order': 1},
      {':id:': 3, 'field': 'test_2', 'order': 2},
      {':id:': 4, 'field': 'test_2', 'order': 3},
      {':id:': 5, 'field': 'test_2', 'order': 4},
      {':id:': 6, 'field': 'test_3', 'order': 5},
      {':id:': 7, 'field': 'test_1', 'order': 6},
      {':id:': 8, 'field': 'test_2', 'order': 7},
      {':id:': 9, 'field': 'test_3', 'order': 8},
      {':id:': 10, 'field': 'test_1', 'order': 9}])




```python
collection.tail(5)
```




    (array([[0.92000002, 0.12      , 0.56      , 0.19      ],
            [0.18000001, 0.34      , 0.56      , 0.70999998],
            [0.01      , 0.33000001, 0.14      , 0.31      ],
            [0.70999998, 0.75      , 0.91000003, 0.81999999],
            [0.75      , 0.44      , 0.38      , 0.75      ]]),
     array([ 6,  7,  8,  9, 10]),
     [{':id:': 6, 'field': 'test_3', 'order': 5},
      {':id:': 7, 'field': 'test_1', 'order': 6},
      {':id:': 8, 'field': 'test_2', 'order': 7},
      {':id:': 9, 'field': 'test_3', 'order': 8},
      {':id:': 10, 'field': 'test_1', 'order': 9}])



### Use Filter

Using the Filter class for result filtering can maximize Recall. 

The Filter class now supports `must`, `any`, and `must_not` parameters, all of which only accept list-type argument values. 

The filtering conditions in `must` must be met, those in `must_not` must not be met. 

After filtering with `must` and `must_not` conditions, the conditions in `any` will be considered, and at least one of the conditions in `any` must be met. 

If there is a conflict between the conditions in `any` and those in `must` or `must_not`, the conditions in `any` will be ignored.


```python
import operator

from lynse.core_components.kv_cache.filter import Filter, FieldCondition, MatchField, MatchID, MatchRange

collection.search(
    vector=[0.36, 0.43, 0.56, 0.12],
    k=10,
    search_filter=Filter(
        must=[
            FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
        ],
        any=[
            FieldCondition(key='order', matcher=MatchRange(start=0, end=8, inclusive=True)),
            FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
        ],
        must_not=[
            FieldCondition(key=":match_id:", matcher=MatchID([8])),
            FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
        ]
    )
)

print(collection.search_report_)
```

    
    * - MOST RECENT SEARCH REPORT -
    | - Collection Shape: (10, 4)
    | - Search Time: 0.00729 s
    | - Search Distance: cosine
    | - Search K: 10
    | - Top 10 Results ID: [2 1 7]
    | - Top 10 Results Similarity: [1.         0.86097705 0.7741583 ]
    


### Query existing text in fields

#### Query via Filter


```python
query_filter=Filter(
    must=[
        FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
    ],
    any=[
        FieldCondition(key='order', matcher=MatchRange(start=0, end=8, inclusive=True)),
        FieldCondition(key=":match_id:", matcher=MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
    ],
    must_not=[
        FieldCondition(key=":match_id:", matcher=MatchID([8])),
        FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
    ]
)

collection.query(query_filter)
```




    [{':id:': 1, 'field': 'test_1', 'order': 0},
     {':id:': 2, 'field': 'test_1', 'order': 1},
     {':id:': 7, 'field': 'test_1', 'order': 6}]



#### Precision Query


```python
collection.query({':id:': 1, 'field': 'test_1', 'order': 0})
```




    [{':id:': 1, 'field': 'test_1', 'order': 0}]



#### Fuzzy Query


```python
collection.query({'field': 'test_1'})
```




    [{':id:': 1, 'field': 'test_1', 'order': 0},
     {':id:': 2, 'field': 'test_1', 'order': 1},
     {':id:': 7, 'field': 'test_1', 'order': 6},
     {':id:': 10, 'field': 'test_1', 'order': 9}]



### Drop a collection

`WARNING: This operation cannot be undone`


```python
print("Collection list before dropping:", my_db.show_collections())
status = my_db.drop_collection("test_collection")
print("Collection list after dropped:", my_db.show_collections())
```

    Collection list before dropping: ['test_collection']
    Collection list after dropped: []


## Drop the database

`WARNING: This operation cannot be undone`


```python
my_db.drop_database()
my_db
```




    Database `test_db` does not exist on the LynseDB remote server.



## What's Next

- [Collection's operations](https://github.com/BirchKwok/LynseDB/blob/main/tutorials/collections.ipynb)
- [Add vectors to collection](https://github.com/BirchKwok/LynseDB/blob/main/tutorials/add_vectors.ipynb)
- [Using different indexing methods](https://github.com/BirchKwok/LynseDB/blob/main/tutorials/index_mode.ipynb)
- [Using different distance metric functions](https://github.com/BirchKwok/LynseDB/blob/main/tutorials/distance.ipynb)
- [Diversified queries](https://github.com/BirchKwok/LynseDB/blob/main/tutorials/queries.ipynb)
