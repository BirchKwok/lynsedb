# Quick Start


```python linenums="1"
import lynse

print("LynseDB version is: ", lynse.__version__)
```

    LynseDB version is:  0.2.0


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
docker run -p 7637:7637 birchkwok/lynsedb:latest
```
- Remote deploy

  If you want to deploy remotely, you can bind the image to port 80 of the remote host, or allow the host to open access to port 7637.
  such as:
```shell
docker run -p 80:7637 birchkwok/lynsedb:latest
```

- test if api available

  You can directly request in the browser http://localhost:7637

  For port 80, you can use this url: http://localhost

  If the image is bound to port 80 of the host in remote deployment, you can directly access it http://your_host_ip



```python linenums="1"
# If you are in a Jupyter environment, you can use this method to start the backend server
# Ignore this code if you are using docker
lynse.launch_in_jupyter()
```

    Server running at http://127.0.0.1:7637


```python linenums="1"
# Use the HTTP API mode, it is suitable for use in production environments.
client = lynse.VectorDBClient("http://127.0.0.1:7637")  # If no url is passed, the native api is used.
# Create a database named "test_db", if it already exists, delete it and rebuild it.
my_db = client.create_database("test_db", drop_if_exists=True)
```

### create a collection

**`WARNING`**

When using the `require_collection` method to request a collection, if the `drop_if_exists` parameter is set to True, it will delete all content of the collection if it already exists.

A safer method is to use the `get_collection` method. It is recommended to use the `require_collection` method only when you need to reinitialize a collection or create a new one.


```python linenums="1"
collection = my_db.require_collection("test_collection", dim=4, drop_if_exists=True, description="demo collection")
```

#### show database collections
If the pandas library is installed, `show_collections_details` method will show as a pandas dataframe. Otherwise, it will be a dict.

```python linenums="1"
my_db.show_collections_details()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dim</th>
      <th>chunk_size</th>
      <th>dtypes</th>
      <th>use_cache</th>
      <th>n_threads</th>
      <th>warm_up</th>
      <th>description</th>
      <th>cache_chunks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_collection</th>
      <td>4</td>
      <td>100000</td>
      <td>float32</td>
      <td>True</td>
      <td>10</td>
      <td>False</td>
      <td>demo collection</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



#### update description


```python linenums="1"
collection.update_description("Hello World")
my_db.show_collections_details()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dim</th>
      <th>chunk_size</th>
      <th>dtypes</th>
      <th>use_cache</th>
      <th>n_threads</th>
      <th>warm_up</th>
      <th>description</th>
      <th>cache_chunks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>test_collection</th>
      <td>4</td>
      <td>100000</td>
      <td>float32</td>
      <td>True</td>
      <td>10</td>
      <td>False</td>
      <td>Hello World</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



### Add vectors

When inserting vectors, the collection requires manually running the `commit` function or inserting within the `insert_session` function context manager, which will run the `commit` function in the background.

It is strongly recommended to use the `insert_session` context manager for insertion, as this provides more comprehensive data security features during the insertion process.


```python linenums="1"
with collection.insert_session() as session:
    id = session.add_item(vector=[0.01, 0.34, 0.74, 0.31], id=1, field={'field': 'test_1', 'order': 0})   # id = 1
    id = session.add_item(vector=[0.36, 0.43, 0.56, 0.12], id=2, field={'field': 'test_1', 'order': 1})   # id = 2
    id = session.add_item(vector=[0.03, 0.04, 0.10, 0.51], id=3, field={'field': 'test_2', 'order': 2})   # id = 3
    id = session.add_item(vector=[0.11, 0.44, 0.23, 0.24], id=4, field={'field': 'test_2', 'order': 3})   # id = 4
    id = session.add_item(vector=[0.91, 0.43, 0.44, 0.67], id=5, field={'field': 'test_2', 'order': 4})   # id = 5
    id = session.add_item(vector=[0.92, 0.12, 0.56, 0.19], id=6, field={'field': 'test_3', 'order': 5})   # id = 6
    id = session.add_item(vector=[0.18, 0.34, 0.56, 0.71], id=7, field={'field': 'test_1', 'order': 6})   # id = 7
    id = session.add_item(vector=[0.01, 0.33, 0.14, 0.31], id=8, field={'field': 'test_2', 'order': 7})   # id = 8
    id = session.add_item(vector=[0.71, 0.75, 0.91, 0.82], id=9, field={'field': 'test_3', 'order': 8})   # id = 9
    id = session.add_item(vector=[0.75, 0.44, 0.38, 0.75], id=10, field={'field': 'test_1', 'order': 9})  # id = 10

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
# print(ids)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```


    2024-09-12 17:33:36 - LynseDB - INFO - Task status: {'status': 'Processing'}
    2024-09-12 17:33:38 - LynseDB - INFO - Task status: {'result': {'collection_name': 'test_collection', 'database_name': 'test_db'}, 'status': 'Success'}

### Find the nearest neighbors of a given vector

The default similarity measure for query is Inner Product (IP). You can specify cosine or L2 to obtain the similarity measure you need.


```python linenums="1"
ids, scores, fields = collection.search(vector=[0.36, 0.43, 0.56, 0.12], k=3, distance="cosine", return_fields=True)
print("ids: ", ids)
print("scores: ", scores)
print("fields: ", fields)
```

    ids:  [ 9  5 10]
    scores:  [ 0.18610001 -0.16069996 -0.23799998]
    fields:  [{':id:': 9, 'field': 'test_3', 'order': 8}, {':id:': 5, 'field': 'test_2', 'order': 4}, {':id:': 10, 'field': 'test_1', 'order': 9}]


### List data


```python linenums="1"
ids, scores, fields = collection.head(5)
print("ids: ", ids)
print("scores: ", scores)
print("fields: ", fields)
```

    ids:  [1 2 3 4 5]
    scores:  [[0.01       0.34       0.74000001 0.31      ]
     [0.36000001 0.43000001 0.56       0.12      ]
     [0.03       0.04       0.1        0.50999999]
     [0.11       0.44       0.23       0.23999999]
     [0.91000003 0.43000001 0.44       0.67000002]]
    fields:  [{':id:': 1, 'field': 'test_1', 'order': 0}, {':id:': 2, 'field': 'test_1', 'order': 1}, {':id:': 3, 'field': 'test_2', 'order': 2}, {':id:': 4, 'field': 'test_2', 'order': 3}, {':id:': 5, 'field': 'test_2', 'order': 4}]



```python linenums="1"
ids, scores, fields = collection.tail(5)
print("ids: ", ids)
print("scores: ", scores)
print("fields: ", fields)
```

    ids:  [ 6  7  8  9 10]
    scores:  [[0.92000002 0.12       0.56       0.19      ]
     [0.18000001 0.34       0.56       0.70999998]
     [0.01       0.33000001 0.14       0.31      ]
     [0.70999998 0.75       0.91000003 0.81999999]
     [0.75       0.44       0.38       0.75      ]]
    fields:  [{':id:': 6, 'field': 'test_3', 'order': 5}, {':id:': 7, 'field': 'test_1', 'order': 6}, {':id:': 8, 'field': 'test_2', 'order': 7}, {':id:': 9, 'field': 'test_3', 'order': 8}, {':id:': 10, 'field': 'test_1', 'order': 9}]


## Use FieldExpression for result filtering

See [FieldExpression Tutorial](FieldExpression.md)

```python linenums="1"
ids, scores, fields = collection.search(
    vector=[0.36, 0.43, 0.56, 0.12],
    k=10,
    search_filter="""
        :field: == 'test_1' and
        ((0 <= :order: <= 8) or (:id: in [1, 2, 3, 4, 5])) and
        not (:id: == 8 and :order: >= 8)
    """,
    return_fields=False
)

print("ids: ", ids)
print("scores: ", scores)
print("fields: ", fields)
```

    ids:  [2 7 1]
    scores:  [-0.35749996 -0.39020002 -0.39859998]
    fields:  None


### Use Filter for freer conditional expression

Using the Filter class for result filtering can maximize Recall.

The Filter class now supports `must`, `any`, and `must_not` parameters, all of which only accept list-type argument values.

The filtering conditions in `must` must be met, those in `must_not` must not be met.

After filtering with `must` and `must_not` conditions, the conditions in `any` will be considered, and at least one of the conditions in `any` must be met.

The filter result must satisfy both `must` and `any`, but not `must_not`.


```python linenums="1"
import operator

from lynse.field_models import Filter, FieldCondition, MatchField, MatchID, MatchRange

ids, scores, fields = collection.search(
    vector=[0.36, 0.43, 0.56, 0.12],
    k=10,
    search_filter=Filter(
        must=[
            FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
        ],
        any=[
            FieldCondition(key='order', matcher=MatchRange(start=0, end=8, inclusive=True)),
            FieldCondition(key=":id:", matcher=MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
        ],
        must_not=[
            FieldCondition(key=":id:", matcher=MatchID([8])),
            FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
        ]
    ),
    return_fields=False
)

print("ids: ", ids)
print("scores: ", scores)
print("fields: ", fields)
```

    ids:  [2 7 1]
    scores:  [-0.35749996 -0.39020002 -0.39859998]
    fields:  None


### Query fields

#### Query via FieldExpression


```python linenums="1"
collection.query("""
    :field: == 'test_1' and
    ((0 <= :order: <= 8) or (:id: in [1, 2, 3, 4, 5])) and
    not (:id: == 8 and :order: >= 8)
""")
```




    [{':id:': 1, 'field': 'test_1', 'order': 0},
     {':id:': 2, 'field': 'test_1', 'order': 1},
     {':id:': 7, 'field': 'test_1', 'order': 6}]



#### Query via Filter


```python linenums="1"
query_filter=Filter(
    must=[
        FieldCondition(key='field', matcher=MatchField('test_1')),  # Support for filtering fields
    ],
    any=[
        FieldCondition(key='order', matcher=MatchRange(start=0, end=8, inclusive=True)),
        FieldCondition(key=":id:", matcher=MatchID([1, 2, 3, 4, 5])),  # Support for filtering IDs
    ],
    must_not=[
        FieldCondition(key=":id:", matcher=MatchID([8])),
        FieldCondition(key='order', matcher=MatchField(8, comparator=operator.ge)),
    ]
)

collection.query(query_filter)
```




    [{':id:': 1, 'field': 'test_1', 'order': 0},
     {':id:': 2, 'field': 'test_1', 'order': 1},
     {':id:': 7, 'field': 'test_1', 'order': 6}]



#### Exact Match


```python linenums="1"
collection.query({':id:': 1, 'field': 'test_1', 'order': 0})
```




    [{':id:': 1, 'field': 'test_1', 'order': 0}]



#### Fuzzy Match


```python linenums="1"
collection.query({'field': 'test_1'})
```




    [{':id:': 1, 'field': 'test_1', 'order': 0},
     {':id:': 2, 'field': 'test_1', 'order': 1},
     {':id:': 10, 'field': 'test_1', 'order': 9},
     {':id:': 7, 'field': 'test_1', 'order': 6}]



## Query Vectors

Much like query, you can query using either the FieldExpression string or the Filter class, fuzzy match, or exact match.


```python linenums="1"
collection.query_vectors("""
    :field: == 'test_1' and
    ((0 <= :order: <= 8) or (:id: in [1, 2, 3, 4, 5])) and
    not (:id: == 8 and :order: >= 8)
""")
```




    (array([1, 2, 7]),
     array([[0.01      , 0.34      , 0.74000001, 0.31      ],
            [0.36000001, 0.43000001, 0.56      , 0.12      ],
            [0.18000001, 0.34      , 0.56      , 0.70999998]]),
     [{':id:': 1, 'field': 'test_1', 'order': 0},
      {':id:': 2, 'field': 'test_1', 'order': 1},
      {':id:': 7, 'field': 'test_1', 'order': 6}])




```python linenums="1"
collection.query_vectors({'field': 'test_1'})
```




    (array([ 1,  2,  7, 10]),
     array([[0.01      , 0.34      , 0.74000001, 0.31      ],
            [0.36000001, 0.43000001, 0.56      , 0.12      ],
            [0.18000001, 0.34      , 0.56      , 0.70999998],
            [0.75      , 0.44      , 0.38      , 0.75      ]]),
     [{':id:': 1, 'field': 'test_1', 'order': 0},
      {':id:': 2, 'field': 'test_1', 'order': 1},
      {':id:': 7, 'field': 'test_1', 'order': 6},
      {':id:': 10, 'field': 'test_1', 'order': 9}])



### Drop a collection

`WARNING: This operation cannot be undone`


```python linenums="1"
print("Collection list before dropping:", my_db.show_collections())
status = my_db.drop_collection("test_collection")
print("Collection list after dropped:", my_db.show_collections())
```

    Collection list before dropping: ['test_collection']
    Collection list after dropped: []


## Drop the database

`WARNING: This operation cannot be undone`


```python linenums="1"
my_db.drop_database()
my_db
```




    RemoteDatabaseInstance(name=test_db, exists=False)
