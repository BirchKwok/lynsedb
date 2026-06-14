# ResultView

`ResultView` is the common return type for search, query, head, tail, range, and
data retrieval APIs.

## Result shapes

Search results contain IDs and distances:

```python
result = collection.search(query, k=3, return_fields=True)

print(result.ids)        # np.ndarray[int64]
print(result.distances)  # np.ndarray[float32]
print(result.fields)     # list[dict]
```

Data results contain vectors:

```python
result = collection.query_vectors(filter_ids=[1, 2, 3])

print(result.ids)
print(result.vectors)    # np.ndarray[float32], shape (n, dim)
print(result.fields)
```

Query results contain IDs and optional fields:

```python
result = collection.query(where="lang = 'en'")

print(result.ids)
print(result.fields)
```

The main result types are:

| Result type | Produced by | Main attributes |
| --- | --- | --- |
| `search` | `search`, `batch_search`, `search_range`, `bm25_search`, `search_sparse`, `hybrid_search` | `ids`, `distances`, optional `fields` |
| `query` | `query` | `ids`, optional `fields` |
| `data` | `query_vectors`, `head`, `tail`, remote `read_by_only_id` | `vectors`, `ids`, `fields` |

`distances` may be lower-is-better distances or higher-is-better scores
depending on the metric. See the indexing and search tutorials for metric
semantics.

## Tuple unpacking

Search results unpack as `(ids, distances, fields)`:

```python
ids, distances, fields = collection.search(query, k=3, return_fields=True)
```

Data results unpack as `(vectors, ids, fields)`:

```python
vectors, ids, fields = collection.query_vectors(filter_ids=[1, 2])
```

Prefer attributes when clarity matters.

## Row iteration

`ResultView` is not a row iterator. Convert to rows first:

```python
for row in result.to_list():
    print(row)
```

`to_list()` returns dictionaries such as:

```python
[
    {"id": 1, "distance": 0.0, "title": "intro"},
    {"id": 2, "distance": 0.01, "title": "guide"},
]
```

## Conversion helpers

```python
result.to_tuple()
result.to_numpy()
result.to_dict()
result.to_list()
result.to_json()
```

Typical use:

```python
ids = result.ids.tolist()
rows = result.to_list()
payload = result.to_json()
```

Use `to_numpy()` for numerical post-processing and `to_list()` for application
responses or debug prints.

Optional conversions:

```python
result.to_pandas()  # requires pandas
result.to_polars()  # requires polars
result.to_arrow()   # requires pyarrow
```

## Indexing by component

Use string keys to access components:

```python
result["ids"]
result["distances"]
result["fields"]
result["vectors"]
```

Integer row indexing is intentionally not the main access path. Use
`result.to_list()[0]` when you need the first row as a dict.

## Empty results

Empty results are still valid `ResultView` objects:

```python
result = collection.query(where="lang = 'missing'")

print(len(result))       # 0
print(bool(result))      # False
print(result.ids)        # empty int64 array
print(result.to_list())  # []
```

This lets application code handle "no match" without special exception paths.
