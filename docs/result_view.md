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
