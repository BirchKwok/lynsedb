# Tutorial: Metadata Filter Cookbook

Metadata filters are standard SQL-style `where` strings used by search and
query methods.
This cookbook shows practical filter shapes and how to store fields so those
filters stay simple.

## Example data

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./filter-cookbook")
db = client.create_database("filters", drop_if_exists=True)
collection = db.require_collection("docs", dim=4, drop_if_exists=True)

items = [
    ([0.10, 0.20, 0.30, 0.40], 1, {
        "tenant": "acme",
        "lang": "en",
        "rank": 1,
        "published": True,
        "tags": ["vector", "docs"],
        "created_at": "2026-06-01",
    }),
    ([0.11, 0.19, 0.29, 0.39], 2, {
        "tenant": "acme",
        "lang": "en",
        "rank": 2,
        "published": False,
        "tags": ["draft"],
        "created_at": "2026-06-03",
    }),
    ([0.80, 0.10, 0.20, 0.10], 3, {
        "tenant": "globex",
        "lang": "fr",
        "rank": 3,
        "published": True,
        "tags": ["archive", "docs"],
        "created_at": "2026-06-05",
    }),
]

with collection.insert_session() as session:
    session.bulk_add_items(items, enable_progress_bar=False)

collection.build_index("FLAT-L2")
query = np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32)
```

## Equality

```python
collection.search(query, k=10, where="tenant = 'acme'", return_fields=True)
collection.query(where="lang = 'fr'")
```

Use equality for tenant, language, source, status, and exact categories.

## Numeric ranges

```python
collection.search(query, k=10, where="rank >= 1 AND rank <= 2")
collection.query(where="rank < 3")
```

Keep numeric fields numeric. Avoid storing numbers as strings if you need range
filters.

## Booleans

```python
collection.search(query, k=10, where="published = true")
collection.query(where="published = false")
```

Use lowercase `true` and `false` in filter strings.

## Arrays and tags

Use `CONTAINS` for array membership:

```python
collection.search(query, k=10, where="tags CONTAINS 'docs'")
collection.query(where="tags CONTAINS 'archive'")
```

This is a good shape for tags, labels, permissions, or feature flags.

## IN lists

```python
collection.search(query, k=10, where="rank IN (1, 3)")
collection.query(where="lang IN ('en', 'fr')")
```

Use `IN` when your application has a short allow-list. For a long list of known
IDs, prefer `filter_ids`.

## Dates and times

Store dates and times as ISO-8601 strings:

```python
collection.search(
    query,
    k=10,
    where="created_at >= '2026-06-01' AND created_at <= '2026-06-30'",
)
```

Consistent ISO strings sort lexicographically in chronological order.

## Compound filters

```python
where = "tenant = 'acme' AND lang = 'en' AND published = true"
result = collection.search(query, k=10, where=where, return_fields=True)
```

Use `AND` for common pre-filters. It is usually the most predictable way to
reduce candidate sets.

Use `OR` for simple alternatives:

```python
collection.query(where="lang = 'en' OR lang = 'fr'")
```

## Quoted field names

Simple identifiers can be unquoted:

```python
where = "tenant = 'acme'"
```

Quote field names that contain punctuation, spaces, or reserved words:

```python
where = "\"document.lang\" = 'en'"
collection.search(query, k=10, where=where)
```

## Filter then retrieve vectors

```python
rows = collection.query_vectors(where="tenant = 'acme' AND published = true")
print(rows.ids)
print(rows.vectors.shape)
print(rows.fields)
```

This is useful for exports, evaluation sets, and offline analysis.

## Filter with vector search

```python
result = collection.search(
    query,
    k=5,
    where="tenant = 'acme' AND tags CONTAINS 'docs'",
    return_fields=True,
)
print(result.to_list())
```

## Filter with text and hybrid search

```python
text = collection.text_search(
    "docs",
    k=5,
    text_fields=["tags"],
    where="tenant = 'acme'",
    return_fields=True,
)

hybrid = collection.hybrid_search(
    vector=query,
    text="docs",
    text_fields=["tags"],
    where="tenant = 'acme'",
    k=5,
    return_fields=True,
)
```

## `filter_ids` instead of a filter expression

When you already know the IDs, use `filter_ids`:

```python
rows = collection.query(filter_ids=[1, 2, 3])
vectors = collection.query_vectors(filter_ids=[1, 2, 3])
```

This avoids building a long `id IN (...)` style expression.

## Empty query behavior

These calls return empty results:

```python
collection.query()
collection.query_vectors()
```

This prevents accidental full scans. Pass a `where` expression or explicit
`filter_ids`.

## Field design checklist

- Use low-to-medium-cardinality fields for frequent filters.
- Keep data types stable within each field.
- Use booleans for visibility and published flags.
- Use ISO date strings for time ranges.
- Use arrays plus `CONTAINS` for tags.
- Keep raw text fields short enough for your text-search and result payload
  needs.
- Use `search_profile()` when filter behavior or latency is surprising.
