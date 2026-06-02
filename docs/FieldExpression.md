# Field Filters

Field filters are SQL-like expressions used by `search(where=...)`,
`batch_search(where=...)`, `search_range(...)`, `query(where=...)`,
`query_vectors(where=...)`, `text_search(where=...)`, and `hybrid_search(where=...)`.

Fields come from the metadata dict supplied during insertion:

```python
with collection.insert_session() as session:
    session.add_item(
        [0.1, 0.2, 0.3, 0.4],
        id=1,
        field={
            "lang": "en",
            "rank": 1,
            "active": True,
            "tags": ["vector", "python"],
            "created_at": "2026-06-02",
        },
    )
```

## Basic syntax

```python
collection.search(query, k=10, where="lang = 'en'")
collection.search(query, k=10, where="rank >= 10 AND rank < 20")
collection.search(query, k=10, where="active = true")
collection.search(query, k=10, where="tags CONTAINS 'vector'")
collection.search(query, k=10, where="created_at >= '2026-06-01'")
```

Field names may be unquoted when they are simple identifiers:

```python
where="lang = 'en'"
```

Use double quotes for field names with spaces, punctuation, or reserved words:

```python
where="\"document.lang\" = 'en'"
```

String literals can use single quotes:

```python
where="title = 'vector database'"
```

## Supported fast-path operators

| Operator | Example | Notes |
| --- | --- | --- |
| `=` | `lang = 'en'` | equality for strings, numbers, booleans, and dates stored as strings |
| `<`, `<=`, `>`, `>=` | `rank >= 10` | range filters for numbers and lexicographically sortable strings |
| `CONTAINS` | `tags CONTAINS 'python'` | array membership |
| `IN (...)` | `rank IN (1, 2, 3)` | equality against several values |
| `AND` | `lang = 'en' AND rank <= 10` | conjunction of indexed predicates |
| `OR` | `lang = 'en' OR lang = 'fr'` | optimized for simple equality leaves |

More complex expressions may fall back to the SQL engine when supported by the
underlying field store.

## Value types

Metadata values are JSON-like:

- strings;
- integers and floats;
- booleans;
- arrays;
- objects.

Arrays and objects are stored as JSON values. Use `CONTAINS` for array
membership.

For date/time filters, store ISO-8601 strings such as `2026-06-02` or
`2026-06-02T10:30:00Z`. ISO strings sort lexicographically in chronological
order when the format is consistent.

## Query examples

```python
# IDs and fields
rows = collection.query(where="lang = 'en' AND rank <= 10")

# IDs only
ids = collection.query(where="tags CONTAINS 'vector'", return_ids_only=True)

# Vectors, IDs, and fields
data = collection.query_vectors(where="active = true")

# Search with a pre-filter
result = collection.search(
    query,
    k=10,
    where="lang = 'en' AND tags CONTAINS 'python'",
    return_fields=True,
)
```

## Empty queries

These calls return empty results:

```python
collection.query()
collection.query_vectors()
```

Pass `filter_ids` if you want specific rows:

```python
collection.query(filter_ids=[1, 2, 3])
collection.query_vectors(filter_ids=[1, 2, 3])
```

## Performance notes

- Equality, range, `CONTAINS`, simple `IN`, and simple `AND` filters are the
  most index-friendly shapes.
- Keep field value types stable. Avoid mixing numbers and strings in the same
  field.
- Use low-to-medium cardinality fields for selective pre-filters.
- For high-cardinality exact lookups, pass `filter_ids` when you already know
  the IDs.
- Use `search_profile()` to inspect filter cardinality and search path.
