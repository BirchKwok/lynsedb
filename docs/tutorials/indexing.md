# Tutorial: Indexing Guide

Indexes control the tradeoff between recall, latency, memory, disk usage, and
build time.

## The default choice

Start with a flat metric index:

```python
collection.build_index("FLAT-L2")
```

Flat search is exhaustive and simple. It is a good baseline for correctness,
small collections, and evaluation.

## Metric names

| Index suffix | Metric | Best for |
| --- | --- | --- |
| `FLAT`, `HNSW`, `IVF`, `DiskANN` | Inner product | normalized embeddings, maximum-score retrieval |
| `-L2` | Squared L2 distance | Euclidean-distance embeddings |
| `-COS` or `-Cos` | Cosine similarity | embeddings where angular similarity matters |
| `-HAMMING-BINARY` | Hamming distance | binary vectors |
| `-JACCARD-BINARY` | Jaccard distance | binary sets |

Choose the metric that matches how your embedding model was trained.

## Index families

| Family | Example | Strength | Notes |
| --- | --- | --- | --- |
| Flat | `FLAT-L2` | highest recall and simplest behavior | Latency grows with collection size. |
| HNSW | `HNSW-L2` | low-latency ANN | Use `nprobe` as search breadth (`ef_search`). |
| IVF | `IVF-L2` | large collections with tunable probes | Requires `n_clusters`; use `nprobe` at search time. |
| DiskANN | `DiskANN-L2` | disk-friendly graph ANN | Useful when memory pressure matters. |
| Quantized flat | `FLAT-IP-SQ8`, `FLAT-L2-PQ`, `FLAT-IP-RABITQ` | lower memory footprint | Some variants use two-pass search to preserve quality. |

## Build and remove indexes

```python
collection.build_index("HNSW-L2")
print(collection.index_mode)

collection.remove_index()
print(collection.index_mode)
```

Removing an index returns the collection to flat search.

## IVF parameters

IVF splits vectors into coarse clusters. `n_clusters` controls how many clusters
are built.

```python
collection.build_index("IVF-L2", n_clusters=256)
```

Rules:

- `n_clusters` is accepted only for IVF indexes.
- `n_clusters` must be greater than zero.
- More clusters usually reduce scanned vectors per query but can require higher
  `nprobe` for recall.

Search with:

```python
result = collection.search(query, k=10, nprobe=20)
```

For IVF, `nprobe` is the number of clusters to scan. Higher values improve
recall and increase latency.

## HNSW search breadth

For HNSW, `nprobe` is used as the search beam width:

```python
collection.build_index("HNSW-L2")
result = collection.search(query, k=10, nprobe=64)
```

Higher `nprobe` generally improves recall and increases latency.

## Approximate flat distance rounding

For flat IP, L2, and cosine paths, `approx=True` enables metric-specific
distance rounding controlled by `eps`.

```python
result = collection.search(query, k=10, approx=True, eps=1e-4)
```

Hamming and Jaccard binary metrics ignore `approx=True` and always use the exact
binary-distance path.

## Named vector indexes

Each named vector field has its own metric, dimension, and index:

```python
collection.create_vector_field("image", dim=512, metric="l2")
collection.add_named_vectors("image", image_vectors, ids=image_ids)
collection.build_index("HNSW-L2", field_name="image")

result = collection.search(image_query, k=10, vector_field="image")
```

Remove only that field's index:

```python
collection.remove_index(field_name="image")
```

## Practical tuning workflow

1. Build `FLAT-*` first and record quality on an evaluation set.
2. Try `HNSW-*` for low-latency online search.
3. Try `IVF-*` when you need more explicit recall/latency tuning.
4. Use quantized variants when memory or disk footprint is the bottleneck.
5. Always compare recall against the flat baseline before deploying.

## Supported index names

Common dense indexes:

```python
collection.build_index("FLAT")
collection.build_index("FLAT-L2")
collection.build_index("FLAT-COS")

collection.build_index("HNSW")
collection.build_index("HNSW-L2")
collection.build_index("HNSW-Cos")

collection.build_index("DiskANN")
collection.build_index("DiskANN-L2")
collection.build_index("DiskANN-Cos")

collection.build_index("IVF", n_clusters=256)
collection.build_index("IVF-L2", n_clusters=256)
collection.build_index("IVF-COS", n_clusters=256)
```

Common quantized variants:

```python
collection.build_index("FLAT-IP-SQ8")
collection.build_index("FLAT-L2-SQ8")
collection.build_index("FLAT-IP-PQ")
collection.build_index("FLAT-L2-PQ")
collection.build_index("FLAT-IP-RABITQ")
collection.build_index("FLAT-L2-RABITQ")
collection.build_index("FLAT-IP-POLARVEC")
collection.build_index("FLAT-L2-POLARVEC")
```

Binary variants:

```python
collection.build_index("FLAT-HAMMING-BINARY")
collection.build_index("FLAT-JACCARD-BINARY")
collection.build_index("IVF-HAMMING-BINARY", n_clusters=256)
collection.build_index("IVF-JACCARD-BINARY", n_clusters=256)
```
