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

A practical first decision:

| Collection size and goal | Start with |
| --- | --- |
| development, tests, evaluation | `FLAT-IP`, `FLAT-L2`, or `FLAT-COS` |
| low-latency online ANN | `HNSW-IP`, `HNSW-L2`, or `HNSW-Cos` |
| large collections with explicit probe tuning | `IVF-L2`, `SPANN-L2`, `IVF-COS`, or `SPANN-COS` |
| memory pressure from graph indexes | `DiskANN-IP`, `DiskANN-L2`, or `DiskANN-Cos` |
| lower memory or disk footprint | SQ8, PQ, RaBitQ, or PolarVec variants |
| binary vectors | `FLAT-HAMMING-BINARY`, `FLAT-JACCARD-BINARY`, or IVF binary variants |
| coordinates, profiles, or distributions | start with the matching domain `FLAT-*` metric |

## Metric names

| Index suffix | Metric | Best for |
| --- | --- | --- |
| `*-IP` | Inner product | normalized embeddings, maximum-score retrieval |
| `-L2` | Squared L2 distance | Euclidean-distance embeddings |
| `-COS` or `-Cos` | Cosine similarity | embeddings where angular similarity matters |
| `-L1` | Manhattan distance | robust numeric and sensor features |
| `-HAVERSINE` | Haversine distance in meters | GeoJSON-order coordinates |
| `-CORRELATION` | Pearson correlation distance | aligned profiles and time buckets |
| `-HELLINGER` | Hellinger distance | non-negative distributions |
| `-WASSERSTEIN` | Wasserstein-1D distance | equal-width ordered histograms |
| `-JENSEN-SHANNON` | Jensen–Shannon distance | probability and topic distributions |
| `-CHEBYSHEV` | Chebyshev/L∞ distance | maximum-deviation constraints |
| `-CANBERRA` | Canberra distance | spectra and sparse abundance data |
| `-BRAY-CURTIS` | Bray–Curtis distance | abundance and compositional profiles |
| `-HAMMING-BINARY` | Hamming distance | binary vectors |
| `-JACCARD-BINARY` | Jaccard distance | binary sets |
| `-TANIMOTO-BINARY` | Binary Tanimoto/Jaccard distance | molecular fingerprints |
| `-DICE-BINARY` | Sørensen-Dice distance | fingerprints and deduplication |

Choose the metric that matches how your embedding model was trained.

Metric guidance:

- use cosine when your embedding model documentation recommends cosine;
- use inner product when embeddings are normalized and maximum score retrieval
  is desired;
- use L2 when Euclidean distance is meaningful for your model;
- use Hamming or Jaccard only for binary vectors or binary-set style features.

See [Domain-aware distance metrics](distance_metrics.md) for input validation,
aliases, result units, and the complete compatibility matrix. New domain
metrics support Flat and numeric domain metrics also support HNSW; they are not
silently routed through IVF, SPANN, DiskANN, or quantizers.

## Index families

| Family | Example | Strength | Notes |
| --- | --- | --- | --- |
| Flat | `FLAT-L2` | highest recall and simplest behavior | Latency grows with collection size. |
| HNSW | `HNSW-L2` | low-latency ANN | Use `nprobe` as search breadth (`ef_search`). |
| IVF | `IVF-L2` | large collections with tunable probes | Requires `n_clusters`; use `nprobe` at search time. |
| SPANN | `SPANN-L2` | partition ANN with boundary replicas | Requires `n_clusters`; use `nprobe` at search time. |
| DiskANN | `DiskANN-L2` | disk-friendly graph ANN | Useful when memory pressure matters. |
| Quantized flat | `FLAT-IP-SQ8`, `FLAT-L2-PQ`, `FLAT-IP-RABITQ` | lower memory footprint | Some variants use two-pass search to preserve quality. |

## Build lifecycle

Indexes are persisted with the collection. Reopening the collection reloads the
index metadata and index files where applicable.

After a large initial load:

```python
with collection.insert_session() as session:
    for ids, vectors, fields in embedding_batches:
        session.add(ids=ids, vectors=vectors, fields=fields, batch_size=1000)

collection.build_index("HNSW-L2")
collection.checkpoint()
```

After many incremental writes, rebuild or switch index modes when recall or
latency has drifted from your target:

```python
collection.build_index("HNSW-L2")
collection.commit()
```

LynseDB can insert into existing graph indexes for supported paths, but a
controlled rebuild after large changes is still the easiest way to re-baseline
performance.

## Build and remove indexes

```python
collection.build_index("HNSW-L2")
print(collection.index_mode)

collection.remove_index()
print(collection.index_mode)
```

Removing an index returns the collection to flat search.

## IVF and SPANN parameters

IVF and SPANN split vectors into coarse clusters. `n_clusters` controls how many
clusters are built.

```python
collection.build_index("IVF-L2", n_clusters=256)
collection.build_index("SPANN-L2", n_clusters=256)
```

Rules:

- `n_clusters` is used only for IVF and SPANN indexes.
- For indexes other than IVF/SPANN, `n_clusters` is allowed and ignored by the Python API.
- For IVF and SPANN indexes, `n_clusters` must be greater than zero.
- More clusters usually reduce scanned vectors per query but can require higher
  `nprobe` for recall.

Search with:

```python
result = collection.search(query, k=10, nprobe=20)
```

For IVF and SPANN, `nprobe` is the number of clusters to scan. Higher values
improve recall and increase latency.

Starting values:

- `n_clusters=64` for small experiments;
- `n_clusters=256` or `1024` for larger collections;
- `nprobe=10` as a default search starting point;
- increase `nprobe` until recall is acceptable against a flat baseline.

## HNSW search breadth

For HNSW, `nprobe` is used as the search beam width:

```python
collection.build_index("HNSW-L2")
result = collection.search(query, k=10, nprobe=64)
```

Higher `nprobe` generally improves recall and increases latency.
Flat, PQ, RaBitQ, PolarVec, and named vector-field searches ignore `nprobe`.

Start with `nprobe=32` or `64` for HNSW evaluation, then tune down for latency
or up for recall.

## Approximate flat distance rounding

For flat IP, L2, and cosine paths, `approx=True` enables metric-specific
distance rounding controlled by `eps`.

```python
result = collection.search(query, k=10, approx=True, eps=1e-4)
```

Hamming and Jaccard binary metrics ignore `approx=True` and always use the exact
binary-distance path.

Use `approx=True` only after measuring quality on your own evaluation set.

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

Rules for named fields:

- create the field before adding named vectors;
- add named vectors only for IDs that already exist in the primary collection;
- choose the field metric at creation time;
- build or remove the named field index with `field_name=...`;
- `where` filters still use the row metadata fields.

## Quantized indexes

Quantized indexes reduce memory or disk pressure. They are most useful when
vector bandwidth or index size is the bottleneck.

| Family | Examples | When to try |
| --- | --- | --- |
| SQ8 | `FLAT-L2-SQ8`, `HNSW-L2-SQ8`, `IVF-COS-SQ8` | You want scalar quantization with familiar index families. |
| PQ | `FLAT-L2-PQ`, `FLAT-IP-PQ8`, `FLAT-IP-PQ16` | You want product quantization and can evaluate recall tradeoffs. |
| RaBitQ | `FLAT-IP-RABITQ`, `FLAT-L2-RABITQ` | You want aggressive binary-style compression. |
| PolarVec | `FLAT-IP-POLARVEC4`, `FLAT-L2-POLARVEC` | You want training-free multi-bit quantization. |

Quantized indexes should be evaluated against a flat baseline. Use the same
queries, filters, and `k` values your application uses.

## Binary indexes

Binary indexes are for binary vectors or binary-set style representations:

```python
collection.build_index("FLAT-HAMMING-BINARY")
collection.build_index("IVF-JACCARD-BINARY", n_clusters=256)
collection.build_index("FLAT-TANIMOTO-BINARY")
collection.build_index("FLAT-DICE-BINARY")
```

For Hamming, Jaccard/Tanimoto, and Dice, lower distance is better. Flat search
uses a lazily built one-bit-per-dimension hot representation. `approx` and
`eps` do not change binary-distance search behavior.

## Practical tuning workflow

1. Build `FLAT-*` first and record quality on an evaluation set.
2. Try `HNSW-*` for low-latency online search.
3. Try `IVF-*` when you need more explicit recall/latency tuning.
4. Use quantized variants when memory or disk footprint is the bottleneck.
5. Always compare recall against the flat baseline before deploying.

Evaluation loop:

```python
def evaluate(index_mode, *, n_clusters=None, nprobe=10):
    collection.build_index(index_mode, n_clusters=n_clusters)
    result = collection.search(query, k=10, nprobe=nprobe)
    return result.ids.tolist()

baseline = evaluate("FLAT-L2")
hnsw = evaluate("HNSW-L2", nprobe=64)
ivf = evaluate("IVF-L2", n_clusters=256, nprobe=20)

print(baseline)
print(hnsw)
print(ivf)
```

Use your own relevance labels when possible. If you do not have labels yet,
measure overlap with the flat baseline as a first recall proxy.

## Supported index names

All index names are case-insensitive. The examples below show the supported
spellings accepted by `build_index()`.

Dense indexes:

```python
collection.build_index("FLAT-IP")
collection.build_index("FLAT-L2")
collection.build_index("FLAT-COS")
collection.build_index("FLAT-COSINE")
collection.build_index("FLAT-IP-SQ8")
collection.build_index("FLAT-L2-SQ8")
collection.build_index("FLAT-COS-SQ8")
collection.build_index("FLAT-COSINE-SQ8")

collection.build_index("HNSW-IP")
collection.build_index("HNSW-L2")
collection.build_index("HNSW-COS")
collection.build_index("HNSW-COSINE")
collection.build_index("HNSW-IP-SQ8")
collection.build_index("HNSW-L2-SQ8")
collection.build_index("HNSW-COS-SQ8")
collection.build_index("HNSW-COSINE-SQ8")

collection.build_index("DiskANN-IP")
collection.build_index("DiskANN-L2")
collection.build_index("DiskANN-COS")
collection.build_index("DiskANN-COSINE")
collection.build_index("DiskANN-IP-SQ8")
collection.build_index("DiskANN-L2-SQ8")
collection.build_index("DiskANN-COS-SQ8")
collection.build_index("DiskANN-COSINE-SQ8")

collection.build_index("IVF-IP", n_clusters=256)
collection.build_index("IVF-L2", n_clusters=256)
collection.build_index("IVF-COS", n_clusters=256)
collection.build_index("IVF-COSINE", n_clusters=256)
collection.build_index("IVF-IP-SQ8", n_clusters=256)
collection.build_index("IVF-L2-SQ8", n_clusters=256)
collection.build_index("IVF-COS-SQ8", n_clusters=256)
collection.build_index("IVF-COSINE-SQ8", n_clusters=256)
```

Domain numeric indexes:

```python
collection.build_index("FLAT-L1")
collection.build_index("FLAT-HAVERSINE")
collection.build_index("FLAT-CORRELATION")
collection.build_index("FLAT-HELLINGER")
collection.build_index("FLAT-WASSERSTEIN")
collection.build_index("FLAT-JENSEN-SHANNON")
collection.build_index("FLAT-CHEBYSHEV")
collection.build_index("FLAT-CANBERRA")
collection.build_index("FLAT-BRAY-CURTIS")

collection.build_index("HNSW-L1")
collection.build_index("HNSW-HAVERSINE")
collection.build_index("HNSW-CORRELATION")
collection.build_index("HNSW-HELLINGER")
collection.build_index("HNSW-WASSERSTEIN")
collection.build_index("HNSW-JENSEN-SHANNON")
collection.build_index("HNSW-CHEBYSHEV")
```

Canberra and Bray–Curtis are intentionally Flat-only until metric-specific ANN
recall has been validated.

Flat quantized variants:

```python
collection.build_index("FLAT-IP-PQ")
collection.build_index("FLAT-L2-PQ")
collection.build_index("FLAT-COS-PQ")
collection.build_index("FLAT-COSINE-PQ")
collection.build_index("FLAT-IP-PQ8")
collection.build_index("FLAT-IP-PQ16")
collection.build_index("FLAT-L2-PQ8")
collection.build_index("FLAT-COS-PQ8")
collection.build_index("FLAT-IP-RABITQ")
collection.build_index("FLAT-L2-RABITQ")
collection.build_index("FLAT-COS-RABITQ")
collection.build_index("FLAT-COSINE-RABITQ")
collection.build_index("FLAT-IP-POLARVEC")
collection.build_index("FLAT-L2-POLARVEC")
collection.build_index("FLAT-COS-POLARVEC")
collection.build_index("FLAT-COSINE-POLARVEC")
collection.build_index("FLAT-IP-POLARVEC3")
collection.build_index("FLAT-IP-POLARVEC4")
collection.build_index("FLAT-IP-POLARVEC8")
```

PQ accepts `FLAT-{IP,L2,COS,COSINE}-PQ` and
`FLAT-{IP,L2,COS,COSINE}-PQ<N>`, where `<N>` is the requested number of
subspaces. If `<N>` is omitted, LynseDB chooses an automatic subspace count.

PolarVec accepts `FLAT-{IP,L2,COS,COSINE}-POLARVEC` and
`FLAT-{IP,L2,COS,COSINE}-POLARVEC<N>`, where `<N>` is a bit width from 1 to 8.
If `<N>` is omitted or invalid, LynseDB uses the default bit width.

Binary variants:

```python
collection.build_index("FLAT-HAMMING-BINARY")
collection.build_index("FLAT-HAMMING")
collection.build_index("FLAT-JACCARD-BINARY")
collection.build_index("FLAT-JACCARD")
collection.build_index("FLAT-TANIMOTO-BINARY")
collection.build_index("FLAT-TANIMOTO")
collection.build_index("FLAT-DICE-BINARY")
collection.build_index("FLAT-DICE")
collection.build_index("IVF-HAMMING-BINARY", n_clusters=256)
collection.build_index("IVF-HAMMING", n_clusters=256)
collection.build_index("IVF-JACCARD-BINARY", n_clusters=256)
collection.build_index("IVF-JACCARD", n_clusters=256)
```

## Troubleshooting index choices

| Symptom | Try |
| --- | --- |
| Results differ too much from expected neighbors | Compare against `FLAT-*`; increase `nprobe`; rebuild index. |
| IVF recall is low | Increase `nprobe`, reduce `n_clusters`, or improve training data coverage. |
| HNSW latency is high | Lower `nprobe`, try IVF, or use a more selective `where` filter. |
| Memory use is high | Try DiskANN or quantized variants; reduce unnecessary named vector fields. |
| Index build is slow | Build after bulk ingestion, not after every small batch; monitor `/metrics` in server mode. |
| Binary index scores look inverted | Remember Hamming, Jaccard/Tanimoto, and Dice are lower-is-better distances. |
