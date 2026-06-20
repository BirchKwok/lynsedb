# Domain-Aware Distance Metrics

LynseDB can search coordinates, binary fingerprints, aligned profiles, and
probability distributions without converting them into semantic embeddings.
The metric is persisted with the index and applies consistently to exact
search, filters, batch search, and range search.

## Metric selection

| Metric | Accepted names | Input contract | Typical uses |
| --- | --- | --- | --- |
| Inner product | `ip`, `dot` | finite numeric vectors | normalized embeddings, maximum-score retrieval |
| Squared L2 | `l2`, `euclidean` | finite numeric vectors | embeddings and geometric features |
| Cosine distance | `cos`, `cosine` | numeric vectors | angular embedding similarity |
| Manhattan | `l1`, `manhattan`, `cityblock` | finite numeric vectors | anomaly, sensor, and tabular feature matching |
| Haversine | `haversine`, `geo` | `[longitude_degrees, latitude_degrees]` | nearby POI, fleet, and device search |
| Correlation | `correlation`, `pearson` | aligned, fixed-length numeric profiles | sensor curves, behavior profiles, gene expression |
| Hellinger | `hellinger` | non-negative weights or probabilities | model outputs, topics, preference distributions |
| Wasserstein-1D | `wasserstein`, `emd` | non-negative mass in equal-width ordered bins | histograms, latency and forecast distributions |
| Jensen–Shannon | `jensen_shannon`, `js` | non-negative weights or probabilities | probability models, topics, prediction comparison |
| Chebyshev | `chebyshev`, `linf` | finite numeric vectors | industrial tolerances, maximum-deviation constraints |
| Canberra | `canberra` | numeric vectors | spectra, sparse abundance and compositional features |
| Bray–Curtis | `bray_curtis` | finite, non-negative abundance vectors | ecology, spectra and compositional profiles |
| Hamming | `hamming` | values thresholded at `> 0.5` | bit signatures and error patterns |
| Jaccard/Tanimoto | `jaccard`, `tanimoto` | values thresholded at `> 0.5` | molecular fingerprints, sets, genomic sketches |
| Sørensen-Dice | `dice`, `sorensen` | values thresholded at `> 0.5` | fingerprints and near-duplicate detection |

All metrics except inner product return lower-is-better distances. Cosine
returns `1 - cosine_similarity`, so identical non-zero vectors have distance
zero. Inner product retains its conventional higher-is-better score for API
compatibility.

## Index compatibility

| Metric family | Flat | HNSW | IVF/SPANN/DiskANN | Quantized variants |
| --- | --- | --- | --- | --- |
| IP, L2, cosine | yes | yes | yes | supported variants |
| L1, Haversine, correlation, Hellinger, Wasserstein-1D | yes | yes | not yet | not yet |
| Jensen–Shannon, Chebyshev | yes | yes | not yet | not yet |
| Canberra, Bray–Curtis | yes | not yet | not yet | not yet |
| Hamming, Jaccard | packed hot path | not yet | IVF binary variants | binary only |
| Tanimoto, Dice | packed hot path | not yet | not yet | binary only |

Start with Flat as the exact quality baseline. Use HNSW for supported domain
metrics after measuring recall on representative queries. LynseDB rejects
unsupported index/metric combinations instead of silently substituting another
metric.

## Reusable example setup

Every metric uses the same collection workflow: create a collection, add
equal-length vectors, commit, and build an index whose suffix selects the
metric. The examples below share this local-client setup; an HTTP
`VectorDBClient` uses the same methods.

```python
import numpy as np
import lynse

client = lynse.VectorDBClient(uri="./lynsedb-distance-demo")
db = client.create_database("distance_demo", drop_if_exists=True)


def metric_collection(name, vectors, index_mode):
    vectors = np.asarray(vectors, dtype=np.float32)
    collection = db.require_collection(
        name,
        dim=vectors.shape[1],
        default_index=None,
        drop_if_exists=True,
    )
    collection.add(
        ids=[f"{name}-{i}" for i in range(len(vectors))],
        vectors=vectors,
    )
    collection.commit()
    collection.build_index(index_mode)
    return collection
```

Passing `default_index=None` is important in these examples: it prevents the
new collection from first building the default `FLAT-IP` index. For an existing
collection, call `build_index(...)` with the desired mode to replace its active
index.

## Core vector distances

### Inner product

Inner product ranks larger scores first. It is commonly used with normalized
embeddings, where it produces the same ordering as cosine similarity:

```python
embeddings = metric_collection(
    "embeddings",
    [
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "FLAT-IP",
)

result = embeddings.search([1.0, 0.0, 0.0], k=2)
print(result.ids)
print(result.distances)  # larger is more similar for IP
```

Unlike every other metric on this page, inner product is a similarity score,
not a lower-is-better distance. Vector magnitude affects the score; normalize
vectors first when magnitude should not affect ranking. HNSW, IVF, SPANN,
DiskANN, and quantized IP variants are also available.

### Squared L2 distance

Squared L2 measures squared Euclidean distance:

```python
points = metric_collection(
    "points_l2",
    [[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]],
    "FLAT-L2",
)

result = points.search([0.0, 1.0], k=2)
print(result.ids)
print(result.distances)  # 1.0 for both [0, 0] and [1, 1]
```

LynseDB returns the squared value and does not take the final square root.
Ranking is identical to Euclidean distance, but range-search thresholds must be
squared as well. For example, an Euclidean radius of `2.0` requires a squared
L2 threshold of `4.0`.

### Cosine distance

Cosine distance compares direction while ignoring positive vector magnitude:

```python
directions = metric_collection(
    "directions",
    [[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]],
    "FLAT-COS",
)

result = directions.search([2.0, 0.0], k=3)
print(result.ids)
print(result.distances)
```

The returned value is `1 - cosine_similarity`; identical directions have
distance zero. Use non-zero vectors when cosine direction is meaningful. For
larger collections, `HNSW-Cos`, `IVF-COS`, `SPANN-COS`, and `DiskANN-Cos` are
available.

### Manhattan distance

Manhattan, or L1, sums absolute component differences. Compared with squared
L2, one large component error has less influence relative to several smaller
errors:

```python
features = metric_collection(
    "features_l1",
    [[1.0, 2.0, 3.0], [1.0, 4.0, 3.0], [3.0, 3.0, 3.0]],
    "FLAT-L1",
)

result = features.search([1.0, 3.0, 3.0], k=3)
print(result.ids)
print(result.distances)
```

The distance is lower-is-better and unbounded. Use `HNSW-L1` for approximate
search on larger collections and tune its search breadth with `nprobe`.

## Use the newly added metrics

### Jensen–Shannon distance

Use Jensen–Shannon when each vector represents a probability distribution or
non-negative counts. LynseDB normalizes every vector internally, so counts do
not have to sum to one.

```python
topic_counts = [
    [70, 20, 10],  # mostly database
    [10, 20, 70],  # mostly frontend
    [34, 33, 33],  # mixed
]

topics = metric_collection(
    "topics",
    topic_counts,
    "FLAT-JENSEN-SHANNON",
)

result = topics.search([60, 30, 10], k=2)
print(result.ids)
print(result.distances)
```

The result is `sqrt(JS divergence)` using natural logarithms. Zero means the
normalized distributions are identical, and the maximum is `sqrt(ln(2))`
(about `0.8326`). Negative or non-finite components are invalid. Two all-zero
vectors compare as zero; one all-zero vector and one positive-mass vector are
maximally distant.

For a larger collection, switch to HNSW and tune search breadth with `nprobe`:

```python
topics.build_index("HNSW-JENSEN-SHANNON")
result = topics.search([60, 30, 10], k=10, nprobe=64)
```

Use `FLAT-JENSEN-SHANNON` as the exact baseline when measuring HNSW recall.

### Chebyshev distance

Chebyshev, also called L-infinity distance, is the largest absolute difference
in any one component. It is a useful fit when a candidate should be considered
far away as soon as one tolerance is badly violated.

```python
measurements = metric_collection(
    "measurements",
    [
        [20.0, 50.0, 100.0],
        [20.2, 49.9, 100.1],
        [21.5, 50.0, 100.0],
    ],
    "FLAT-CHEBYSHEV",
)

result = measurements.search([20.1, 50.0, 100.0], k=2)
print(result.ids)
print(result.distances)
```

Distances are in the same units as the vector components and are unbounded.
If components use different units or scales, standardize them or divide each
component by its accepted tolerance before insertion. HNSW is also available:

```python
measurements.build_index("HNSW-CHEBYSHEV")
result = measurements.search([20.1, 50.0, 100.0], k=10, nprobe=64)
```

### Canberra distance

Canberra sums the relative difference of every component. It emphasizes a
change near zero, making it useful for spectra and sparse abundance features:

```python
spectra = metric_collection(
    "spectra",
    [
        [0.0, 0.10, 0.00, 0.90],
        [0.0, 0.12, 0.00, 0.88],
        [0.2, 0.00, 0.30, 0.50],
    ],
    "FLAT-CANBERRA",
)

result = spectra.search([0.0, 0.11, 0.0, 0.89], k=2)
print(result.ids)
print(result.distances)
```

For each component LynseDB evaluates
`abs(a - b) / (abs(a) + abs(b))`; a `0 / 0` component contributes zero. For a
`d`-dimensional finite vector, the distance lies between zero and `d`.
Canberra currently supports exact Flat search only; `HNSW-CANBERRA` and
quantized variants are rejected.

### Bray–Curtis distance

Bray–Curtis compares the total absolute difference with the total abundance.
Use non-negative vectors whose dimensions represent the same species, bins, or
features:

```python
abundance = metric_collection(
    "abundance",
    [
        [12, 4, 0, 8],
        [11, 5, 0, 8],
        [1, 0, 15, 2],
    ],
    "FLAT-BRAY-CURTIS",
)

result = abundance.search([12, 5, 0, 7], k=2)
print(result.ids)
print(result.distances)
```

For non-negative inputs the distance is in the range `0..1`: zero means equal
abundance profiles and one means no shared positive abundance. Two all-zero
vectors compare as zero. Bray–Curtis currently supports exact Flat search only;
`HNSW-BRAY-CURTIS` and quantized variants are rejected.

### Use a new metric on a named vector field

The lowercase accepted names are useful when defining a named vector field;
the uppercase strings select its index implementation:

```python
item_ids = ["item-0", "item-1", "item-2"]
items = db.require_collection(
    "items",
    dim=2,
    default_index=None,
    drop_if_exists=True,
)
items.add(
    ids=item_ids,
    vectors=[[1, 0], [0, 1], [1, 1]],
)
items.commit()

items.create_vector_field(
    "topic_distribution",
    dim=3,
    metric="jensen_shannon",
)
items.add_named_vectors(
    "topic_distribution",
    vectors=topic_counts,
    ids=item_ids,
)
items.build_index(
    "FLAT-JENSEN-SHANNON",
    field_name="topic_distribution",
)

result = items.search(
    [60, 30, 10],
    k=10,
    vector_field="topic_distribution",
)
```

The IDs passed to `add_named_vectors` must already exist in the primary
collection. The field's metric is fixed when the field is created.

## Geospatial coordinates

Haversine uses GeoJSON coordinate order and returns meters:

```python
places = db.require_collection(
    "places",
    dim=2,
    default_index=None,
    drop_if_exists=True,
)
places.add(
    ids=["shanghai", "beijing"],
    vectors=[
        [121.4737, 31.2304],
        [116.4074, 39.9042],
    ],
)
places.commit()
places.build_index("HNSW-HAVERSINE")

nearest = places.search([121.50, 31.24], k=5, nprobe=64)
```

The dimension must be exactly two. Latitude must be between -90 and 90
degrees. Haversine models Earth as a mean-radius sphere; applications requiring
ellipsoidal survey precision should re-rank the returned candidates with their
preferred geodesic library.

## Binary fingerprints

Flat binary search lazily converts stored rows into packed `u64` words. The hot
representation uses one bit per dimension—up to 32x smaller than `float32`—while the
original mmap remains the durable source of truth. The packed cache is rebuilt
after writes and does not create another full-size vector copy.

All binary metrics interpret a component greater than `0.5` as one and any
other component as zero. Supplying explicit zero/one vectors makes that
contract easiest to see:

```python
binary_vectors = [
    [1, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0],
]
binary_query = [1, 1, 0, 0, 1, 0, 0, 0]

hamming = metric_collection(
    "fingerprints_hamming",
    binary_vectors,
    "FLAT-HAMMING-BINARY",
)
hamming_result = hamming.search(binary_query, k=3)

jaccard = metric_collection(
    "fingerprints_jaccard",
    binary_vectors,
    "FLAT-JACCARD-BINARY",
)
jaccard_result = jaccard.search(binary_query, k=3)

tanimoto = metric_collection(
    "fingerprints_tanimoto",
    binary_vectors,
    "FLAT-TANIMOTO-BINARY",
)
tanimoto_result = tanimoto.search(binary_query, k=3)

dice = metric_collection(
    "fingerprints_dice",
    binary_vectors,
    "FLAT-DICE-BINARY",
)
dice_result = dice.search(binary_query, k=3)

print(hamming_result.distances)
print(jaccard_result.distances)
print(tanimoto_result.distances)
print(dice_result.distances)
```

- Hamming counts positions whose bits differ, so its range is `0..dimension`.
- Jaccard returns `1 - intersection / union`.
- Tanimoto is the domain name for binary Jaccard and produces the same values.
- Dice returns `1 - 2 * intersection / (count_a + count_b)`.

For Jaccard/Tanimoto and Dice, two empty bit sets compare at distance zero.
These metrics measure fingerprint similarity, not exact molecular substructure
matching. Hamming and Jaccard also have `IVF-HAMMING-BINARY` and
`IVF-JACCARD-BINARY` variants; Tanimoto and Dice are currently Flat-only.

## Aligned profiles

Correlation distance is `1 - Pearson r`. It ignores uniform shifts and scales,
which is useful when profile shape matters more than absolute magnitude:

```python
profiles = metric_collection(
    "profiles",
    [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [5, 4, 3, 2, 1],
    ],
    "FLAT-CORRELATION",
)

result = profiles.search([2, 4, 6, 8, 10], k=3)
print(result.ids)
print(result.distances)
```

Inputs must represent aligned positions or time buckets. This metric is not
Dynamic Time Warping and does not compensate for time shifts. Two identical
constant profiles have distance zero; other comparisons involving a constant
profile return distance one instead of `NaN`. Use `HNSW-CORRELATION` with
`nprobe` for approximate search on larger collections.

## Probability distributions and histograms

Hellinger accepts non-negative counts or probabilities and normalizes each row
internally. Its result lies between zero and one:

```python
topic_weights = metric_collection(
    "topic_weights",
    [[70, 20, 10], [10, 20, 70], [34, 33, 33]],
    "FLAT-HELLINGER",
)

result = topic_weights.search([60, 30, 10], k=3)
print(result.ids)
print(result.distances)
```

Counts do not need to sum to one. Negative inputs are invalid. Hellinger is a
good default when square-root geometry for probability distributions is
appropriate; `HNSW-HELLINGER` provides the approximate alternative.

Wasserstein-1D also normalizes each row, but preserves the order of bins. Moving
mass by two bins costs twice as much as moving it by one bin:

```python
latency_histograms = metric_collection(
    "latency_histograms",
    [
        [80, 15, 5, 0],
        [10, 70, 15, 5],
        [0, 5, 15, 80],
    ],
    "FLAT-WASSERSTEIN",
)

result = latency_histograms.search([70, 20, 10, 0], k=3)
print(result.ids)
print(result.distances)
```

Wasserstein assumes equal-width ordered bins and reports distance in bin-width
units. Inputs must be non-negative, but do not need to be pre-normalized.
Resample unequal histograms to a shared bin grid before insertion. Use
`HNSW-WASSERSTEIN` with `nprobe` when approximate search is appropriate.

## Performance guidance

- L1 uses native AVX2 or NEON kernels.
- Chebyshev, Canberra, and Bray–Curtis use native AVX2 or NEON kernels and do
  not allocate per candidate.
- Flat binary metrics use packed words and hardware population counts.
- Haversine performs only the two-dimensional great-circle calculation and is
  well suited to HNSW when collections grow.
- Correlation, Hellinger, and Wasserstein scan rows without allocating temporary
  vectors. They favor memory efficiency; Hellinger's square roots make it more
  compute-intensive than L1, L2, or cosine.
- Jensen–Shannon uses AVX2/NEON vector logarithms. Exact Flat search lazily
  caches inverse mass and entropy in two `float32` values per row, then applies
  the entropy identity so the hot scan evaluates only one logarithm per
  dimension. Its dual-row kernel ranks squared distances and takes square roots
  only for the final top-k; very small distances are recomputed with the stable
  `float64` path. The cache adds eight bytes per row and no per-candidate
  allocation.
- Use the reproducible benchmark with, for example,
  `python benchmarks/flat_search_bench.py --index-mode FLAT-TANIMOTO-BINARY
  --dim 2048` on deployment hardware.
