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
| Bray–Curtis | `bray_curtis` | numeric abundance vectors | ecology, spectra and compositional profiles |
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

```python
fingerprints = db.require_collection(
    "fingerprints",
    dim=2048,
    default_index=None,
    drop_if_exists=True,
)
fingerprints.add(ids=molecule_ids, vectors=rdkit_fingerprints)
fingerprints.commit()
fingerprints.build_index("FLAT-TANIMOTO-BINARY")

matches = fingerprints.search(query_fingerprint, k=20)
```

`Tanimoto` is the domain name for binary Jaccard similarity. LynseDB returns
the corresponding distance, `1 - similarity`. This is fingerprint similarity,
not exact molecular substructure matching.

## Aligned profiles

Correlation distance is `1 - Pearson r`. It ignores uniform shifts and scales,
which is useful when profile shape matters more than absolute magnitude:

```python
profiles.build_index("HNSW-CORRELATION")
similar_shapes = profiles.search(query_profile, k=10, nprobe=64)
```

Inputs must represent aligned positions or time buckets. This metric is not
Dynamic Time Warping and does not compensate for time shifts. Two identical
constant profiles have distance zero; other comparisons involving a constant
profile return distance one instead of `NaN`.

## Probability distributions and histograms

Hellinger accepts non-negative counts or probabilities and normalizes each row
internally. Its result lies between zero and one:

```python
topics.build_index("FLAT-HELLINGER")
matches = topics.search(query_topic_weights, k=10)
```

Wasserstein-1D also normalizes each row, but preserves the order of bins. Moving
mass by two bins costs twice as much as moving it by one bin:

```python
latency_histograms.build_index("HNSW-WASSERSTEIN")
matches = latency_histograms.search(query_histogram, k=10, nprobe=64)
```

Wasserstein assumes equal-width ordered bins and reports distance in bin-width
units. Resample unequal histograms to a shared bin grid before insertion.

Jensen–Shannon also accepts counts or probabilities and normalizes each row.
LynseDB returns the square root of the divergence using natural logarithms, so
it is a distance in the range `0..sqrt(ln(2))`. It is symmetric and finite even
for disjoint support. A negative or non-finite component is rejected from
ranking; two zero-mass rows compare as zero, while a single zero-mass row is
maximally distant.

```python
predictions.build_index("HNSW-JENSEN-SHANNON")
similar_predictions = predictions.search(query_probabilities, k=10, nprobe=64)
```

## Maximum deviation and abundance data

Chebyshev returns the largest absolute component difference. It is useful when
one violated tolerance should dominate the match, and supports both exact Flat
and HNSW search:

```python
measurements.build_index("HNSW-CHEBYSHEV")
```

Canberra sums `|a-b| / (|a|+|b|)` per component; a `0/0` component contributes
zero. Bray–Curtis returns `sum(|a-b|) / sum(|a+b|)`. Both are currently exact
Flat metrics because their ANN recall behavior depends strongly on the data:

```python
spectra.build_index("FLAT-CANBERRA")
abundance.build_index("FLAT-BRAY-CURTIS")
```

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
