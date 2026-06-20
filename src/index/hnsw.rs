//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! High-performance multi-layer proximity graph for approximate nearest neighbor search.
//! O(log N) search complexity with high recall.
//!
//! Architecture inspired by usearch:
//! - Flat `Vec<Vec<u32>>` graph — cache-friendly, zero hash overhead
//! - Generation-based visited set — no allocation per search_layer call
//! - Zero-copy distance — borrow encoded_data slices, never copy vectors
//! - Parallel insertion via rayon + per-node RwLock (like usearch's threading)
//! - Pre-seeded RNG — avoid thread_rng() per call

use super::{IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use crate::distance::{compute_distance_f32, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{create_quantizer, Quantizer, QuantizerType};
use parking_lot::RwLock;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

// ─── Heap element ────────────────────────────────────────────────────────────

/// Distance-node pair ordered by distance (max-heap by default).
#[derive(Clone, Copy)]
struct DistNode {
    dist: f32,
    node: u32,
}

impl PartialEq for DistNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.node == other.node
    }
}
impl Eq for DistNode {}
impl PartialOrd for DistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

// ─── Generation-based visited set ────────────────────────────────────────────

/// Visited set that uses a generation counter to avoid per-search allocation.
/// Instead of `vec![false; n]` each call, we bump `generation` and compare.
struct VisitedSet {
    generation: Vec<u32>,
    current_gen: u32,
}

impl VisitedSet {
    fn new(capacity: usize) -> Self {
        Self {
            generation: vec![0; capacity],
            current_gen: 0,
        }
    }

    /// Reset for a new search — O(1) instead of O(n).
    #[inline(always)]
    fn reset(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            // Overflow: clear the entire array (extremely rare)
            self.generation.fill(0);
            self.current_gen = 1;
        }
    }

    /// Mark as visited. Returns true if it was already visited.
    #[inline(always)]
    fn visit(&mut self, node: u32) -> bool {
        let idx = node as usize;
        if self.generation[idx] == self.current_gen {
            true // already visited
        } else {
            self.generation[idx] = self.current_gen;
            false // first visit
        }
    }
}

// ─── Flat graph storage ──────────────────────────────────────────────────────

/// Per-level adjacency: `adj[node]` = Vec of neighbor node IDs.
/// Much faster than HashMap — contiguous memory, no hashing.
#[derive(Clone, Default, Serialize, Deserialize)]
struct LevelGraph {
    adj: Vec<Vec<u32>>,
}

impl LevelGraph {
    fn new() -> Self {
        Self { adj: Vec::new() }
    }

    fn grow(&mut self, n: usize) {
        if n > self.adj.len() {
            self.adj.resize_with(n, Vec::new);
        }
    }

    #[inline(always)]
    fn neighbors(&self, node: u32) -> &[u32] {
        &self.adj[node as usize]
    }

    #[inline(always)]
    fn set_neighbors(&mut self, node: u32, neighbors: Vec<u32>) {
        self.adj[node as usize] = neighbors;
    }

    #[inline(always)]
    fn push_neighbor(&mut self, node: u32, neighbor: u32) {
        self.adj[node as usize].push(neighbor);
    }
}

// ─── Concurrent graph for parallel build ─────────────────────────────────────

/// Per-level concurrent adjacency list for parallel HNSW construction.
/// Each node's neighbor list is protected by a RwLock: multiple search threads
/// can read concurrently, while connection updates take an exclusive write lock.
struct ConcurrentGraph {
    levels: Vec<Vec<RwLock<Vec<u32>>>>,
}

impl ConcurrentGraph {
    fn new(n_levels: usize, n_nodes: usize) -> Self {
        Self {
            levels: (0..n_levels)
                .map(|_| (0..n_nodes).map(|_| RwLock::new(Vec::new())).collect())
                .collect(),
        }
    }

    /// Convert to non-concurrent LevelGraphs after build is complete.
    fn into_level_graphs(self) -> Vec<LevelGraph> {
        self.levels
            .into_iter()
            .map(|level| LevelGraph {
                adj: level.into_iter().map(|rw| rw.into_inner()).collect(),
            })
            .collect()
    }
}

// ─── Free functions for concurrent insertion ─────────────────────────────────

#[inline(always)]
fn get_vec_raw(encoded_data: &[f32], node: u32, dim: usize) -> &[f32] {
    let start = node as usize * dim;
    unsafe { encoded_data.get_unchecked(start..start + dim) }
}

#[inline(always)]
fn distance_raw(
    encoded_data: &[f32],
    node: u32,
    query: &[f32],
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> f32 {
    let raw = compute_distance_f32(get_vec_raw(encoded_data, node, dim), query, metric);
    if ascending {
        raw
    } else {
        -raw
    }
}

fn search_layer_concurrent(
    graph: &ConcurrentGraph,
    encoded_data: &[f32],
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
    query: &[f32],
    entry: u32,
    ef: usize,
    level: usize,
    visited: &mut VisitedSet,
) -> Vec<DistNode> {
    visited.reset();
    visited.visit(entry);

    let entry_dist = distance_raw(encoded_data, entry, query, dim, metric, ascending);

    let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::with_capacity(ef * 2);
    candidates.push(Reverse(DistNode {
        dist: entry_dist,
        node: entry,
    }));

    let mut result: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef + 1);
    result.push(DistNode {
        dist: entry_dist,
        node: entry,
    });

    while let Some(Reverse(closest)) = candidates.pop() {
        let worst_dist = result.peek().map_or(f32::MAX, |x| x.dist);
        if closest.dist > worst_dist && result.len() >= ef {
            break;
        }

        // Read lock: held only for neighbor iteration, allows concurrent readers
        let neighbors = graph.levels[level][closest.node as usize].read();
        for &neighbor in neighbors.iter() {
            if visited.visit(neighbor) {
                continue;
            }

            let n_dist = distance_raw(encoded_data, neighbor, query, dim, metric, ascending);
            let worst = result.peek().map_or(f32::MAX, |x| x.dist);

            if result.len() < ef || n_dist < worst {
                let dn = DistNode {
                    dist: n_dist,
                    node: neighbor,
                };
                candidates.push(Reverse(dn));
                result.push(dn);
                if result.len() > ef {
                    result.pop();
                }
            }
        }
        // read lock dropped here
    }

    let mut res: Vec<DistNode> = result.into_vec();
    res.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    res
}

fn greedy_closest_concurrent(
    graph: &ConcurrentGraph,
    encoded_data: &[f32],
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
    query: &[f32],
    mut entry: u32,
    level: usize,
) -> u32 {
    let mut curr_dist = distance_raw(encoded_data, entry, query, dim, metric, ascending);
    loop {
        let mut changed = false;
        {
            let neighbors = graph.levels[level][entry as usize].read();
            for &neighbor in neighbors.iter() {
                let n_dist = distance_raw(encoded_data, neighbor, query, dim, metric, ascending);
                if n_dist < curr_dist {
                    entry = neighbor;
                    curr_dist = n_dist;
                    changed = true;
                }
            }
        } // read lock dropped before next iteration
        if !changed {
            break;
        }
    }
    entry
}

/// Insert a single point into the concurrent HNSW graph.
/// Called from multiple rayon threads simultaneously.
fn insert_point_concurrent(
    graph: &ConcurrentGraph,
    encoded_data: &[f32],
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
    element_levels: &[i32],
    entry_point: u32,
    max_level: i32,
    m: usize,
    m0: usize,
    ef_construction: usize,
    point_id: u32,
    visited: &mut VisitedSet,
) {
    let level = element_levels[point_id as usize];
    let point_vec = get_vec_raw(encoded_data, point_id, dim);

    let mut curr_obj = entry_point;

    // Phase 1: greedy descent through levels above point's level
    for lc in ((level as usize + 1)..=max_level as usize).rev() {
        curr_obj = greedy_closest_concurrent(
            graph,
            encoded_data,
            dim,
            metric,
            ascending,
            point_vec,
            curr_obj,
            lc,
        );
    }

    // Phase 2: ef-search + connect at levels min(level, max_level) .. 0
    let top = (level as usize).min(max_level as usize);
    for lc in (0..=top).rev() {
        let candidates = search_layer_concurrent(
            graph,
            encoded_data,
            dim,
            metric,
            ascending,
            point_vec,
            curr_obj,
            ef_construction,
            lc,
            visited,
        );
        let m_for_level = if lc == 0 { m0 } else { m };

        if let Some(first) = candidates.first() {
            curr_obj = first.node;
        }

        let neighbors: Vec<u32> = candidates
            .iter()
            .take(m_for_level)
            .map(|dn| dn.node)
            .collect();

        // Set forward connections (write lock on this node)
        *graph.levels[lc][point_id as usize].write() = neighbors.clone();

        // Add reverse connections + prune if over max
        let max_conn = if lc == 0 { m0 } else { m };
        for &neighbor in &neighbors {
            // Write lock on the neighbor node for reverse connection
            let mut nn = graph.levels[lc][neighbor as usize].write();
            nn.push(point_id);

            if nn.len() > max_conn {
                // Prune: re-score all neighbors and keep closest max_conn
                let neighbor_vec = get_vec_raw(encoded_data, neighbor, dim);
                let mut scored: Vec<DistNode> = nn
                    .iter()
                    .map(|&nb| {
                        let raw = compute_distance_f32(
                            neighbor_vec,
                            get_vec_raw(encoded_data, nb, dim),
                            metric,
                        );
                        DistNode {
                            dist: if ascending { raw } else { -raw },
                            node: nb,
                        }
                    })
                    .collect();
                if scored.len() > max_conn {
                    scored.select_nth_unstable_by(max_conn, |a, b| {
                        a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                    });
                    scored.truncate(max_conn);
                }
                *nn = scored.iter().map(|dn| dn.node).collect();
            }
            // write lock dropped here
        }
    }
}

// ─── HNSW Index ──────────────────────────────────────────────────────────────

/// HNSW index with hierarchical graph structure.
///
/// Optimized architecture:
/// - Flat `Vec<Vec<u32>>` graph per level (no HashMap overhead)
/// - Generation-based visited set (O(1) reset per search)
/// - Zero-copy distance via `get_vec()` borrow
/// - Heuristic neighbor selection for graph diversity
pub struct HNSWIndex {
    config: IndexConfig,
    quantizer: Box<dyn Quantizer>,
    /// Original data (flattened, row-major: data[i*dim..(i+1)*dim])
    data: Vec<f32>,
    /// Encoded data for distance computation (same layout)
    encoded_data: Vec<f32>,
    /// Vector IDs (external)
    ids: Vec<u64>,
    /// Graph adjacency per level: graphs[level].adj[node] = vec of neighbor u32
    graphs: Vec<LevelGraph>,
    /// Max level assigned to each element
    element_levels: Vec<i32>,
    /// Entry point node
    entry_point: Option<u32>,
    /// Current maximum level in the graph
    max_level: i32,
    /// HNSW params
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: Option<usize>,
    /// Multiplier for level generation: 1/ln(M)
    level_mult: f64,
    trained: bool,
}

impl HNSWIndex {
    pub fn new(
        metric: DistanceMetric,
        quant_type: QuantizerType,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        ml: Option<usize>,
    ) -> Self {
        let quantizer = create_quantizer(match quant_type {
            QuantizerType::None => "none",
            QuantizerType::Scalar => "sq8",
            QuantizerType::Binary => "binary",
            QuantizerType::Product => "pq",
        })
        .unwrap();

        Self {
            config: IndexConfig {
                index_type: IndexType::HNSW,
                distance_metric: metric,
                quantizer_type: quant_type,
                dimension: 0,
                params: IndexParams::HNSW {
                    m,
                    ef_construction,
                    ef_search,
                    max_level: ml,
                },
            },
            quantizer,
            data: Vec::new(),
            encoded_data: Vec::new(),
            ids: Vec::new(),
            graphs: Vec::new(),
            element_levels: Vec::new(),
            entry_point: None,
            max_level: -1,
            m,
            m0: 2 * m,
            ef_construction,
            ef_search,
            ml,
            level_mult: 1.0 / (m.max(1) as f64).ln(),
            trained: false,
        }
    }

    /// Get the vector slice for a node — zero-copy borrow from encoded_data.
    #[inline(always)]
    fn get_vec(&self, node: u32) -> &[f32] {
        let dim = self.config.dimension;
        let start = node as usize * dim;
        unsafe { self.encoded_data.get_unchecked(start..start + dim) }
    }

    /// Internal distance: always returns values where lower = more similar.
    /// For IP, we negate so the graph traversal works uniformly.
    #[inline(always)]
    fn distance(&self, a_idx: u32, b: &[f32]) -> f32 {
        let raw = compute_distance_f32(self.get_vec(a_idx), b, self.config.distance_metric);
        if self.config.distance_metric.is_ascending() {
            raw
        } else {
            -raw
        }
    }

    /// Select up to `m` nearest neighbors from pre-scored candidates.
    /// Candidates are already sorted by search_layer — just truncate.
    /// This is O(m) vs the heuristic's O(candidates × m × dim) distance calls.
    #[inline]
    fn select_neighbors(candidates: &[DistNode], m: usize) -> Vec<u32> {
        candidates.iter().take(m).map(|dn| dn.node).collect()
    }

    /// Core HNSW layer search using BinaryHeap.
    ///
    /// Returns up to `ef` nearest neighbors at the given level as scored DistNodes.
    /// Uses min-heap for candidates (pop closest) and max-heap for results (evict furthest).
    fn search_layer(
        &self,
        query: &[f32],
        entry: u32,
        ef: usize,
        level: usize,
        visited: &mut VisitedSet,
    ) -> Vec<DistNode> {
        visited.reset();
        visited.visit(entry);

        let entry_dist = self.distance(entry, query);

        // candidates: min-heap (closest first via Reverse)
        let mut candidates: BinaryHeap<Reverse<DistNode>> = BinaryHeap::with_capacity(ef * 2);
        candidates.push(Reverse(DistNode {
            dist: entry_dist,
            node: entry,
        }));

        // result: max-heap (furthest first for eviction)
        let mut result: BinaryHeap<DistNode> = BinaryHeap::with_capacity(ef + 1);
        result.push(DistNode {
            dist: entry_dist,
            node: entry,
        });

        let graph = &self.graphs[level];

        while let Some(Reverse(closest)) = candidates.pop() {
            // Stop if closest candidate is farther than worst in result
            let worst_dist = result.peek().map_or(f32::MAX, |x| x.dist);
            if closest.dist > worst_dist && result.len() >= ef {
                break;
            }

            let neighbors = graph.neighbors(closest.node);
            for &neighbor in neighbors {
                if visited.visit(neighbor) {
                    continue; // already visited
                }

                let n_dist = self.distance(neighbor, query);
                let worst = result.peek().map_or(f32::MAX, |x| x.dist);

                if result.len() < ef || n_dist < worst {
                    let dn = DistNode {
                        dist: n_dist,
                        node: neighbor,
                    };
                    candidates.push(Reverse(dn));
                    result.push(dn);
                    if result.len() > ef {
                        result.pop(); // evict furthest
                    }
                }
            }
        }

        // Return sorted by distance (closest first)
        let mut res: Vec<DistNode> = result.into_vec();
        res.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        res
    }

    /// Greedy descent: find closest node at a given level (ef=1 search).
    fn greedy_closest(&self, query: &[f32], mut entry: u32, level: usize) -> u32 {
        let mut curr_dist = self.distance(entry, query);
        let graph = &self.graphs[level];
        loop {
            let mut changed = false;
            let neighbors = graph.neighbors(entry);
            for &neighbor in neighbors {
                let n_dist = self.distance(neighbor, query);
                if n_dist < curr_dist {
                    entry = neighbor;
                    curr_dist = n_dist;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        entry
    }

    /// Insert a single point into the HNSW graph.
    ///
    /// 2-phase algorithm:
    ///   Phase 1: greedy descent from max_level down to (point_level + 1)
    ///   Phase 2: ef-search + heuristic connect at levels min(point_level, max_level) .. 0
    fn insert_point(&mut self, point_id: u32, rng: &mut SmallRng, visited: &mut VisitedSet) {
        let dim = self.config.dimension;

        // SAFETY: Zero-copy borrow of the point vector via raw pointer.
        // `encoded_data` is never reallocated during graph construction
        // (all data is loaded before `insert_point` is called), so the pointer
        // stays valid for the entire duration of this function. Only `graphs`
        // (a separate field) is mutated, so there is no actual aliasing.
        let point_vec: &[f32] = unsafe {
            let ptr = self.encoded_data.as_ptr().add(point_id as usize * dim);
            std::slice::from_raw_parts(ptr, dim)
        };

        let level = {
            let raw = (-rng.gen::<f64>().ln() * self.level_mult) as i32;
            if let Some(ml) = self.ml {
                raw.min(ml as i32)
            } else {
                raw
            }
        };
        self.element_levels[point_id as usize] = level;

        // Extend graphs if needed (rare — only when new level is created)
        if level as usize >= self.graphs.len() {
            let n = self.element_levels.len();
            while self.graphs.len() <= level as usize {
                let mut g = LevelGraph::new();
                g.grow(n);
                self.graphs.push(g);
            }
        }

        let entry = match self.entry_point {
            Some(ep) => ep,
            None => return,
        };

        let mut curr_obj = entry;

        // Phase 1: greedy descent through levels above point's level
        for lc in ((level as usize + 1)..=self.max_level as usize).rev() {
            curr_obj = self.greedy_closest(point_vec, curr_obj, lc);
        }

        // Phase 2: ef-search + connect at levels min(level, max_level) .. 0
        let top = (level as usize).min(self.max_level as usize);
        let ef_c = self.ef_construction;
        let m = self.m;
        let m0 = self.m0;
        let metric = self.config.distance_metric;
        let ascending = metric.is_ascending();

        for lc in (0..=top).rev() {
            let candidates = self.search_layer(point_vec, curr_obj, ef_c, lc, visited);
            let m_for_level = if lc == 0 { m0 } else { m };

            // Update entry for next lower level to closest found
            if let Some(first) = candidates.first() {
                curr_obj = first.node;
            }

            let neighbors = Self::select_neighbors(&candidates, m_for_level);

            // Set forward connections
            self.graphs[lc].set_neighbors(point_id, neighbors.clone());

            // Add reverse connections + prune if over max
            let max_conn = if lc == 0 { m0 } else { m };
            for &neighbor in &neighbors {
                self.graphs[lc].push_neighbor(neighbor, point_id);

                let nn = self.graphs[lc].neighbors(neighbor);
                if nn.len() > max_conn {
                    // Prune: re-score all neighbors and keep closest max_conn
                    // SAFETY: same as point_vec — only graphs are mutated, not encoded_data
                    let n_start = neighbor as usize * dim;
                    let neighbor_vec = unsafe {
                        std::slice::from_raw_parts(self.encoded_data.as_ptr().add(n_start), dim)
                    };
                    let mut scored: Vec<DistNode> = nn
                        .iter()
                        .map(|&nb| {
                            let nb_vec = unsafe {
                                std::slice::from_raw_parts(
                                    self.encoded_data.as_ptr().add(nb as usize * dim),
                                    dim,
                                )
                            };
                            let raw = compute_distance_f32(neighbor_vec, nb_vec, metric);
                            DistNode {
                                dist: if ascending { raw } else { -raw },
                                node: nb,
                            }
                        })
                        .collect();
                    // Partial sort: only need top max_conn, O(n) average
                    if scored.len() > max_conn {
                        scored.select_nth_unstable_by(max_conn, |a, b| {
                            a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
                        });
                        scored.truncate(max_conn);
                    }
                    let pruned: Vec<u32> = scored.iter().map(|dn| dn.node).collect();
                    self.graphs[lc].set_neighbors(neighbor, pruned);
                }
            }
        }

        // Update entry point if new node has higher level
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(point_id);
        }
    }
}

impl VectorIndex for HNSWIndex {
    fn build(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        ids: Option<&[u64]>,
    ) -> Result<()> {
        if !self.config.distance_metric.accepts_dimension(dim) {
            return Err(LynseError::InvalidArgument(format!(
                "metric '{}' requires dimension 2 as [longitude_degrees, latitude_degrees]",
                self.config.distance_metric.name()
            )));
        }
        self.config.dimension = dim;
        self.ids = match ids {
            Some(id_slice) => id_slice.to_vec(),
            None => (0..n_vectors as u64).collect(),
        };

        self.data = vectors.to_vec();

        // Encode data
        if self.config.quantizer_type != QuantizerType::None {
            self.quantizer.fit(vectors, n_vectors, dim)?;
            let encoded_bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            self.encoded_data = self.quantizer.decode(&encoded_bytes, n_vectors, dim)?;
        } else {
            self.encoded_data = vectors.to_vec();
        }

        if n_vectors == 0 {
            self.trained = true;
            return Ok(());
        }

        // ── Pre-assign all levels using fast RNG ──
        let mut rng = SmallRng::from_entropy();
        let level_mult = self.level_mult;
        let ml = self.ml;
        self.element_levels = (0..n_vectors)
            .map(|_| {
                let raw = (-rng.gen::<f64>().ln() * level_mult) as i32;
                if let Some(ml_val) = ml {
                    raw.min(ml_val as i32)
                } else {
                    raw
                }
            })
            .collect();

        let max_level = self.element_levels.iter().copied().max().unwrap_or(0);
        // Entry point = first node with the highest level
        let ep = self
            .element_levels
            .iter()
            .position(|&l| l == max_level)
            .unwrap() as u32;
        self.max_level = max_level;
        self.entry_point = Some(ep);

        // ── Build concurrent graph ──
        let n_levels = max_level as usize + 1;
        let concurrent_graph = ConcurrentGraph::new(n_levels, n_vectors);

        let encoded_data = &self.encoded_data;
        let element_levels = &self.element_levels;
        let metric = self.config.distance_metric;
        let ascending = metric.is_ascending();
        let m = self.m;
        let m0 = self.m0;
        let ef_c = self.ef_construction;

        // Insert entry point first (must be connected before parallel insertion)
        {
            let mut visited = VisitedSet::new(n_vectors);
            insert_point_concurrent(
                &concurrent_graph,
                encoded_data,
                dim,
                metric,
                ascending,
                element_levels,
                ep,
                max_level,
                m,
                m0,
                ef_c,
                ep,
                &mut visited,
            );
        }

        // ── Parallel insertion of all other nodes ──
        // `for_each_init` creates one VisitedSet per rayon thread (reused across tasks)
        let points: Vec<u32> = (0..n_vectors as u32).filter(|&i| i != ep).collect();
        points.into_par_iter().for_each_init(
            || VisitedSet::new(n_vectors),
            |visited, point_id| {
                insert_point_concurrent(
                    &concurrent_graph,
                    encoded_data,
                    dim,
                    metric,
                    ascending,
                    element_levels,
                    ep,
                    max_level,
                    m,
                    m0,
                    ef_c,
                    point_id,
                    visited,
                );
            },
        );

        // ── Convert concurrent graph → flat LevelGraph for fast search ──
        self.graphs = concurrent_graph.into_level_graphs();

        self.trained = true;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        if !self.trained || self.ids.is_empty() {
            return Err(LynseError::IndexNotBuilt);
        }

        let dim = self.config.dimension;
        let ef = params.ef_search.unwrap_or(self.ef_search).max(k);

        // Encode query
        let encoded_query = if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        let entry = self.entry_point.ok_or(LynseError::IndexNotBuilt)?;

        // Temporary visited set for search
        let mut visited = VisitedSet::new(self.ids.len());

        // Phase 1: greedy descent from top to level 1
        let mut curr_obj = entry;
        for level in (1..=self.max_level as usize).rev() {
            curr_obj = self.greedy_closest(&encoded_query, curr_obj, level);
        }

        // Phase 2: ef-search at level 0
        let candidates = self.search_layer(&encoded_query, curr_obj, ef, 0, &mut visited);

        if candidates.is_empty() {
            // Fallback: brute force
            let (top_idx, top_dist) = crate::distance::top_k_search(
                &encoded_query,
                &self.encoded_data,
                dim,
                k,
                self.config.distance_metric,
            );
            let result_ids: Vec<u64> = top_idx.iter().map(|&i| self.ids[i as usize]).collect();
            return Ok((result_ids, top_dist));
        }

        // Score and select top-k (candidates already sorted by search_layer)
        let take = k.min(candidates.len());
        let result_ids: Vec<u64> = candidates[..take]
            .iter()
            .map(|c| self.ids[c.node as usize])
            .collect();
        let result_dists: Vec<f32> = if self.config.distance_metric.is_ascending() {
            candidates[..take].iter().map(|c| c.dist).collect()
        } else {
            candidates[..take].iter().map(|c| -c.dist).collect()
        };

        Ok((result_ids, result_dists))
    }

    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        let id_set: HashSet<u64> = ids.iter().cloned().collect();
        let indices_to_delete: HashSet<usize> = self
            .ids
            .iter()
            .enumerate()
            .filter(|(_, id)| id_set.contains(id))
            .map(|(i, _)| i)
            .collect();

        if indices_to_delete.is_empty() {
            return Ok(());
        }

        // Build index remap: old_idx → new_idx
        let mut index_map: Vec<u32> = vec![u32::MAX; self.ids.len()];
        let mut new_idx = 0u32;
        for old_idx in 0..self.ids.len() {
            if !indices_to_delete.contains(&old_idx) {
                index_map[old_idx] = new_idx;
                new_idx += 1;
            }
        }

        // Update graphs
        let new_count = new_idx as usize;
        for level in 0..self.graphs.len() {
            let mut new_graph = LevelGraph::new();
            new_graph.grow(new_count);
            for old_node in 0..self.graphs[level].adj.len() {
                if indices_to_delete.contains(&old_node) {
                    continue;
                }
                let new_node = index_map[old_node];
                let new_neighbors: Vec<u32> = self.graphs[level]
                    .neighbors(old_node as u32)
                    .iter()
                    .filter(|&&n| !indices_to_delete.contains(&(n as usize)))
                    .map(|&n| index_map[n as usize])
                    .collect();
                new_graph.set_neighbors(new_node, new_neighbors);
            }
            self.graphs[level] = new_graph;
        }

        // Update entry point
        if let Some(ep) = self.entry_point {
            if indices_to_delete.contains(&(ep as usize)) {
                // Pick first non-deleted node
                self.entry_point = (0..self.ids.len())
                    .find(|i| !indices_to_delete.contains(i))
                    .map(|i| index_map[i]);
            } else {
                self.entry_point = Some(index_map[ep as usize]);
            }
        }

        // Remove data
        let dim = self.config.dimension;
        let mut new_data = Vec::with_capacity(new_count * dim);
        let mut new_encoded = Vec::with_capacity(new_count * dim);
        let mut new_ids = Vec::with_capacity(new_count);
        let mut new_levels = Vec::with_capacity(new_count);

        for (i, &id) in self.ids.iter().enumerate() {
            if !id_set.contains(&id) {
                let start = i * dim;
                new_data.extend_from_slice(&self.data[start..start + dim]);
                new_encoded.extend_from_slice(&self.encoded_data[start..start + dim]);
                new_ids.push(id);
                new_levels.push(self.element_levels[i]);
            }
        }

        self.data = new_data;
        self.encoded_data = new_encoded;
        self.ids = new_ids;
        self.element_levels = new_levels;

        Ok(())
    }

    fn insert(&mut self, vectors: &[f32], n_vectors: usize, dim: usize, ids: &[u64]) -> Result<()> {
        if dim != self.config.dimension {
            return Err(LynseError::DimensionMismatch {
                expected: self.config.dimension,
                got: dim,
            });
        }

        let old_count = self.ids.len();
        self.data.extend_from_slice(vectors);
        self.ids.extend_from_slice(ids);

        if self.config.quantizer_type != QuantizerType::None {
            let bytes = self.quantizer.encode(vectors, n_vectors, dim)?;
            let decoded = self.quantizer.decode(&bytes, n_vectors, dim)?;
            self.encoded_data.extend_from_slice(&decoded);
        } else {
            self.encoded_data.extend_from_slice(vectors);
        }

        self.element_levels.resize(old_count + n_vectors, -1);

        let mut rng = SmallRng::from_entropy();
        let mut visited = VisitedSet::new(old_count + n_vectors);

        for i in 0..n_vectors {
            self.insert_point((old_count + i) as u32, &mut rng, &mut visited);
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn config(&self) -> &IndexConfig {
        &self.config
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let state = HNSWState {
            data: self.data.clone(),
            encoded_data: self.encoded_data.clone(),
            ids: self.ids.clone(),
            graphs: self.graphs.clone(),
            element_levels: self.element_levels.clone(),
            entry_point: self.entry_point,
            max_level: self.max_level,
            config: self.config.clone(),
            m: self.m,
            m0: self.m0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            ml: self.ml,
            level_mult: self.level_mult,
            trained: self.trained,
        };
        bincode::serialize(&state).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let state: HNSWState =
            bincode::deserialize(data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        self.data = state.data;
        self.encoded_data = state.encoded_data;
        self.ids = state.ids;
        self.graphs = state.graphs;
        self.element_levels = state.element_levels;
        self.entry_point = state.entry_point;
        self.max_level = state.max_level;
        self.config = state.config;
        self.m = state.m;
        self.m0 = state.m0;
        self.ef_construction = state.ef_construction;
        self.ef_search = state.ef_search;
        self.ml = state.ml;
        self.level_mult = state.level_mult;
        self.trained = state.trained;
        Ok(())
    }

    fn name(&self) -> String {
        let metric = self.config.distance_metric.name();
        let quant = match self.config.quantizer_type {
            QuantizerType::None => "",
            QuantizerType::Scalar => "-sq8",
            QuantizerType::Binary => "-binary",
            QuantizerType::Product => "-pq",
        };
        format!("hnsw-{}{}", metric, quant)
    }
}

#[derive(Serialize, Deserialize)]
struct HNSWState {
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    graphs: Vec<LevelGraph>,
    element_levels: Vec<i32>,
    entry_point: Option<u32>,
    max_level: i32,
    config: IndexConfig,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    ml: Option<usize>,
    level_mult: f64,
    trained: bool,
}
