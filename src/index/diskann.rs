//! DiskANN / Vamana graph index.
//!
//! Build uses full-precision Vamana in memory, then (when a data directory is
//! attached) flushes the adjacency list to SSD and keeps PQ codes in RAM for
//! ADC beam search. Exact re-ranking reuses the collection [`VectorStore`]
//! via `uses_store_rescore` — vectors are not duplicated into the index.

use super::{IndexConfig, IndexParams, IndexType, SearchParams, VectorIndex};
use crate::distance::{compute_distance_f32, top_k_search, DistanceMetric};
use crate::error::{LynseError, Result};
use crate::quantizer::{self, Quantizer, QuantizerType};
use crate::storage::diskann_graph::DiskGraphStore;
use crate::storage::pq_mmap::{parse_n_subspaces, PQIndex, DEFAULT_OVERSAMPLE};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};
use std::io::{stderr, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Soft ceiling on construction beam width (paper uses L ≫ R; avoid unbounded RAM).
/// Search still uses full `l` / nprobe.
const L_BUILD_CAP: usize = 128;

/// IP-DiskANN delete search list size (paper `C`).
const IP_DELETE_L: usize = 128;
/// IP-DiskANN candidate pool for edge replacement (paper `L_c`).
const IP_CANDIDATE_L: usize = 50;
/// Edges copied per endpoint during in-place delete (paper `σ`).
const IP_SIGMA: usize = 3;
/// Trigger lightweight dangling-edge cleanup after this deleted/live ratio.
const IP_DANGLING_RATIO: f32 = 0.20;

/// Ranked graph node for beam search (lower rank = closer).
#[derive(Clone, Copy)]
struct RankNode {
    rank: f32,
    raw: f32,
    idx: usize,
}

impl PartialEq for RankNode {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.idx == other.idx
    }
}
impl Eq for RankNode {}
impl PartialOrd for RankNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank
            .partial_cmp(&other.rank)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// Generation-tagged visited set (avoids clearing a bitvec every search).
struct VisitedSet {
    generation: Vec<u32>,
    current: u32,
}

impl VisitedSet {
    fn new(capacity: usize) -> Self {
        Self {
            generation: vec![0; capacity],
            current: 1,
        }
    }

    fn ensure_capacity(&mut self, capacity: usize) {
        if self.generation.len() < capacity {
            self.generation.resize(capacity, 0);
        }
    }

    fn reset(&mut self) {
        if self.current == u32::MAX {
            self.generation.fill(0);
            self.current = 1;
        } else {
            self.current += 1;
        }
    }

    /// Returns true if newly marked, false if already visited this generation.
    #[inline]
    fn try_visit(&mut self, idx: usize) -> bool {
        if self.generation[idx] == self.current {
            return false;
        }
        self.generation[idx] = self.current;
        true
    }
}

thread_local! {
    static SEARCH_VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new(0));
    static BUILD_VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new(0));
}

// ─── Full-precision batched Vamana build helpers ─────────────────────────────

#[inline(always)]
fn diskann_vec(data: &[f32], idx: u32, dim: usize) -> &[f32] {
    let start = idx as usize * dim;
    unsafe { data.get_unchecked(start..start + dim) }
}

#[inline(always)]
fn diskann_rank(
    data: &[f32],
    idx: u32,
    query: &[f32],
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> f32 {
    let raw = compute_distance_f32(diskann_vec(data, idx, dim), query, metric);
    if ascending {
        raw
    } else {
        -raw
    }
}

#[inline(always)]
fn diskann_rank_between(
    data: &[f32],
    a: u32,
    b: u32,
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> f32 {
    diskann_rank(data, a, diskann_vec(data, b, dim), dim, metric, ascending)
}

/// Free-standing robust prune (same α-RNG rules as DiskANNIndex::robust_prune).
fn robust_prune_raw(
    data: &[f32],
    p: u32,
    candidates: &[(f32, u32)],
    degree: usize,
    alpha: f32,
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> Vec<u32> {
    let degree = degree.max(1);
    let alpha = alpha.max(1.0);
    let ln_alpha = if ascending || (alpha - 1.0).abs() <= f32::EPSILON {
        0.0
    } else {
        alpha.ln()
    };

    let mut pool: Vec<(f32, u32)> = Vec::with_capacity(candidates.len());
    let mut seen = HashSet::with_capacity(candidates.len());
    for &(rank, idx) in candidates {
        if idx == p || !seen.insert(idx) {
            continue;
        }
        pool.push((rank, idx));
    }
    pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut selected: Vec<u32> = Vec::with_capacity(degree);
    let mut remaining = pool;
    while !remaining.is_empty() && selected.len() < degree {
        let best = remaining[0].1;
        selected.push(best);
        remaining.retain(|&(dist_p_v, v)| {
            if v == best {
                return false;
            }
            let dist_best_v = diskann_rank_between(data, best, v, dim, metric, ascending);
            if ascending {
                alpha * dist_best_v > dist_p_v
            } else {
                dist_p_v < dist_best_v + ln_alpha
            }
        });
    }
    selected
}

/// Fast reverse-edge prune: keep the closest `degree` neighbors (HNSW-style).
fn prune_closest_raw(
    data: &[f32],
    center: u32,
    neighbors: &[u32],
    degree: usize,
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
) -> Vec<u32> {
    if neighbors.len() <= degree {
        return neighbors.to_vec();
    }
    let mut scored: Vec<(f32, u32)> = neighbors
        .iter()
        .map(|&nb| {
            (
                diskann_rank_between(data, center, nb, dim, metric, ascending),
                nb,
            )
        })
        .collect();
    if scored.len() > degree {
        scored.select_nth_unstable_by(degree - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        });
        scored.truncate(degree);
    }
    scored.into_iter().map(|(_, n)| n).collect()
}

#[allow(dead_code)]
fn take_closest_ranked(cand: &[(f32, u32)], point: u32, degree: usize) -> Vec<u32> {
    let mut pool = cand.to_vec();
    pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let mut out = Vec::with_capacity(degree);
    let mut seen = HashSet::with_capacity(degree * 2);
    seen.insert(point);
    for (_, nb) in pool {
        if seen.insert(nb) {
            out.push(nb);
            if out.len() >= degree {
                break;
            }
        }
    }
    out
}

/// Total-order wrapper so f32 ranks can live in BinaryHeap.
#[derive(Clone, Copy)]
struct OrderedF32(f32);

impl PartialEq for OrderedF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

fn search_beam_snapshot(
    graph: &[Vec<u32>],
    data: &[f32],
    dim: usize,
    metric: DistanceMetric,
    ascending: bool,
    query: &[f32],
    entry: u32,
    ef: usize,
    visited: &mut VisitedSet,
) -> Vec<(f32, u32)> {
    let n = graph.len();
    if n == 0 {
        return Vec::new();
    }
    let ef = ef.max(1);
    visited.ensure_capacity(n);
    visited.reset();
    visited.try_visit(entry as usize);

    let entry_rank = diskann_rank(data, entry, query, dim, metric, ascending);
    let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> =
        BinaryHeap::with_capacity(ef * 2);
    candidates.push(Reverse((OrderedF32(entry_rank), entry)));
    let mut result: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::with_capacity(ef + 1);
    result.push((OrderedF32(entry_rank), entry));

    while let Some(Reverse((OrderedF32(closest_rank), closest))) = candidates.pop() {
        let worst = result.peek().map_or(f32::MAX, |x| x.0 .0);
        if result.len() >= ef && closest_rank > worst {
            break;
        }
        for &neighbor in &graph[closest as usize] {
            if !visited.try_visit(neighbor as usize) {
                continue;
            }
            let n_rank = diskann_rank(data, neighbor, query, dim, metric, ascending);
            let worst = result.peek().map_or(f32::MAX, |x| x.0 .0);
            if result.len() < ef || n_rank < worst {
                candidates.push(Reverse((OrderedF32(n_rank), neighbor)));
                result.push((OrderedF32(n_rank), neighbor));
                if result.len() > ef {
                    result.pop();
                }
            }
        }
    }

    let mut out: Vec<(f32, u32)> = result.into_iter().map(|(r, i)| (r.0, i)).collect();
    out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    out
}

/// DiskANN / Vamana graph-based index.
pub struct DiskANNIndex {
    config: IndexConfig,
    quantizer: Box<dyn Quantizer>,
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    /// Soft-deleted graph slots (index aligned with `ids` / VectorStore rows).
    deleted_mask: Vec<bool>,
    /// Out-neighbor adjacency list (memory; cleared after layered flush)
    graph: Vec<Vec<usize>>,
    entry_point: Option<usize>,
    r: usize,
    l: usize,
    alpha: f32,
    max_degree: usize,
    trained: bool,
    /// Collection root; layered sidecars live in `{data_dir}/diskann/`.
    data_dir: Option<PathBuf>,
    pq: Option<PQIndex>,
    disk_graph: Option<DiskGraphStore>,
    /// True when graph+PQ live on diskann/ sidecars (no in-memory f32/graph).
    layered: bool,
    /// Original mode alias (e.g. DISKANN-IP-PQ8) for PQ subspace parsing.
    index_alias: String,
    /// Soft-deleted slots since last Alg-6 dangling cleanup.
    pending_dangling: usize,
}

impl DiskANNIndex {
    pub fn new(
        metric: DistanceMetric,
        quant_type: QuantizerType,
        r: usize,
        l: usize,
        alpha: f32,
        max_degree: usize,
    ) -> Self {
        Self::with_alias(metric, quant_type, r, l, alpha, max_degree, "DISKANN")
    }

    pub fn with_alias(
        metric: DistanceMetric,
        quant_type: QuantizerType,
        r: usize,
        l: usize,
        alpha: f32,
        max_degree: usize,
        index_alias: &str,
    ) -> Self {
        let quantizer = quantizer::create_quantizer(match quant_type {
            QuantizerType::None => "none",
            QuantizerType::Scalar => "sq8",
            QuantizerType::Binary => "binary",
            QuantizerType::Product => "pq",
        })
        .unwrap();

        let r = r.max(1);
        let max_degree = max_degree.max(r);

        Self {
            config: IndexConfig {
                index_type: IndexType::DiskANN,
                distance_metric: metric,
                quantizer_type: quant_type,
                dimension: 0,
                params: IndexParams::DiskANN {
                    r,
                    l: l.max(r),
                    alpha: alpha.max(1.0),
                    max_degree,
                },
            },
            quantizer,
            data: Vec::new(),
            encoded_data: Vec::new(),
            ids: Vec::new(),
            deleted_mask: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            r,
            l: l.max(r),
            alpha: alpha.max(1.0),
            max_degree,
            trained: false,
            data_dir: None,
            pq: None,
            disk_graph: None,
            layered: false,
            index_alias: index_alias.to_ascii_uppercase(),
            pending_dangling: 0,
        }
    }

    #[inline]
    fn wants_layered(&self) -> bool {
        matches!(
            self.config.quantizer_type,
            QuantizerType::None | QuantizerType::Product
        )
    }

    fn diskann_dir(&self) -> Option<PathBuf> {
        self.data_dir.as_ref().map(|d| d.join("diskann"))
    }

    fn flush_layered(&mut self) -> Result<()> {
        let dir = self
            .diskann_dir()
            .ok_or_else(|| LynseError::InvalidArgument("DiskANN data_dir not attached".into()))?;
        std::fs::create_dir_all(&dir).map_err(|e| LynseError::Storage(e.to_string()))?;
        let degree = self.r.min(self.max_degree);
        let n = self.ids.len();
        let dim = self.config.dimension;
        if n == 0 || self.graph.len() != n {
            return Err(LynseError::IndexNotBuilt);
        }
        let graph_path = dir.join("graph.bin");
        let t_g = Instant::now();
        let disk_graph = DiskGraphStore::write_from_adj(&graph_path, &self.graph, degree)?;
        let write_s = t_g.elapsed().as_secs_f64();
        // Layered DiskANN: PQ is only a candidate generator; exact re-rank uses
        // VectorStore. Honor PQ8/PQ16 from the index alias; K=256 matches classic PQ.
        let n_subspaces = parse_n_subspaces(&self.index_alias, dim);
        let t_pq = Instant::now();
        let pq = PQIndex::build_with_clusters(&self.encoded_data, n, dim, n_subspaces, 256);
        let pq_s = t_pq.elapsed().as_secs_f64();
        let pq_path = dir.join("pq.bin");
        pq.save(&pq_path)
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        let _ = writeln!(
            stderr(),
            "diskann flush detail: graph_write={:.1}s pq_build={:.1}s (M={}, K=256)",
            write_s, pq_s, n_subspaces
        );
        self.disk_graph = Some(disk_graph);
        self.pq = Some(pq);
        self.data.clear();
        self.encoded_data.clear();
        self.graph.clear();
        self.layered = true;
        self.deleted_mask = vec![false; n];
        self.pending_dangling = 0;
        Ok(())
    }

    #[inline]
    fn is_live(&self, idx: usize) -> bool {
        idx < self.ids.len() && (idx >= self.deleted_mask.len() || !self.deleted_mask[idx])
    }

    fn live_entry(&self) -> Option<usize> {
        if let Some(ep) = self.entry_point {
            if self.is_live(ep) {
                return Some(ep);
            }
        }
        (0..self.ids.len()).find(|&i| self.is_live(i))
    }

    fn out_neighbors(&self, idx: usize) -> Vec<usize> {
        if self.layered {
            self.disk_graph
                .as_ref()
                .map(|g| {
                    g.neighbors(idx)
                        .into_iter()
                        .map(|x| x as usize)
                        .filter(|&n| self.is_live(n))
                        .collect()
                })
                .unwrap_or_default()
        } else if idx < self.graph.len() {
            self.graph[idx]
                .iter()
                .copied()
                .filter(|&n| self.is_live(n))
                .collect()
        } else {
            Vec::new()
        }
    }

    fn write_out_neighbors(&mut self, idx: usize, neighbors: &[usize]) -> Result<()> {
        let degree = self.r.min(self.max_degree);
        if self.layered {
            let raw: Vec<u32> = neighbors
                .iter()
                .take(degree)
                .map(|&n| n as u32)
                .collect();
            if let Some(g) = self.disk_graph.as_mut() {
                g.set_neighbors(idx, &raw)?;
            }
        } else if idx < self.graph.len() {
            self.graph[idx] = neighbors.iter().copied().take(degree).collect();
        }
        Ok(())
    }

    /// Rank distance using in-memory encoded data or PQ reconstruction.
    fn rank_between_nodes(&self, a: usize, b: usize) -> f32 {
        if !self.layered {
            return self.rank_between(a, b);
        }
        let Some(pq) = self.pq.as_ref() else {
            return f32::MAX;
        };
        let Some(va) = pq.reconstruct(a) else {
            return f32::MAX;
        };
        let Some(vb) = pq.reconstruct(b) else {
            return f32::MAX;
        };
        let raw = compute_distance_f32(&va, &vb, self.config.distance_metric);
        if self.config.distance_metric.is_ascending() {
            raw
        } else {
            -raw
        }
    }

    fn robust_prune_nodes(
        &self,
        p: usize,
        candidates: &[(f32, usize)],
        degree: usize,
        alpha: f32,
    ) -> Vec<usize> {
        let degree = degree.max(1);
        let alpha = alpha.max(1.0);
        let ascending = self.config.distance_metric.is_ascending();
        let ln_alpha = if ascending || (alpha - 1.0).abs() <= f32::EPSILON {
            0.0
        } else {
            alpha.ln()
        };

        let mut pool: Vec<(f32, usize)> = Vec::with_capacity(candidates.len());
        let mut seen = HashSet::with_capacity(candidates.len());
        for &(rank, idx) in candidates {
            if idx == p || !self.is_live(idx) || !seen.insert(idx) {
                continue;
            }
            pool.push((rank, idx));
        }
        pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut selected: Vec<usize> = Vec::with_capacity(degree);
        let mut remaining = pool;
        while !remaining.is_empty() && selected.len() < degree {
            let best = remaining[0].1;
            selected.push(best);
            remaining.retain(|&(dist_p_v, v)| {
                if v == best {
                    return false;
                }
                let dist_best_v = self.rank_between_nodes(best, v);
                if ascending {
                    alpha * dist_best_v > dist_p_v
                } else {
                    dist_p_v < dist_best_v + ln_alpha
                }
            });
        }
        selected
    }

    fn link_bidirectional_ip(&mut self, src: usize, neighbors: &[usize], alpha: f32) -> Result<()> {
        let degree = self.r.min(self.max_degree);
        for &dst in neighbors {
            if dst == src || !self.is_live(dst) {
                continue;
            }
            let mut nbrs = self.out_neighbors(dst);
            if !nbrs.contains(&src) {
                nbrs.push(src);
            }
            if nbrs.len() > degree {
                let cand: Vec<(f32, usize)> = nbrs
                    .iter()
                    .map(|&n| (self.rank_between_nodes(dst, n), n))
                    .collect();
                nbrs = self.robust_prune_nodes(dst, &cand, degree, alpha);
            }
            self.write_out_neighbors(dst, &nbrs)?;
        }
        Ok(())
    }

    fn closest_sigma_in(
        &self,
        center: usize,
        candidates: &[(f32, usize)],
        sigma: usize,
        exclude: usize,
    ) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&(_, idx)| {
                if idx == exclude || idx == center || !self.is_live(idx) {
                    (f32::MAX, idx)
                } else {
                    (self.rank_between_nodes(center, idx), idx)
                }
            })
            .filter(|(d, _)| d.is_finite() && *d < f32::MAX / 2.0)
            .collect();
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scored.truncate(sigma.max(1));
        scored.into_iter().map(|(_, i)| i).collect()
    }

    /// IP-DiskANN Algorithm 6: strip edges to deleted nodes (no distance calc).
    fn consolidate_dangling_edges(&mut self) -> Result<()> {
        let n = self.ids.len();
        let degree = self.r.min(self.max_degree);
        for i in 0..n {
            if !self.is_live(i) {
                continue;
            }
            let nbrs: Vec<usize> = if self.layered {
                self.disk_graph
                    .as_ref()
                    .map(|g| {
                        g.neighbors(i)
                            .into_iter()
                            .map(|x| x as usize)
                            .filter(|&nb| self.is_live(nb))
                            .take(degree)
                            .collect()
                    })
                    .unwrap_or_default()
            } else if i < self.graph.len() {
                self.graph[i]
                    .iter()
                    .copied()
                    .filter(|&nb| self.is_live(nb))
                    .take(degree)
                    .collect()
            } else {
                Vec::new()
            };
            self.write_out_neighbors(i, &nbrs)?;
        }
        self.pending_dangling = 0;
        if let Some(g) = self.disk_graph.as_ref() {
            g.flush()?;
        }
        if let (Some(pq), Some(dir)) = (self.pq.as_ref(), self.diskann_dir()) {
            pq.save(&dir.join("pq.bin"))
                .map_err(|e| LynseError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    fn live_count(&self) -> usize {
        if self.deleted_mask.len() != self.ids.len() {
            return self.ids.len();
        }
        self.deleted_mask.iter().filter(|&&d| !d).count()
    }

    fn maybe_consolidate_dangling(&mut self) -> Result<()> {
        let live = self.live_count().max(1);
        let deleted = self.ids.len().saturating_sub(live);
        if deleted as f32 / live as f32 >= IP_DANGLING_RATIO || self.pending_dangling >= live / 5 {
            self.consolidate_dangling_edges()?;
        }
        Ok(())
    }

    /// Persist layered PQ sidecar after incremental code updates.
    fn persist_layered_pq(&self) -> Result<()> {
        if let (Some(pq), Some(dir)) = (self.pq.as_ref(), self.diskann_dir()) {
            pq.save(&dir.join("pq.bin"))
                .map_err(|e| LynseError::Storage(e.to_string()))?;
        }
        if let Some(g) = self.disk_graph.as_ref() {
            g.flush()?;
        }
        Ok(())
    }

    fn open_layered_sidecars(&mut self) -> Result<()> {
        let dir = match self.diskann_dir() {
            Some(d) => d,
            None => return Ok(()),
        };
        let n = self.ids.len();
        if n == 0 {
            return Ok(());
        }
        let degree = self.r.min(self.max_degree);
        let graph_path = dir.join("graph.bin");
        let pq_path = dir.join("pq.bin");
        if !graph_path.exists() || !pq_path.exists() {
            return Err(LynseError::Storage(
                "DiskANN layered sidecars missing (graph.bin / pq.bin)".into(),
            ));
        }
        self.disk_graph = Some(DiskGraphStore::open(&graph_path, n, degree)?);
        self.pq = Some(
            PQIndex::load(&pq_path).map_err(|e| LynseError::Storage(e.to_string()))?,
        );
        if self.deleted_mask.len() != n {
            self.deleted_mask = vec![false; n];
        }
        self.layered = true;
        self.data.clear();
        self.encoded_data.clear();
        self.graph.clear();
        Ok(())
    }

    /// PQ + disk-graph beam search. Returns (adc_raw, idx) sorted best-first.
    fn search_graph_pq(
        &self,
        query: &[f32],
        ef: usize,
        subset: Option<&HashSet<usize>>,
        visited: &mut VisitedSet,
        entry: usize,
    ) -> Vec<(f32, usize)> {
        let pq = match self.pq.as_ref() {
            Some(p) => p,
            None => return Vec::new(),
        };
        let disk = match self.disk_graph.as_ref() {
            Some(g) => g,
            None => return Vec::new(),
        };
        let n = self.ids.len();
        if n == 0 {
            return Vec::new();
        }
        let ef = ef.max(1);
        let ascending = self.config.distance_metric.is_ascending();
        let lut = pq.adc_lut(query, self.config.distance_metric);
        visited.ensure_capacity(n);
        visited.reset();

        let entry_raw = pq.adc_raw(entry, &lut);
        let entry_rank = if ascending { entry_raw } else { -entry_raw };
        let mut candidates: BinaryHeap<Reverse<RankNode>> = BinaryHeap::with_capacity(ef * 2);
        if self.is_live(entry) {
            candidates.push(Reverse(RankNode {
                rank: entry_rank,
                raw: entry_raw,
                idx: entry,
            }));
            visited.try_visit(entry);
        }

        let mut result: BinaryHeap<RankNode> = BinaryHeap::with_capacity(ef + 1);
        let entry_ok = self.is_live(entry) && subset.map(|s| s.contains(&entry)).unwrap_or(true);
        if entry_ok {
            result.push(RankNode {
                rank: entry_rank,
                raw: entry_raw,
                idx: entry,
            });
        }

        while let Some(Reverse(closest)) = candidates.pop() {
            let worst_rank = result.peek().map_or(f32::MAX, |node| node.rank);
            if result.len() >= ef && closest.rank > worst_rank {
                break;
            }
            let neighbors = disk.neighbors(closest.idx);
            let prefetch_ids: Vec<usize> = neighbors.iter().map(|&x| x as usize).collect();
            disk.prefetch(&prefetch_ids);
            for &neighbor_u32 in &neighbors {
                let neighbor = neighbor_u32 as usize;
                if !self.is_live(neighbor) || !visited.try_visit(neighbor) {
                    continue;
                }
                let n_raw = pq.adc_raw(neighbor, &lut);
                let n_rank = if ascending { n_raw } else { -n_raw };
                let in_subset = subset.map(|s| s.contains(&neighbor)).unwrap_or(true);
                let worst = result.peek().map_or(f32::MAX, |node| node.rank);
                let explore = result.len() < ef || n_rank < worst || !in_subset;
                if explore {
                    candidates.push(Reverse(RankNode {
                        rank: n_rank,
                        raw: n_raw,
                        idx: neighbor,
                    }));
                }
                if in_subset && (result.len() < ef || n_rank < worst) {
                    result.push(RankNode {
                        rank: n_rank,
                        raw: n_raw,
                        idx: neighbor,
                    });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out: Vec<(f32, usize)> = result.into_iter().map(|n| (n.raw, n.idx)).collect();
        if ascending {
            out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            out.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }
        out
    }

    #[inline]
    fn vec_at(&self, idx: usize) -> &[f32] {
        let dim = self.config.dimension;
        let start = idx * dim;
        &self.encoded_data[start..start + dim]
    }

    #[inline]
    fn raw_distance(&self, a_idx: usize, b: &[f32]) -> f32 {
        compute_distance_f32(self.vec_at(a_idx), b, self.config.distance_metric)
    }

    /// Lower always means closer (IP is negated).
    #[inline]
    fn rank_distance(&self, a_idx: usize, b: &[f32]) -> f32 {
        let raw = self.raw_distance(a_idx, b);
        if self.config.distance_metric.is_ascending() {
            raw
        } else {
            -raw
        }
    }

    #[inline]
    fn rank_between(&self, a_idx: usize, b_idx: usize) -> f32 {
        self.rank_distance(a_idx, self.vec_at(b_idx))
    }

    /// Approximate medoid: minimize average distance to a random sample.
    fn choose_medoid(&self, rng: &mut impl Rng) -> usize {
        let n = self.ids.len();
        if n <= 1 {
            return 0;
        }
        let sample_refs: usize = 32.min(n);
        let candidates: usize = 64.min(n);
        let mut refs: Vec<usize> = (0..n).collect();
        refs.shuffle(rng);
        refs.truncate(sample_refs);

        let mut cand: Vec<usize> = (0..n).collect();
        cand.shuffle(rng);
        cand.truncate(candidates);

        let mut best = cand[0];
        let mut best_sum = f32::INFINITY;
        for &c in &cand {
            let mut sum = 0.0f32;
            for &r in &refs {
                sum += self.rank_between(c, r);
            }
            if sum < best_sum {
                best_sum = sum;
                best = c;
            }
        }
        best
    }

    /// Initialize each node with up to `r` random out-neighbors (Vamana start).
    #[allow(dead_code)]
    fn init_random_graph(&mut self, rng: &mut impl Rng) {
        let n = self.ids.len();
        self.graph = vec![Vec::with_capacity(self.r); n];
        if n <= 1 {
            return;
        }
        let degree = self.r.min(n - 1);
        for i in 0..n {
            let mut chosen = Vec::with_capacity(degree);
            // Rejection sampling — O(R) expected, avoids O(n²) buffer fills.
            while chosen.len() < degree {
                let j = rng.gen_range(0..n);
                if j != i && !chosen.contains(&j) {
                    chosen.push(j);
                }
            }
            self.graph[i] = chosen;
        }
    }

    /// Vamana robust prune: keep up to `degree` α-RNG neighbors of `p`.
    ///
    /// `candidates` are (rank_distance(p, ·), neighbor_idx). Existing out-neighbors
    /// of `p` should already be merged into the candidate set by the caller.
    ///
    /// For ascending metrics (L2, cosine distance, …) this is the standard
    /// DiskANN rule: keep `v` iff `α · d(best, v) > d(p, v)`.
    ///
    /// For Inner Product, ranks are negative (`−IP`) so multiplying by `α > 1`
    /// *inverts* the long-range intent and collapses out-degree (~1). We apply
    /// the mathematically equivalent α-RNG on `exp(rank)` in log-space:
    /// keep `v` iff `rank(p,v) < rank(best,v) + ln(α)`.
    fn robust_prune(&self, p: usize, candidates: &[(f32, usize)], degree: usize, alpha: f32) -> Vec<usize> {
        let degree = degree.max(1);
        let alpha = alpha.max(1.0);
        let ascending = self.config.distance_metric.is_ascending();
        let ln_alpha = if ascending || (alpha - 1.0).abs() <= f32::EPSILON {
            0.0
        } else {
            alpha.ln()
        };

        let mut pool: Vec<(f32, usize)> = Vec::with_capacity(candidates.len());
        let mut seen = HashSet::with_capacity(candidates.len());
        for &(rank, idx) in candidates {
            if idx == p || !seen.insert(idx) {
                continue;
            }
            pool.push((rank, idx));
        }
        pool.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut selected: Vec<usize> = Vec::with_capacity(degree);
        let mut remaining = pool;
        while !remaining.is_empty() && selected.len() < degree {
            // Closest remaining candidate (pool pre-sorted; re-sort after retain).
            let best = remaining[0].1;
            selected.push(best);
            remaining.retain(|&(dist_p_v, v)| {
                if v == best {
                    return false;
                }
                let dist_best_v = self.rank_between(best, v);
                if ascending {
                    alpha * dist_best_v > dist_p_v
                } else {
                    dist_p_v < dist_best_v + ln_alpha
                }
            });
        }
        selected
    }

    /// Best-first beam search. Returns up to `ef` (raw_distance, idx) pairs,
    /// sorted best-first. Filtered mode explores bridge nodes outside the
    /// subset but only returns in-subset hits.
    fn search_graph(
        &self,
        query: &[f32],
        ef: usize,
        subset: Option<&HashSet<usize>>,
        visited: &mut VisitedSet,
        entry: usize,
    ) -> Vec<(f32, usize)> {
        let n = self.ids.len();
        if n == 0 {
            return Vec::new();
        }
        let ef = ef.max(1);
        let ascending = self.config.distance_metric.is_ascending();
        visited.ensure_capacity(n);
        visited.reset();

        let entry_raw = self.raw_distance(entry, query);
        let entry_rank = if ascending { entry_raw } else { -entry_raw };
        let mut candidates: BinaryHeap<Reverse<RankNode>> = BinaryHeap::with_capacity(ef * 2);
        if self.is_live(entry) {
            candidates.push(Reverse(RankNode {
                rank: entry_rank,
                raw: entry_raw,
                idx: entry,
            }));
            visited.try_visit(entry);
        }

        // Max-heap of accepted (in-subset) results by rank.
        let mut result: BinaryHeap<RankNode> = BinaryHeap::with_capacity(ef + 1);
        let entry_ok = self.is_live(entry) && subset.map(|s| s.contains(&entry)).unwrap_or(true);
        if entry_ok {
            result.push(RankNode {
                rank: entry_rank,
                raw: entry_raw,
                idx: entry,
            });
        }

        while let Some(Reverse(closest)) = candidates.pop() {
            let worst_rank = result.peek().map_or(f32::MAX, |node| node.rank);
            if result.len() >= ef && closest.rank > worst_rank {
                break;
            }
            if closest.idx >= self.graph.len() {
                continue;
            }
            for &neighbor in &self.graph[closest.idx] {
                if !self.is_live(neighbor) || !visited.try_visit(neighbor) {
                    continue;
                }
                let n_raw = self.raw_distance(neighbor, query);
                let n_rank = if ascending { n_raw } else { -n_raw };
                let in_subset = subset.map(|s| s.contains(&neighbor)).unwrap_or(true);

                // Always expand (bridges help filtered connectivity).
                let worst = result.peek().map_or(f32::MAX, |node| node.rank);
                let explore = result.len() < ef || n_rank < worst || !in_subset;
                if explore {
                    candidates.push(Reverse(RankNode {
                        rank: n_rank,
                        raw: n_raw,
                        idx: neighbor,
                    }));
                }
                if in_subset && (result.len() < ef || n_rank < worst) {
                    result.push(RankNode {
                        rank: n_rank,
                        raw: n_raw,
                        idx: neighbor,
                    });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        let mut out: Vec<(f32, usize)> = result
            .into_iter()
            .map(|node| (node.raw, node.idx))
            .collect();
        if ascending {
            out.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            out.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }
        out
    }

    fn link_bidirectional(&mut self, src: usize, neighbors: &[usize], alpha: f32) {
        let degree = self.r.min(self.max_degree);
        for &dst in neighbors {
            if dst == src {
                continue;
            }
            if !self.graph[dst].contains(&src) {
                self.graph[dst].push(src);
            }
            if self.graph[dst].len() > degree {
                let cand: Vec<(f32, usize)> = self.graph[dst]
                    .iter()
                    .map(|&n| (self.rank_between(dst, n), n))
                    .collect();
                let pruned = self.robust_prune(dst, &cand, degree, alpha);
                self.graph[dst] = pruned;
            }
        }
    }

    /// One Vamana pass over all points with the given alpha.
    ///
    /// Uses batched parallel search/prune against a graph snapshot, then applies
    /// adjacency updates sequentially (same strategy as GPU/Parallel DiskANN).
    #[allow(dead_code)]
    fn vamana_pass(&mut self, alpha: f32, order: &[usize], _visited: &mut VisitedSet) {
        let entry = self.entry_point.unwrap_or(0);
        let degree = self.r.min(self.max_degree);
        let l = self.l;
        let ascending = self.config.distance_metric.is_ascending();
        let n = self.ids.len();
        // Larger batches → more parallelism, slightly staler snapshot. Keep
        // moderate size so reverse-link updates stay fresh enough for recall.
        let batch = 64.min(n.max(1));

        for chunk in order.chunks(batch) {
            let updates: Vec<(usize, Vec<usize>)> = chunk
                .par_iter()
                .map(|&p| {
                    BUILD_VISITED.with(|cell| {
                        let mut local_visited = cell.borrow_mut();
                        local_visited.ensure_capacity(n);
                        let query = {
                            let dim = self.config.dimension;
                            let start = p * dim;
                            unsafe {
                                std::slice::from_raw_parts(
                                    self.encoded_data.as_ptr().add(start),
                                    dim,
                                )
                            }
                        };
                        let mut cand =
                            self.search_graph(query, l, None, &mut local_visited, entry);
                        for &nb in &self.graph[p] {
                            cand.push((self.raw_distance(p, self.vec_at(nb)), nb));
                        }
                        let ranked: Vec<(f32, usize)> = cand
                            .into_iter()
                            .map(|(raw, idx)| {
                                let rank = if ascending { raw } else { -raw };
                                (rank, idx)
                            })
                            .collect();
                        let pruned = self.robust_prune(p, &ranked, degree, alpha);
                        (p, pruned)
                    })
                })
                .collect();

            for (p, pruned) in updates {
                self.graph[p] = pruned.clone();
                self.link_bidirectional(p, &pruned, alpha);
            }
        }
    }

    /// Full-precision batched Vamana.
    ///
    /// Each batch: parallel beam-search against a frozen adjacency snapshot,
    /// then apply forward edges. Reverse edges are merged and degree-pruned.
    /// When α>1, run a first pass at α=1 (connectivity) then α (long edges),
    /// matching the DiskANN/Vamana two-pass construction.
    fn build_vamana_parallel(&mut self) {
        let n = self.ids.len();
        let dim = self.config.dimension;
        let metric = self.config.distance_metric;
        let ascending = metric.is_ascending();
        let degree = self.r.min(self.max_degree);
        let l_build = self.l.min(L_BUILD_CAP).max(degree);
        let data = self.encoded_data.as_slice();

        // Optional reproducible build: LYNSE_DISKANN_SEED=<u64>.
        // Per-node init uses seed^mix(i) so parallel construction stays deterministic.
        let build_seed: Option<u64> = std::env::var("LYNSE_DISKANN_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok());
        let mut rng: StdRng = match build_seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let entry = self.choose_medoid(&mut rng);
        self.entry_point = Some(entry);

        // Random R-regular init (cheap) so early beam searches have connectivity.
        let mut graph: Vec<Vec<u32>> = vec![Vec::with_capacity(degree); n];
        if n > 1 {
            let d = degree.min(n - 1);
            let seed_base = build_seed.unwrap_or_else(|| rng.gen::<u64>());
            graph.par_iter_mut().enumerate().for_each(|(i, nbrs)| {
                let mut local = StdRng::seed_from_u64(
                    seed_base
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(i as u64),
                );
                let mut chosen = Vec::with_capacity(d);
                while chosen.len() < d {
                    let j = local.gen_range(0..n) as u32;
                    if j as usize != i && !chosen.contains(&j) {
                        chosen.push(j);
                    }
                }
                *nbrs = chosen;
            });
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        if let Some(pos) = order.iter().position(|&x| x == entry) {
            order.swap(0, pos);
        }

        let alphas: Vec<f32> = if self.alpha > 1.0 + f32::EPSILON {
            vec![1.0, self.alpha]
        } else {
            vec![self.alpha.max(1.0)]
        };

        let batch = 8192.min(n.max(1));
        for &alpha in &alphas {
            for chunk in order.chunks(batch) {
                let snapshot = &graph;
                let updates: Vec<(usize, Vec<u32>)> = chunk
                    .par_iter()
                    .map(|&p| {
                        BUILD_VISITED.with(|cell| {
                            let mut visited = cell.borrow_mut();
                            let query = diskann_vec(data, p as u32, dim);
                            let mut cand = search_beam_snapshot(
                                snapshot,
                                data,
                                dim,
                                metric,
                                ascending,
                                query,
                                entry as u32,
                                l_build,
                                &mut visited,
                            );
                            for &nb in &snapshot[p] {
                                cand.push((
                                    diskann_rank_between(
                                        data, p as u32, nb, dim, metric, ascending,
                                    ),
                                    nb,
                                ));
                            }
                            let pruned = robust_prune_raw(
                                data,
                                p as u32,
                                &cand,
                                degree,
                                alpha,
                                dim,
                                metric,
                                ascending,
                            );
                            (p, pruned)
                        })
                    })
                    .collect();

                for (p, pruned) in updates {
                    graph[p] = pruned;
                }
            }
        }

        // Materialize reverse edges, then degree-prune by distance (never drop
        // all reverse links just because the forward list is already full).
        let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (src, nbrs) in graph.iter().enumerate() {
            for &dst in nbrs {
                let d = dst as usize;
                if d != src {
                    rev[d].push(src as u32);
                }
            }
        }
        graph
            .par_iter_mut()
            .enumerate()
            .zip(rev.into_par_iter())
            .for_each(|((center, fwd), rev_nbrs)| {
                let mut merged = std::mem::take(fwd);
                for nb in rev_nbrs {
                    if !merged.contains(&nb) {
                        merged.push(nb);
                    }
                }
                *fwd = if merged.len() > degree {
                    prune_closest_raw(
                        data,
                        center as u32,
                        &merged,
                        degree,
                        dim,
                        metric,
                        ascending,
                    )
                } else {
                    merged
                };
            });

        self.graph = graph
            .into_iter()
            .map(|v| v.into_iter().map(|x| x as usize).collect())
            .collect();
    }

    fn brute_force_candidates(
        &self,
        encoded_query: &[f32],
        k: usize,
        subset: Option<&HashSet<usize>>,
    ) -> Result<(Vec<u64>, Vec<f32>)> {
        let candidate_idxs: Vec<usize> = match subset {
            Some(sub) => (0..self.ids.len())
                .filter(|i| self.is_live(*i) && sub.contains(i))
                .collect(),
            None => (0..self.ids.len()).filter(|&i| self.is_live(i)).collect(),
        };
        if candidate_idxs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        if subset.is_none() {
            let (top_idx, top_dist) = top_k_search(
                encoded_query,
                &self.encoded_data,
                self.config.dimension,
                k,
                self.config.distance_metric,
            );
            let result_ids: Vec<u64> = top_idx.iter().map(|&i| self.ids[i as usize]).collect();
            return Ok((result_ids, top_dist));
        }

        let mut scored: Vec<(f32, usize)> = candidate_idxs
            .into_iter()
            .map(|c| (self.raw_distance(c, encoded_query), c))
            .collect();
        if self.config.distance_metric.is_ascending() {
            scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        } else {
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        }
        scored.truncate(k);
        Ok((
            scored.iter().map(|(_, idx)| self.ids[*idx]).collect(),
            scored.iter().map(|(d, _)| *d).collect(),
        ))
    }
}

impl VectorIndex for DiskANNIndex {
    fn build(
        &mut self,
        vectors: &[f32],
        n_vectors: usize,
        dim: usize,
        ids: Option<&[u64]>,
    ) -> Result<()> {
        self.build_owned(
            vectors.to_vec(),
            n_vectors,
            dim,
            ids.map(|s| s.to_vec()),
        )
    }

    fn build_owned(
        &mut self,
        vectors: Vec<f32>,
        n_vectors: usize,
        dim: usize,
        ids: Option<Vec<u64>>,
    ) -> Result<()> {
        self.config.dimension = dim;
        self.ids = ids.unwrap_or_else(|| (0..n_vectors as u64).collect());

        // Layered mode: take ownership of the buffer (no second full copy).
        if self.wants_layered() {
            self.encoded_data = vectors;
            self.data.clear();
        } else if self.config.quantizer_type != QuantizerType::None {
            self.quantizer.fit(&vectors, n_vectors, dim)?;
            let bytes = self.quantizer.encode(&vectors, n_vectors, dim)?;
            self.encoded_data = self.quantizer.decode(&bytes, n_vectors, dim)?;
            self.data = vectors;
        } else {
            self.encoded_data = vectors.clone();
            self.data = vectors;
        }

        self.layered = false;
        self.pq = None;
        self.disk_graph = None;
        self.deleted_mask = vec![false; n_vectors];
        self.pending_dangling = 0;

        if n_vectors == 0 {
            self.graph.clear();
            self.entry_point = None;
            self.trained = true;
            return Ok(());
        }

        let t_graph = Instant::now();
        self.build_vamana_parallel();
        let _ = writeln!(
            stderr(),
            "diskann: full-precision graph build {:.1}s (n={}, R={}, L_build={})",
            t_graph.elapsed().as_secs_f64(),
            n_vectors,
            self.r.min(self.max_degree),
            self.l.min(L_BUILD_CAP).max(self.r.min(self.max_degree)),
        );

        self.trained = true;
        if self.wants_layered() && self.data_dir.is_some() {
            let t_flush = Instant::now();
            self.flush_layered()?;
            let _ = writeln!(
                stderr(),
                "diskann: layered flush (graph.bin+PQ) {:.1}s",
                t_flush.elapsed().as_secs_f64()
            );
        }
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
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let dim = self.config.dimension;
        let encoded_query = if !self.layered
            && self.config.quantizer_type != QuantizerType::None
            && !self.wants_layered()
        {
            let bytes = self.quantizer.encode(query, 1, dim)?;
            self.quantizer.decode(&bytes, 1, dim)?
        } else {
            query.to_vec()
        };

        let subset = params.subset_indices.as_ref().map(|ids| {
            let id_set: HashSet<u64> = ids.iter().cloned().collect();
            self.ids
                .iter()
                .enumerate()
                .filter(|(_, id)| id_set.contains(id))
                .map(|(i, _)| i)
                .collect::<HashSet<usize>>()
        });

        let ef = params
            .ef_search
            .unwrap_or(0)
            .max(params.nprobe)
            .max(self.l)
            .max(k.saturating_mul(if self.layered {
                DEFAULT_OVERSAMPLE
            } else if self.config.quantizer_type != QuantizerType::None {
                20
            } else {
                10
            }));

        let entry = self.live_entry().unwrap_or(0);
        let mut hits = SEARCH_VISITED.with(|cell| {
            let mut visited = cell.borrow_mut();
            if self.layered {
                self.search_graph_pq(&encoded_query, ef, subset.as_ref(), &mut visited, entry)
            } else {
                self.search_graph(&encoded_query, ef, subset.as_ref(), &mut visited, entry)
            }
        });

        if hits.is_empty() {
            if self.layered {
                return Ok((Vec::new(), Vec::new()));
            }
            return self.brute_force_candidates(&encoded_query, k, subset.as_ref());
        }

        // Non-layered SQ8/Binary: beam used reconstructed vectors — re-rank with
        // full-precision `self.data` when available.
        if !self.layered
            && self.config.quantizer_type != QuantizerType::None
            && self.data.len() == self.ids.len() * dim
        {
            let ascending = self.config.distance_metric.is_ascending();
            let mut rescored: Vec<(f32, usize)> = hits
                .iter()
                .map(|(_, idx)| {
                    let start = *idx * dim;
                    let dist = compute_distance_f32(
                        query,
                        &self.data[start..start + dim],
                        self.config.distance_metric,
                    );
                    (dist, *idx)
                })
                .collect();
            if ascending {
                rescored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            } else {
                rescored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            }
            rescored.truncate(k);
            return Ok((
                rescored.iter().map(|(_, idx)| self.ids[*idx]).collect(),
                rescored.iter().map(|(d, _)| *d).collect(),
            ));
        }

        hits.truncate(k);
        Ok((
            hits.iter().map(|(_, idx)| self.ids[*idx]).collect(),
            hits.iter().map(|(d, _)| *d).collect(),
        ))
    }

    fn delete(&mut self, ids: &[u64]) -> Result<()> {
        self.delete_with_vectors(ids, &[])
    }

    fn delete_with_vectors(&mut self, ids: &[u64], vectors: &[f32]) -> Result<()> {
        if !self.trained {
            return Err(LynseError::IndexNotBuilt);
        }
        if ids.is_empty() {
            return Ok(());
        }
        if self.deleted_mask.len() != self.ids.len() {
            self.deleted_mask.resize(self.ids.len(), false);
        }

        let dim = self.config.dimension;
        let id_set: HashSet<u64> = ids.iter().copied().collect();
        let mut targets: Vec<(usize, Option<Vec<f32>>)> = Vec::new();
        for (i, &id) in self.ids.iter().enumerate() {
            if id_set.contains(&id) && self.is_live(i) {
                let vec = if vectors.len() >= ids.len() * dim.max(1) && dim > 0 {
                    // Match by position in `ids` when caller supplied vectors.
                    ids.iter()
                        .position(|&x| x == id)
                        .map(|pos| vectors[pos * dim..(pos + 1) * dim].to_vec())
                } else if !self.layered && self.encoded_data.len() >= (i + 1) * dim {
                    let start = i * dim;
                    Some(self.encoded_data[start..start + dim].to_vec())
                } else if !self.layered && self.data.len() >= (i + 1) * dim {
                    let start = i * dim;
                    Some(self.data[start..start + dim].to_vec())
                } else {
                    self.pq.as_ref().and_then(|pq| pq.reconstruct(i))
                };
                targets.push((i, vec));
            }
        }
        if targets.is_empty() {
            return Ok(());
        }

        let degree = self.r.min(self.max_degree);
        let delete_l = IP_DELETE_L.max(self.l).max(degree);
        let cand_l = IP_CANDIDATE_L.min(delete_l).max(IP_SIGMA + 1);
        let sigma = IP_SIGMA;
        let alpha = self.alpha.max(1.2);

        for (p, query_opt) in targets {
            let query = match query_opt {
                Some(q) => q,
                None => {
                    // Fall back: mark deleted without edge repair.
                    self.deleted_mask[p] = true;
                    self.write_out_neighbors(p, &[])?;
                    self.pending_dangling += 1;
                    continue;
                }
            };

            let entry = self
                .live_entry()
                .filter(|&e| e != p)
                .unwrap_or_else(|| (0..self.ids.len()).find(|&i| i != p).unwrap_or(0));

            let (visited_nodes, candidates) = SEARCH_VISITED.with(|cell| {
                let mut visited = cell.borrow_mut();
                let hits = if self.layered {
                    self.search_graph_pq(&query, delete_l, None, &mut visited, entry)
                } else {
                    self.search_graph(&query, delete_l, None, &mut visited, entry)
                };
                // Approximate visited ≈ all expanded hits from beam.
                let visited_nodes: Vec<usize> = hits.iter().map(|(_, idx)| *idx).collect();
                let mut candidates: Vec<(f32, usize)> = hits
                    .into_iter()
                    .map(|(raw, idx)| {
                        let rank = if self.config.distance_metric.is_ascending() {
                            raw
                        } else {
                            -raw
                        };
                        (rank, idx)
                    })
                    .collect();
                candidates.truncate(cand_l);
                (visited_nodes, candidates)
            });

            // Approx in-neighbors: visited nodes that currently point to p.
            let mut approx_in: Vec<usize> = Vec::new();
            for &u in &visited_nodes {
                if u == p || !self.is_live(u) {
                    continue;
                }
                if self.out_neighbors(u).contains(&p) {
                    approx_in.push(u);
                }
            }

            let out_nbrs = self.out_neighbors(p);
            let mut touched: HashSet<usize> = HashSet::new();

            for &u in &approx_in {
                let replacements = self.closest_sigma_in(u, &candidates, sigma, p);
                let mut nbrs = self.out_neighbors(u);
                nbrs.retain(|&x| x != p);
                for &r in &replacements {
                    if !nbrs.contains(&r) {
                        nbrs.push(r);
                    }
                }
                if nbrs.len() > degree {
                    let cand: Vec<(f32, usize)> = nbrs
                        .iter()
                        .map(|&n| (self.rank_between_nodes(u, n), n))
                        .collect();
                    nbrs = self.robust_prune_nodes(u, &cand, degree, alpha);
                }
                self.write_out_neighbors(u, &nbrs)?;
                touched.insert(u);
            }

            for &v in &out_nbrs {
                if !self.is_live(v) {
                    continue;
                }
                let sources = self.closest_sigma_in(v, &candidates, sigma, p);
                for &s in &sources {
                    if s == v || !self.is_live(s) {
                        continue;
                    }
                    let mut nbrs = self.out_neighbors(s);
                    if !nbrs.contains(&v) {
                        nbrs.push(v);
                    }
                    if nbrs.len() > degree {
                        let cand: Vec<(f32, usize)> = nbrs
                            .iter()
                            .map(|&n| (self.rank_between_nodes(s, n), n))
                            .collect();
                        nbrs = self.robust_prune_nodes(s, &cand, degree, alpha);
                    }
                    self.write_out_neighbors(s, &nbrs)?;
                    touched.insert(s);
                }
            }

            self.deleted_mask[p] = true;
            self.write_out_neighbors(p, &[])?;
            self.pending_dangling += 1;
            let _ = touched;
        }

        if let Some(ep) = self.entry_point {
            if !self.is_live(ep) {
                self.entry_point = self.live_entry();
            }
        }
        self.maybe_consolidate_dangling()?;
        self.persist_layered_pq()?;
        Ok(())
    }

    fn insert(&mut self, vectors: &[f32], n_vectors: usize, dim: usize, ids: &[u64]) -> Result<()> {
        if !self.trained {
            return Err(LynseError::IndexNotBuilt);
        }
        if dim != self.config.dimension {
            return Err(LynseError::DimensionMismatch {
                expected: self.config.dimension,
                got: dim,
            });
        }
        if n_vectors == 0 {
            return Ok(());
        }
        if ids.len() != n_vectors || vectors.len() < n_vectors * dim {
            return Err(LynseError::InvalidArgument(
                "DiskANN insert: ids/vectors length mismatch".into(),
            ));
        }
        if self.deleted_mask.len() != self.ids.len() {
            self.deleted_mask.resize(self.ids.len(), false);
        }

        let degree = self.r.min(self.max_degree);
        let ascending = self.config.distance_metric.is_ascending();
        let alpha = self.alpha;

        for i in 0..n_vectors {
            let row = ids[i] as usize;
            let vec = &vectors[i * dim..(i + 1) * dim];

            // Grow to include this row (row indices must match VectorStore).
            while self.ids.len() <= row {
                let new_idx = self.ids.len();
                self.ids.push(new_idx as u64);
                self.deleted_mask.push(true); // hole until filled
                if self.layered {
                    if let Some(g) = self.disk_graph.as_mut() {
                        g.append_nodes(1)?;
                    }
                    // Pad PQ with a dummy encode of zeros if needed.
                    if let Some(pq) = self.pq.as_mut() {
                        let zeros = vec![0.0f32; dim];
                        pq.append_encoded(&zeros, 1, dim)
                            .map_err(|e| LynseError::Storage(e))?;
                    }
                } else {
                    self.graph.push(Vec::new());
                    self.encoded_data.extend(std::iter::repeat(0.0).take(dim));
                    if self.config.quantizer_type != QuantizerType::None {
                        self.data.extend(std::iter::repeat(0.0).take(dim));
                    } else {
                        self.data.extend(std::iter::repeat(0.0).take(dim));
                    }
                }
            }

            self.ids[row] = ids[i];
            self.deleted_mask[row] = false;

            if self.layered {
                if let Some(pq) = self.pq.as_mut() {
                    match pq.set_encoded_row(row, vec, dim) {
                        Ok(()) => {}
                        Err(_) => {
                            pq.append_encoded(vec, 1, dim)
                                .map_err(|e| LynseError::Storage(e))?;
                        }
                    }
                }
            } else {
                let start = row * dim;
                if self.config.quantizer_type != QuantizerType::None {
                    let bytes = self.quantizer.encode(vec, 1, dim)?;
                    let decoded = self.quantizer.decode(&bytes, 1, dim)?;
                    self.encoded_data[start..start + dim].copy_from_slice(&decoded);
                    if self.data.len() >= start + dim {
                        self.data[start..start + dim].copy_from_slice(vec);
                    }
                } else {
                    self.encoded_data[start..start + dim].copy_from_slice(vec);
                    if self.data.len() >= start + dim {
                        self.data[start..start + dim].copy_from_slice(vec);
                    }
                }
            }

            let entry = self
                .live_entry()
                .filter(|&e| e != row)
                .unwrap_or(row);

            let mut visited = VisitedSet::new(self.ids.len());
            let mut cand = if self.layered {
                self.search_graph_pq(vec, self.l, None, &mut visited, entry)
            } else {
                // Ensure encoded_data has the new vector for search.
                self.search_graph(
                    if self.config.quantizer_type != QuantizerType::None {
                        &self.encoded_data[row * dim..row * dim + dim]
                    } else {
                        vec
                    },
                    self.l,
                    None,
                    &mut visited,
                    entry,
                )
            };
            for &n in &self.out_neighbors(row) {
                cand.push((
                    if self.layered {
                        let lut = self
                            .pq
                            .as_ref()
                            .map(|pq| pq.adc_lut(vec, self.config.distance_metric));
                        if let (Some(pq), Some(lut)) = (self.pq.as_ref(), lut.as_ref()) {
                            let raw = pq.adc_raw(n, lut);
                            raw
                        } else {
                            0.0
                        }
                    } else {
                        self.raw_distance(row, self.vec_at(n))
                    },
                    n,
                ));
            }
            let ranked: Vec<(f32, usize)> = cand
                .into_iter()
                .map(|(raw, idx)| {
                    let rank = if ascending { raw } else { -raw };
                    (rank, idx)
                })
                .collect();
            let pruned = self.robust_prune_nodes(row, &ranked, degree, alpha);
            self.write_out_neighbors(row, &pruned)?;
            self.link_bidirectional_ip(row, &pruned, alpha)?;

            if self.entry_point.is_none() || !self.is_live(self.entry_point.unwrap_or(0)) {
                self.entry_point = Some(row);
            }
        }

        self.persist_layered_pq()?;
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
        let (data, encoded_data, graph) = if self.layered {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            (
                self.data.clone(),
                self.encoded_data.clone(),
                self.graph.clone(),
            )
        };
        let state = DiskANNState {
            data,
            encoded_data,
            ids: self.ids.clone(),
            deleted_mask: self.deleted_mask.clone(),
            graph,
            entry_point: self.entry_point,
            config: self.config.clone(),
            r: self.r,
            l: self.l,
            alpha: self.alpha,
            max_degree: self.max_degree,
            trained: self.trained,
            layered: self.layered,
            index_alias: self.index_alias.clone(),
            pending_dangling: self.pending_dangling,
        };
        bincode::serialize(&state).map_err(|e| LynseError::Serialization(e.to_string()))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let state: DiskANNState =
            bincode::deserialize(data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        self.data = state.data;
        self.encoded_data = state.encoded_data;
        self.ids = state.ids;
        self.deleted_mask = if state.deleted_mask.len() == self.ids.len() {
            state.deleted_mask
        } else {
            vec![false; self.ids.len()]
        };
        self.graph = state.graph;
        self.entry_point = state.entry_point;
        self.config = state.config;
        self.r = state.r;
        self.l = state.l;
        self.alpha = state.alpha;
        self.max_degree = state.max_degree;
        self.trained = state.trained;
        self.layered = state.layered;
        self.pending_dangling = state.pending_dangling;
        if !state.index_alias.is_empty() {
            self.index_alias = state.index_alias;
        }
        self.pq = None;
        self.disk_graph = None;
        // Sidecars opened via attach_data_dir after deserialize.
        Ok(())
    }

    fn attach_data_dir(&mut self, dir: &Path) -> Result<()> {
        self.data_dir = Some(dir.to_path_buf());
        if self.trained && self.layered {
            self.open_layered_sidecars()?;
        }
        Ok(())
    }

    fn uses_store_rescore(&self) -> bool {
        self.layered
    }

    fn search_candidates(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<Vec<u32>> {
        if !self.trained || self.ids.is_empty() {
            return Err(LynseError::IndexNotBuilt);
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let ef = params
            .ef_search
            .unwrap_or(0)
            .max(params.nprobe)
            .max(self.l)
            .max(k.saturating_mul(DEFAULT_OVERSAMPLE));
        let entry = self.live_entry().unwrap_or(0);
        let hits = SEARCH_VISITED.with(|cell| {
            let mut visited = cell.borrow_mut();
            if self.layered {
                self.search_graph_pq(query, ef, None, &mut visited, entry)
            } else {
                self.search_graph(query, ef, None, &mut visited, entry)
            }
        });
        let n_cand = ef.min(hits.len());
        Ok(hits
            .into_iter()
            .take(n_cand)
            .map(|(_, idx)| idx as u32)
            .collect())
    }

    fn name(&self) -> String {
        let metric = self.config.distance_metric.name();
        let quant = match self.config.quantizer_type {
            QuantizerType::None if self.layered => "-pq",
            QuantizerType::None => "",
            QuantizerType::Scalar => "-sq8",
            QuantizerType::Binary => "-binary",
            QuantizerType::Product => "-pq",
        };
        format!("diskann-{}{}", metric, quant)
    }
}

#[derive(Serialize, Deserialize)]
struct DiskANNState {
    data: Vec<f32>,
    encoded_data: Vec<f32>,
    ids: Vec<u64>,
    #[serde(default)]
    deleted_mask: Vec<bool>,
    graph: Vec<Vec<usize>>,
    entry_point: Option<usize>,
    config: IndexConfig,
    r: usize,
    l: usize,
    alpha: f32,
    max_degree: usize,
    trained: bool,
    #[serde(default)]
    layered: bool,
    #[serde(default)]
    index_alias: String,
    #[serde(default)]
    pending_dangling: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantizer::QuantizerType;
    use std::sync::Arc;

    #[test]
    fn diskann_ip_search_returns_max_inner_product() {
        let mut idx = DiskANNIndex::new(
            DistanceMetric::InnerProduct,
            QuantizerType::None,
            4,
            16,
            1.2,
            8,
        );
        let vectors = vec![
            1.0, 0.0, // id 0, IP=1.0
            0.9, 0.0, // id 1, IP=0.9
            0.1, 0.0, // id 2, IP=0.1
        ];
        let ids = [10u64, 20, 30];
        idx.build(&vectors, 3, 2, Some(&ids)).unwrap();

        let params = SearchParams {
            k: 1,
            nprobe: 1,
            ef_search: None,
            subset_indices: None,
        };
        let (result_ids, dists) = idx.search(&[1.0, 0.0], 1, &params).unwrap();
        assert_eq!(
            result_ids,
            vec![10],
            "ids={:?} dists={:?}",
            result_ids,
            dists
        );
        assert!(dists[0] > 0.95, "expected high IP, got {}", dists[0]);
    }

    #[test]
    fn diskann_filtered_empty_graph_does_not_leak_unfiltered_ids() {
        let mut idx =
            DiskANNIndex::new(DistanceMetric::L2Squared, QuantizerType::None, 2, 8, 1.2, 4);
        let vectors = vec![
            0.0, 0.0, // id 1
            0.1, 0.0, // id 2
            10.0, 10.0, // id 3
            10.1, 10.0, // id 4
        ];
        let ids = [1u64, 2, 3, 4];
        idx.build(&vectors, 4, 2, Some(&ids)).unwrap();

        let subset = Arc::new(vec![3u64, 4u64]);
        let params = SearchParams {
            k: 2,
            nprobe: 1,
            ef_search: None,
            subset_indices: Some(subset),
        };
        let (result_ids, _) = idx.search(&[0.0, 0.0], 2, &params).unwrap();
        assert!(!result_ids.is_empty(), "expected filtered fallback hits");
        assert!(
            result_ids.iter().all(|id| *id == 3 || *id == 4),
            "unfiltered ids leaked: {:?}",
            result_ids
        );
    }

    #[test]
    fn diskann_unfiltered_search_uses_graph_candidates() {
        let mut idx =
            DiskANNIndex::new(DistanceMetric::L2Squared, QuantizerType::None, 8, 32, 1.2, 16);
        let mut vectors = Vec::new();
        let mut ids = Vec::new();
        for i in 0..64u64 {
            vectors.extend_from_slice(&[i as f32 * 0.1, 0.0]);
            ids.push(i + 1);
        }
        idx.build(&vectors, 64, 2, Some(&ids)).unwrap();
        assert!(
            idx.graph.iter().any(|edges| !edges.is_empty()),
            "expected non-empty DiskANN graph after build"
        );
        assert!(idx.graph.iter().all(|e| e.len() <= idx.max_degree));

        let params = SearchParams {
            k: 5,
            nprobe: 32,
            ef_search: None,
            subset_indices: None,
        };
        let (result_ids, _) = idx.search(&[0.0, 0.0], 2, &params).unwrap();
        assert!(!result_ids.is_empty());
        assert_eq!(result_ids[0], 1);
    }

    #[test]
    fn diskann_ip_graph_maintains_degree_and_self_recall() {
        let mut idx = DiskANNIndex::new(
            DistanceMetric::InnerProduct,
            QuantizerType::None,
            16,
            64,
            1.2,
            16,
        );
        let n = 500usize;
        let dim = 32usize;
        let mut vectors = Vec::with_capacity(n * dim);
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = Vec::with_capacity(dim);
            // Distinct pseudo-random directions (avoid modular collisions).
            for j in 0..dim {
                let x = (((i * 131 + j * 17 + 1) % 997) as f32 / 997.0) - 0.5;
                v.push(x);
            }
            // Unit-normalize so IP(x,x)=1 is uniquely maximal for distinct vectors.
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in &mut v {
                *x /= norm;
            }
            vectors.extend_from_slice(&v);
            ids.push(i as u64);
        }
        idx.build(&vectors, n, dim, Some(&ids)).unwrap();
        let degrees: Vec<usize> = idx.graph.iter().map(|e| e.len()).collect();
        let avg = degrees.iter().sum::<usize>() as f32 / n as f32;
        let min_deg = *degrees.iter().min().unwrap();
        let max_deg = *degrees.iter().max().unwrap();
        eprintln!("IP degrees avg={avg:.2} min={min_deg} max={max_deg}");
        assert!(
            avg >= 4.0,
            "IP α-prune collapsed graph: avg degree {avg:.2} (min={min_deg} max={max_deg})"
        );

        let q = &vectors[100 * dim..(100 + 1) * dim];
        let params = SearchParams {
            k: 5,
            nprobe: 64,
            ef_search: Some(128),
            subset_indices: None,
        };
        let (result_ids, _) = idx.search(q, 5, &params).unwrap();
        eprintln!("IP self-search top: {:?}", result_ids);
        assert!(
            result_ids.contains(&100),
            "unit-normalized self vector missing from IP DiskANN results: {:?}",
            result_ids
        );
        // Brute-force top-1 must be reachable with high probability on this graph.
        let mut best_ip = f32::NEG_INFINITY;
        let mut best_id = 0u64;
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let ip: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            if ip > best_ip {
                best_ip = ip;
                best_id = i as u64;
            }
        }
        assert!(
            result_ids.contains(&best_id),
            "IP brute-force top-1 {best_id} missing from {:?} (best_ip={best_ip})",
            result_ids
        );
    }

    #[test]
    fn diskann_layered_flush_and_candidate_search() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskANNIndex::new(
            DistanceMetric::InnerProduct,
            QuantizerType::None,
            16,
            64,
            1.2,
            16,
        );
        idx.attach_data_dir(tmp.path()).unwrap();
        let n = 200usize;
        let dim = 32usize;
        let mut vectors = Vec::with_capacity(n * dim);
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            for j in 0..dim {
                let x = (((i * 131 + j * 17 + 1) % 997) as f32 / 997.0) - 0.5;
                vectors.push(x);
            }
            ids.push(i as u64);
        }
        idx.build(&vectors, n, dim, Some(&ids)).unwrap();
        assert!(idx.layered, "expected layered flush");
        assert!(tmp.path().join("diskann/graph.bin").exists());
        assert!(tmp.path().join("diskann/pq.bin").exists());
        assert!(idx.uses_store_rescore());
        let params = SearchParams {
            k: 10,
            nprobe: 64,
            ef_search: Some(128),
            subset_indices: None,
        };
        let cands = idx.search_candidates(&vectors[..dim], 10, &params).unwrap();
        assert!(!cands.is_empty());
        assert!(cands.contains(&0), "self row should be a candidate: {:?}", cands);
        let bytes = idx.serialize().unwrap();
        assert!(
            bytes.len() < n * dim,
            "layered serialize should not embed full f32 ({} bytes)",
            bytes.len()
        );
        let mut loaded = DiskANNIndex::new(
            DistanceMetric::InnerProduct,
            QuantizerType::None,
            16,
            64,
            1.2,
            16,
        );
        loaded.deserialize(&bytes).unwrap();
        loaded.attach_data_dir(tmp.path()).unwrap();
        assert!(loaded.layered);
        let cands2 = loaded
            .search_candidates(&vectors[..dim], 10, &params)
            .unwrap();
        assert!(!cands2.is_empty());
    }

    #[test]
    fn diskann_robust_prune_uses_alpha() {
        let mut idx =
            DiskANNIndex::new(DistanceMetric::L2Squared, QuantizerType::None, 2, 8, 1.2, 2);
        // p at origin; a close; b almost behind a; c far orthogonal
        let vectors = vec![
            0.0, 0.0, // p=0
            1.0, 0.0, // a=1
            1.1, 0.05, // b=2 (occluded by a for alpha~1)
            0.0, 2.0, // c=3
        ];
        idx.config.dimension = 2;
        idx.encoded_data = vectors.clone();
        idx.data = vectors;
        idx.ids = vec![0, 1, 2, 3];
        let cand = vec![
            (1.0, 1usize),
            (idx.rank_between(0, 2), 2usize),
            (2.0, 3usize),
        ];
        let pruned = idx.robust_prune(0, &cand, 2, 1.2);
        assert!(pruned.contains(&1), "nearest neighbor must be kept");
        assert!(pruned.contains(&3), "orthogonal neighbor should survive α-prune");
        assert_eq!(pruned.len(), 2);
    }

    #[test]
    fn diskann_layered_ip_insert_delete_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut idx = DiskANNIndex::new(
            DistanceMetric::L2Squared,
            QuantizerType::None,
            8,
            32,
            1.2,
            8,
        );
        idx.attach_data_dir(tmp.path()).unwrap();
        let n = 80usize;
        let dim = 16usize;
        let mut vectors = Vec::with_capacity(n * dim);
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            for j in 0..dim {
                vectors.push(((i * 13 + j * 7) % 97) as f32);
            }
            ids.push(i as u64);
        }
        idx.build(&vectors, n, dim, Some(&ids)).unwrap();
        assert!(idx.layered);

        // Delete a mid point with its vector (IP-DiskANN Alg 5).
        let del = 17usize;
        let del_vec = vectors[del * dim..(del + 1) * dim].to_vec();
        idx.delete_with_vectors(&[del as u64], &del_vec).unwrap();
        assert!(!idx.is_live(del));

        let params = SearchParams {
            k: 5,
            nprobe: 32,
            ef_search: Some(64),
            subset_indices: None,
        };
        let cands = idx
            .search_candidates(&del_vec, 5, &params)
            .unwrap();
        assert!(
            !cands.contains(&(del as u32)),
            "deleted row must not appear in candidates: {:?}",
            cands
        );

        // Append a new vector at the next row index.
        let new_row = n as u64;
        let mut new_vec = vec![0.0f32; dim];
        new_vec[0] = 42.0;
        idx.insert(&new_vec, 1, dim, &[new_row]).unwrap();
        assert!(idx.is_live(n));
        assert_eq!(idx.len(), n + 1);

        let cands2 = idx
            .search_candidates(&new_vec, 5, &params)
            .unwrap();
        assert!(
            cands2.contains(&(n as u32)),
            "newly inserted row should be a candidate: {:?}",
            cands2
        );

        // Revive deleted slot via insert at same row (upsert pattern).
        let mut revived = del_vec.clone();
        revived[0] += 1.0;
        idx.insert(&revived, 1, dim, &[del as u64]).unwrap();
        assert!(idx.is_live(del));
    }
}
