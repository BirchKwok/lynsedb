//! DiskANN adjacency graph stored on SSD with a small RAM page cache.
//!
//! Layout: fixed out-degree `R` per node. Node `i` occupies
//! `i * R * 4` bytes; empty slots are `u32::MAX`.
//!
//! The store is mutable so IP-DiskANN can update / append adjacency rows
//! without a full graph rebuild.

use crate::error::{LynseError, Result};
use memmap2::{MmapMut, MmapOptions};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Sentinel for unused neighbor slots.
pub const EMPTY_NEIGHBOR: u32 = u32::MAX;

/// Default neighbor-row cache capacity (nodes).
const DEFAULT_CACHE_NODES: usize = 65_536;

/// Fixed-degree adjacency list backed by a mutable memory-mapped file.
pub struct DiskGraphStore {
    path: PathBuf,
    mmap: MmapMut,
    n: usize,
    degree: usize,
    cache: Mutex<NeighborCache>,
}

struct NeighborCache {
    map: HashMap<usize, Vec<u32>>,
    order: VecDeque<usize>,
    capacity: usize,
}

impl NeighborCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity.min(1024)),
            order: VecDeque::with_capacity(capacity.min(1024)),
            capacity: capacity.max(1),
        }
    }

    fn get(&mut self, idx: usize) -> Option<&Vec<u32>> {
        if self.map.contains_key(&idx) {
            if let Some(pos) = self.order.iter().position(|&x| x == idx) {
                self.order.remove(pos);
                self.order.push_back(idx);
            }
            self.map.get(&idx)
        } else {
            None
        }
    }

    fn insert(&mut self, idx: usize, neighbors: Vec<u32>) {
        if self.map.contains_key(&idx) {
            self.map.insert(idx, neighbors);
            return;
        }
        while self.map.len() >= self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            } else {
                break;
            }
        }
        self.map.insert(idx, neighbors);
        self.order.push_back(idx);
    }

    fn invalidate(&mut self, idx: usize) {
        self.map.remove(&idx);
        if let Some(pos) = self.order.iter().position(|&x| x == idx) {
            self.order.remove(pos);
        }
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

impl DiskGraphStore {
    /// Write adjacency lists to `path` and open as mmap-backed store.
    pub fn write_from_adj(path: &Path, graph: &[Vec<usize>], degree: usize) -> Result<Self> {
        let degree = degree.max(1);
        let n = graph.len();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| LynseError::Storage(e.to_string()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        let mut file = BufWriter::with_capacity(8 * 1024 * 1024, file);

        let mut row_bytes = vec![0u8; degree * 4];
        for neighbors in graph {
            for slot in 0..degree {
                let v = neighbors
                    .get(slot)
                    .copied()
                    .map(|nb| nb as u32)
                    .unwrap_or(EMPTY_NEIGHBOR);
                row_bytes[slot * 4..slot * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            file.write_all(&row_bytes)
                .map_err(|e| LynseError::Storage(e.to_string()))?;
        }
        file.flush()
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        drop(file);
        Self::open(path, n, degree)
    }

    /// Open an existing fixed-degree graph file for read/write.
    pub fn open(path: &Path, n: usize, degree: usize) -> Result<Self> {
        let degree = degree.max(1);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        let expected = n
            .checked_mul(degree)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| LynseError::Storage("diskann graph size overflow".into()))?;
        let meta = file
            .metadata()
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        if meta.len() as usize != expected {
            return Err(LynseError::Storage(format!(
                "diskann graph size mismatch: file={} expected={}",
                meta.len(),
                expected
            )));
        }
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| LynseError::Storage(e.to_string()))?
        };
        #[cfg(unix)]
        {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            let len = mmap.len();
            unsafe {
                libc::madvise(ptr, len, libc::MADV_RANDOM);
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            n,
            degree,
            cache: Mutex::new(NeighborCache::new(DEFAULT_CACHE_NODES.min(n.max(1)))),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return valid out-neighbors of `idx` (EMPTY slots removed). Cached.
    pub fn neighbors(&self, idx: usize) -> Vec<u32> {
        if idx >= self.n {
            return Vec::new();
        }
        {
            let mut cache = self.cache.lock();
            if let Some(hit) = cache.get(idx) {
                return hit.clone();
            }
        }
        let row = self.read_row_raw(idx);
        let valid: Vec<u32> = row
            .into_iter()
            .filter(|&v| v != EMPTY_NEIGHBOR && (v as usize) < self.n)
            .collect();
        {
            let mut cache = self.cache.lock();
            cache.insert(idx, valid.clone());
        }
        valid
    }

    fn read_row_raw(&self, idx: usize) -> Vec<u32> {
        let start = idx * self.degree * 4;
        let bytes = &self.mmap[start..start + self.degree * 4];
        let mut out = Vec::with_capacity(self.degree);
        for chunk in bytes.chunks_exact(4) {
            out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        out
    }

    /// Overwrite the fixed-degree neighbor row for `idx`.
    pub fn set_neighbors(&mut self, idx: usize, neighbors: &[u32]) -> Result<()> {
        if idx >= self.n {
            return Err(LynseError::InvalidArgument(format!(
                "diskann graph set_neighbors: idx {} out of range (n={})",
                idx, self.n
            )));
        }
        let start = idx * self.degree * 4;
        for slot in 0..self.degree {
            let v = neighbors.get(slot).copied().unwrap_or(EMPTY_NEIGHBOR);
            let off = start + slot * 4;
            self.mmap[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.cache.lock().invalidate(idx);
        Ok(())
    }

    /// Append `count` empty nodes (all EMPTY neighbors) and remap.
    pub fn append_nodes(&mut self, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let new_n = self
            .n
            .checked_add(count)
            .ok_or_else(|| LynseError::Storage("diskann graph append overflow".into()))?;
        let new_len = new_n
            .checked_mul(self.degree)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| LynseError::Storage("diskann graph size overflow".into()))?;

        // Flush + drop mmap before resizing the underlying file.
        self.mmap
            .flush()
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        // Replace with empty mut map so the file handle is released.
        let path = self.path.clone();
        let degree = self.degree;
        let old_n = self.n;
        drop(std::mem::replace(
            &mut self.mmap,
            MmapMut::map_anon(1).map_err(|e| LynseError::Storage(e.to_string()))?,
        ));

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        file.set_len(new_len as u64)
            .map_err(|e| LynseError::Storage(e.to_string()))?;

        // Fill new rows with EMPTY_NEIGHBOR.
        let row_bytes = self.degree * 4;
        let mut empty_row = vec![0u8; row_bytes];
        for slot in 0..self.degree {
            empty_row[slot * 4..slot * 4 + 4].copy_from_slice(&EMPTY_NEIGHBOR.to_le_bytes());
        }
        use std::io::{Seek, SeekFrom};
        let mut file = file;
        file.seek(SeekFrom::Start((old_n * row_bytes) as u64))
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        for _ in 0..count {
            file.write_all(&empty_row)
                .map_err(|e| LynseError::Storage(e.to_string()))?;
        }
        file.flush()
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        drop(file);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| LynseError::Storage(e.to_string()))?;
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| LynseError::Storage(e.to_string()))?
        };
        #[cfg(unix)]
        {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            let len = mmap.len();
            unsafe {
                libc::madvise(ptr, len, libc::MADV_RANDOM);
            }
        }
        self.mmap = mmap;
        self.n = new_n;
        self.cache.lock().clear();
        let _ = degree;
        Ok(())
    }

    /// Persist dirty mmap pages to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap
            .flush()
            .map_err(|e| LynseError::Storage(e.to_string()))
    }

    /// Advise the OS that these nodes' adjacency rows will be needed soon.
    pub fn prefetch(&self, ids: &[usize]) {
        #[cfg(unix)]
        {
            for &idx in ids {
                if idx >= self.n {
                    continue;
                }
                let start = idx * self.degree * 4;
                let len = self.degree * 4;
                if start + len > self.mmap.len() {
                    continue;
                }
                let ptr = unsafe { self.mmap.as_ptr().add(start) } as *mut libc::c_void;
                unsafe {
                    libc::madvise(ptr, len, libc::MADV_WILLNEED);
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = ids;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_open_neighbors_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("graph.bin");
        let graph = vec![vec![1usize, 2], vec![0, 2, 3], vec![0], vec![1]];
        let store = DiskGraphStore::write_from_adj(&path, &graph, 4).unwrap();
        assert_eq!(store.neighbors(0), vec![1, 2]);
        assert_eq!(store.neighbors(1), vec![0, 2, 3]);
        assert_eq!(store.neighbors(2), vec![0]);
        let reopened = DiskGraphStore::open(&path, 4, 4).unwrap();
        assert_eq!(reopened.neighbors(3), vec![1]);
    }

    #[test]
    fn set_neighbors_and_append_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("graph.bin");
        let graph = vec![vec![1usize], vec![0usize]];
        let mut store = DiskGraphStore::write_from_adj(&path, &graph, 4).unwrap();
        store.set_neighbors(0, &[1, 1, 1]).unwrap();
        assert_eq!(store.neighbors(0), vec![1, 1, 1]);

        store.append_nodes(2).unwrap();
        assert_eq!(store.len(), 4);
        assert!(store.neighbors(2).is_empty());
        store.set_neighbors(2, &[0, 3]).unwrap();
        assert_eq!(store.neighbors(2), vec![0, 3]);

        store.flush().unwrap();
        let reopened = DiskGraphStore::open(&path, 4, 4).unwrap();
        assert_eq!(reopened.neighbors(0), vec![1, 1, 1]);
        assert_eq!(reopened.neighbors(2), vec![0, 3]);
    }
}
