//! BitSet implementation for time-travel queries and deletion tracking.
//!
//! Port of Python `lynse/core_components/bitset.py` to Rust.
//! Features:
//! - Compact bit storage using u64 words
//! - Bitwise operations (AND, OR, XOR, NOT)
//! - Serialization/deserialization for persistence
//! - Iterator over set bits
//! - Auto-resize on set_bit beyond current size

use crate::error::{LynseError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A compact bit set backed by Vec&lt;u64&gt; words.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitSet {
    /// Storage words (each u64 holds 64 bits)
    words: Vec<u64>,
    /// Actual number of bits (may be less than words.len() * 64)
    len: usize,
}

impl BitSet {
    /// Create a new BitSet with `size` bits, all initialized to `fill` (0 or 1).
    pub fn new(size: usize, fill: bool) -> Self {
        let n_words = (size + 63) / 64;
        let fill_word = if fill { !0u64 } else { 0u64 };
        let mut words = vec![fill_word; n_words];

        // Mask off trailing bits in last word if fill=true
        if fill && size > 0 {
            let remainder = size % 64;
            if remainder != 0 {
                if let Some(last) = words.last_mut() {
                    *last = (1u64 << remainder) - 1;
                }
            }
        }

        Self { words, len: size }
    }

    /// Create an empty BitSet.
    pub fn empty() -> Self {
        Self {
            words: Vec::new(),
            len: 0,
        }
    }

    /// Number of bits in the BitSet.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the BitSet has no bits.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resize the BitSet to `new_size` bits. New bits are set to `fill`.
    pub fn resize(&mut self, new_size: usize, fill: bool) {
        if new_size <= self.len {
            let n_words = (new_size + 63) / 64;
            self.words.truncate(n_words);
            if new_size > 0 {
                let remainder = new_size % 64;
                if remainder != 0 {
                    if let Some(last) = self.words.last_mut() {
                        *last &= (1u64 << remainder) - 1;
                    }
                }
            }
        } else {
            let new_n_words = (new_size + 63) / 64;
            let fill_word = if fill { !0u64 } else { 0u64 };

            if fill && self.len > 0 {
                let old_remainder = self.len % 64;
                if old_remainder != 0 {
                    if let Some(last) = self.words.last_mut() {
                        *last |= !((1u64 << old_remainder) - 1);
                    }
                }
            }

            self.words.resize(new_n_words, fill_word);

            if fill {
                let new_remainder = new_size % 64;
                if new_remainder != 0 {
                    if let Some(last) = self.words.last_mut() {
                        *last &= (1u64 << new_remainder) - 1;
                    }
                }
            }
        }
        self.len = new_size;
    }

    /// Set the bit at `index` to 1. Auto-resizes if needed.
    #[inline]
    pub fn set_bit(&mut self, index: usize) {
        if index >= self.len {
            self.resize(index + 1, false);
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        self.words[word_idx] |= 1u64 << bit_idx;
    }

    /// Clear the bit at `index` to 0.
    #[inline]
    pub fn clear_bit(&mut self, index: usize) {
        if index >= self.len {
            return;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        self.words[word_idx] &= !(1u64 << bit_idx);
    }

    /// Get the value of the bit at `index`.
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }

    /// Toggle the bit at `index`. Auto-resizes if needed.
    #[inline]
    pub fn toggle_bit(&mut self, index: usize) {
        if index >= self.len {
            self.resize(index + 1, false);
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        self.words[word_idx] ^= 1u64 << bit_idx;
    }

    /// Count the number of set bits (popcount).
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Set all bits to 0.
    pub fn clear_all(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// Set all bits to 1.
    pub fn set_all(&mut self) {
        for w in &mut self.words {
            *w = !0u64;
        }
        // Mask trailing bits
        if self.len > 0 {
            let remainder = self.len % 64;
            if remainder != 0 {
                if let Some(last) = self.words.last_mut() {
                    *last = (1u64 << remainder) - 1;
                }
            }
        }
    }

    /// Iterate over indices of set bits.
    pub fn iter_set_bits(&self) -> SetBitIter<'_> {
        SetBitIter {
            bitset: self,
            word_idx: 0,
            current_word: if self.words.is_empty() {
                0
            } else {
                self.words[0]
            },
            base: 0,
        }
    }

    /// Collect all set bit indices into a Vec.
    pub fn to_vec(&self) -> Vec<u64> {
        self.iter_set_bits().map(|i| i as u64).collect()
    }

    /// Bitwise AND.
    pub fn and(&self, other: &BitSet) -> BitSet {
        let min_len = self.len.min(other.len);
        let n_words = (min_len + 63) / 64;
        let mut words = Vec::with_capacity(n_words);
        for i in 0..n_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words.push(a & b);
        }
        BitSet {
            words,
            len: min_len,
        }
    }

    /// Bitwise OR.
    pub fn or(&self, other: &BitSet) -> BitSet {
        let max_len = self.len.max(other.len);
        let n_words = (max_len + 63) / 64;
        let mut words = Vec::with_capacity(n_words);
        for i in 0..n_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words.push(a | b);
        }
        BitSet {
            words,
            len: max_len,
        }
    }

    /// Bitwise XOR.
    pub fn xor(&self, other: &BitSet) -> BitSet {
        let max_len = self.len.max(other.len);
        let n_words = (max_len + 63) / 64;
        let mut words = Vec::with_capacity(n_words);
        for i in 0..n_words {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            words.push(a ^ b);
        }
        BitSet {
            words,
            len: max_len,
        }
    }

    /// Bitwise NOT.
    pub fn not(&self) -> BitSet {
        let mut words: Vec<u64> = self.words.iter().map(|w| !w).collect();
        // Mask trailing bits
        if self.len > 0 {
            let remainder = self.len % 64;
            if remainder != 0 {
                if let Some(last) = words.last_mut() {
                    *last &= (1u64 << remainder) - 1;
                }
            }
        }
        BitSet {
            words,
            len: self.len,
        }
    }

    /// Save the BitSet to a file.
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let data =
            bincode::serialize(self).map_err(|e| LynseError::Serialization(e.to_string()))?;
        std::fs::write(path, &data)?;
        Ok(())
    }

    /// Load a BitSet from a file.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let bitset: Self =
            bincode::deserialize(&data).map_err(|e| LynseError::Serialization(e.to_string()))?;
        Ok(bitset)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // len as u64 LE + raw word bytes
        let mut buf = Vec::with_capacity(8 + self.words.len() * 8);
        buf.extend_from_slice(&(self.len as u64).to_le_bytes());
        for w in &self.words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(LynseError::Serialization("BitSet data too short".into()));
        }
        let len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let n_words = (len + 63) / 64;
        let expected_size = 8 + n_words * 8;
        if data.len() < expected_size {
            return Err(LynseError::Serialization("BitSet data truncated".into()));
        }
        let mut words = Vec::with_capacity(n_words);
        for i in 0..n_words {
            let offset = 8 + i * 8;
            let w = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            words.push(w);
        }
        Ok(BitSet { words, len })
    }
}

impl PartialEq for BitSet {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.words == other.words
    }
}
impl Eq for BitSet {}

/// Iterator over set bit indices.
pub struct SetBitIter<'a> {
    bitset: &'a BitSet,
    word_idx: usize,
    current_word: u64,
    base: usize,
}

impl<'a> Iterator for SetBitIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.current_word != 0 {
                let tz = self.current_word.trailing_zeros() as usize;
                self.current_word &= self.current_word - 1; // clear lowest set bit
                let idx = self.base + tz;
                if idx < self.bitset.len {
                    return Some(idx);
                }
                return None;
            }
            self.word_idx += 1;
            if self.word_idx >= self.bitset.words.len() {
                return None;
            }
            self.current_word = self.bitset.words[self.word_idx];
            self.base = self.word_idx * 64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut bs = BitSet::new(100, false);
        assert_eq!(bs.len(), 100);
        assert_eq!(bs.count(), 0);

        bs.set_bit(0);
        bs.set_bit(50);
        bs.set_bit(99);
        assert!(bs.get_bit(0));
        assert!(bs.get_bit(50));
        assert!(bs.get_bit(99));
        assert!(!bs.get_bit(1));
        assert_eq!(bs.count(), 3);

        bs.clear_bit(50);
        assert!(!bs.get_bit(50));
        assert_eq!(bs.count(), 2);
    }

    #[test]
    fn test_auto_resize() {
        let mut bs = BitSet::new(10, false);
        bs.set_bit(200);
        assert!(bs.get_bit(200));
        assert_eq!(bs.len(), 201);
    }

    #[test]
    fn test_bitwise_ops() {
        let mut a = BitSet::new(64, false);
        let mut b = BitSet::new(64, false);
        a.set_bit(0);
        a.set_bit(1);
        b.set_bit(1);
        b.set_bit(2);

        let and_result = a.and(&b);
        assert!(and_result.get_bit(1));
        assert!(!and_result.get_bit(0));
        assert_eq!(and_result.count(), 1);

        let or_result = a.or(&b);
        assert!(or_result.get_bit(0));
        assert!(or_result.get_bit(1));
        assert!(or_result.get_bit(2));
        assert_eq!(or_result.count(), 3);
    }

    #[test]
    fn test_iter_set_bits() {
        let mut bs = BitSet::new(200, false);
        bs.set_bit(5);
        bs.set_bit(100);
        bs.set_bit(199);
        let bits: Vec<usize> = bs.iter_set_bits().collect();
        assert_eq!(bits, vec![5, 100, 199]);
    }

    #[test]
    fn test_not() {
        let mut bs = BitSet::new(8, false);
        bs.set_bit(0);
        bs.set_bit(2);
        let inv = bs.not();
        assert!(!inv.get_bit(0));
        assert!(inv.get_bit(1));
        assert!(!inv.get_bit(2));
        assert!(inv.get_bit(3));
        assert_eq!(inv.count(), 6);
    }

    #[test]
    fn test_serialization() {
        let mut bs = BitSet::new(150, false);
        bs.set_bit(0);
        bs.set_bit(64);
        bs.set_bit(149);

        let bytes = bs.to_bytes();
        let bs2 = BitSet::from_bytes(&bytes).unwrap();
        assert_eq!(bs, bs2);
    }

    #[test]
    fn test_fill_true() {
        let bs = BitSet::new(10, true);
        assert_eq!(bs.count(), 10);
        for i in 0..10 {
            assert!(bs.get_bit(i));
        }
        assert!(!bs.get_bit(10)); // beyond len
    }
}
