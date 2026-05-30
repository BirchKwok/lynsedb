//! Bloom Filter for fast string membership testing
//!
//! Used to skip row groups that definitely don't contain a value.
//! This significantly speeds up string equality filters on large tables.

use bloomfilter::Bloom;
use std::io::{self, Read, Write};

/// Row group bloom filter for string columns
/// Each row group (default 64K rows) has its own bloom filter
pub struct RowGroupBloomFilter {
    /// Bloom filter for this row group
    filter: Bloom<[u8]>,
    /// Number of items in this row group
    item_count: usize,
    /// Row group start index
    pub start_row: usize,
    /// Row group end index (exclusive)
    pub end_row: usize,
}

impl RowGroupBloomFilter {
    /// Create a new bloom filter for a row group
    /// false_positive_rate: typically 0.01 (1%)
    pub fn new(
        expected_items: usize,
        false_positive_rate: f64,
        start_row: usize,
        end_row: usize,
    ) -> Self {
        let filter = Bloom::new_for_fp_rate(expected_items.max(1), false_positive_rate);
        Self {
            filter,
            item_count: 0,
            start_row,
            end_row,
        }
    }

    /// Add a string value to the bloom filter
    #[inline]
    pub fn insert(&mut self, value: &[u8]) {
        self.filter.set(value);
        self.item_count += 1;
    }

    /// Check if a value might be in this row group
    /// Returns false = definitely not present (can skip this row group)
    /// Returns true = might be present (need to scan)
    #[inline]
    pub fn might_contain(&self, value: &[u8]) -> bool {
        self.filter.check(value)
    }

    /// Serialize bloom filter to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let bitmap = self.filter.bitmap();
        let num_bits = self.filter.number_of_bits();
        let num_hashes = self.filter.number_of_hash_functions();
        let sip_keys = self.filter.sip_keys();

        let mut bytes = Vec::with_capacity(72 + bitmap.len());

        // Header: item_count(8) + start_row(8) + end_row(8) + num_bits(8) + num_hashes(4) + bitmap_len(4) + sip_keys(32)
        bytes.extend_from_slice(&(self.item_count as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.start_row as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.end_row as u64).to_le_bytes());
        bytes.extend_from_slice(&(num_bits as u64).to_le_bytes());
        bytes.extend_from_slice(&(num_hashes as u32).to_le_bytes());
        bytes.extend_from_slice(&(bitmap.len() as u32).to_le_bytes());
        // SIP keys: 4 x u64 = 32 bytes
        bytes.extend_from_slice(&sip_keys[0].0.to_le_bytes());
        bytes.extend_from_slice(&sip_keys[0].1.to_le_bytes());
        bytes.extend_from_slice(&sip_keys[1].0.to_le_bytes());
        bytes.extend_from_slice(&sip_keys[1].1.to_le_bytes());
        bytes.extend_from_slice(&bitmap);

        bytes
    }

    /// Deserialize bloom filter from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < 72 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter data too short",
            ));
        }

        let item_count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let start_row = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let end_row = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;
        let num_bits = u64::from_le_bytes(data[24..32].try_into().unwrap()) as usize;
        let num_hashes = u32::from_le_bytes(data[32..36].try_into().unwrap()) as u32;
        let bitmap_len = u32::from_le_bytes(data[36..40].try_into().unwrap()) as usize;
        // Read SIP keys
        let sk0_0 = u64::from_le_bytes(data[40..48].try_into().unwrap());
        let sk0_1 = u64::from_le_bytes(data[48..56].try_into().unwrap());
        let sk1_0 = u64::from_le_bytes(data[56..64].try_into().unwrap());
        let sk1_1 = u64::from_le_bytes(data[64..72].try_into().unwrap());
        let sip_keys = [(sk0_0, sk0_1), (sk1_0, sk1_1)];

        if data.len() < 72 + bitmap_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom filter bitmap incomplete",
            ));
        }

        let bitmap = data[72..72 + bitmap_len].to_vec();
        let filter = Bloom::from_existing(&bitmap, num_bits as u64, num_hashes, sip_keys);

        Ok(Self {
            filter,
            item_count,
            start_row,
            end_row,
        })
    }
}

/// Column bloom filter index - collection of bloom filters for all row groups
pub struct ColumnBloomIndex {
    /// Bloom filters for each row group
    pub filters: Vec<RowGroupBloomFilter>,
    /// Column name
    pub column_name: String,
    /// Default row group size
    pub row_group_size: usize,
}

impl ColumnBloomIndex {
    /// Create a new empty bloom index
    pub fn new(column_name: &str, row_group_size: usize) -> Self {
        Self {
            filters: Vec::new(),
            column_name: column_name.to_string(),
            row_group_size,
        }
    }

    /// Build bloom index from string column data
    pub fn build_from_strings(
        column_name: &str,
        offsets: &[u32],
        data: &[u8],
        row_group_size: usize,
        false_positive_rate: f64,
    ) -> Self {
        let row_count = offsets.len().saturating_sub(1);
        let num_groups = (row_count + row_group_size - 1) / row_group_size;

        let mut filters = Vec::with_capacity(num_groups);

        for group_idx in 0..num_groups {
            let start_row = group_idx * row_group_size;
            let end_row = ((group_idx + 1) * row_group_size).min(row_count);
            let group_size = end_row - start_row;

            let mut filter =
                RowGroupBloomFilter::new(group_size, false_positive_rate, start_row, end_row);

            // Add all strings in this row group
            for i in start_row..end_row {
                let str_start = offsets[i] as usize;
                let str_end = offsets[i + 1] as usize;
                if str_end <= data.len() {
                    filter.insert(&data[str_start..str_end]);
                }
            }

            filters.push(filter);
        }

        Self {
            filters,
            column_name: column_name.to_string(),
            row_group_size,
        }
    }

    /// Build bloom index from dictionary-encoded string column
    pub fn build_from_dict(
        column_name: &str,
        indices: &[u32],
        dict_offsets: &[u32],
        dict_data: &[u8],
        row_group_size: usize,
        false_positive_rate: f64,
    ) -> Self {
        let row_count = indices.len();
        let num_groups = (row_count + row_group_size - 1) / row_group_size;
        let dict_count = dict_offsets.len().saturating_sub(1);

        // Pre-build dictionary strings for fast lookup
        let dict_strings: Vec<&[u8]> = (0..dict_count)
            .map(|i| {
                let start = dict_offsets[i] as usize;
                let end = if i + 1 < dict_offsets.len() {
                    dict_offsets[i + 1] as usize
                } else {
                    dict_data.len()
                };
                &dict_data[start..end]
            })
            .collect();

        let mut filters = Vec::with_capacity(num_groups);

        for group_idx in 0..num_groups {
            let start_row = group_idx * row_group_size;
            let end_row = ((group_idx + 1) * row_group_size).min(row_count);

            // For dictionary columns, we only add unique values seen in this group
            let mut seen_dict_indices = vec![false; dict_count + 1];
            for i in start_row..end_row {
                let dict_idx = indices[i] as usize;
                if dict_idx > 0 && dict_idx <= dict_count {
                    seen_dict_indices[dict_idx] = true;
                }
            }

            // Count unique values for this group
            let unique_count = seen_dict_indices.iter().filter(|&&x| x).count().max(1);

            let mut filter =
                RowGroupBloomFilter::new(unique_count, false_positive_rate, start_row, end_row);

            // Add unique dictionary values
            for (dict_idx, &seen) in seen_dict_indices.iter().enumerate().skip(1) {
                if seen && dict_idx <= dict_strings.len() {
                    filter.insert(dict_strings[dict_idx - 1]);
                }
            }

            filters.push(filter);
        }

        Self {
            filters,
            column_name: column_name.to_string(),
            row_group_size,
        }
    }

    /// Find row groups that might contain the given value
    /// Returns indices of row groups to scan
    pub fn find_candidate_groups(&self, value: &[u8]) -> Vec<usize> {
        self.filters
            .iter()
            .enumerate()
            .filter(|(_, f)| f.might_contain(value))
            .map(|(i, _)| i)
            .collect()
    }

    /// Get row ranges to scan (skipping groups that definitely don't match)
    pub fn get_scan_ranges(&self, value: &[u8]) -> Vec<(usize, usize)> {
        self.filters
            .iter()
            .filter(|f| f.might_contain(value))
            .map(|f| (f.start_row, f.end_row))
            .collect()
    }

    /// Calculate skip ratio (how much we can skip)
    pub fn skip_ratio(&self, value: &[u8]) -> f64 {
        if self.filters.is_empty() {
            return 0.0;
        }

        let total_rows: usize = self.filters.iter().map(|f| f.end_row - f.start_row).sum();
        let scan_rows: usize = self
            .filters
            .iter()
            .filter(|f| f.might_contain(value))
            .map(|f| f.end_row - f.start_row)
            .sum();

        if total_rows == 0 {
            0.0
        } else {
            1.0 - (scan_rows as f64 / total_rows as f64)
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: column_name_len(4) + column_name + row_group_size(8) + num_filters(4)
        let name_bytes = self.column_name.as_bytes();
        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&(self.row_group_size as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.filters.len() as u32).to_le_bytes());

        // Each filter with length prefix
        for filter in &self.filters {
            let filter_bytes = filter.to_bytes();
            bytes.extend_from_slice(&(filter_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(&filter_bytes);
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Bloom index data too short",
            ));
        }

        let mut offset = 0;

        let name_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let column_name = String::from_utf8_lossy(&data[offset..offset + name_len]).to_string();
        offset += name_len;

        let row_group_size =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        let num_filters = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let mut filters = Vec::with_capacity(num_filters);
        for _ in 0..num_filters {
            if offset + 4 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Unexpected end of bloom index",
                ));
            }
            let filter_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + filter_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Bloom filter data incomplete",
                ));
            }
            let filter = RowGroupBloomFilter::from_bytes(&data[offset..offset + filter_len])?;
            filters.push(filter);
            offset += filter_len;
        }

        Ok(Self {
            filters,
            column_name,
            row_group_size,
        })
    }
}

/// Default row group size for bloom filters (64K rows)
pub const BLOOM_ROW_GROUP_SIZE: usize = 65536;

/// Default false positive rate (1%)
pub const BLOOM_FP_RATE: f64 = 0.01;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_group_bloom_filter() {
        let mut filter = RowGroupBloomFilter::new(100, 0.01, 0, 100);

        filter.insert(b"apple");
        filter.insert(b"banana");
        filter.insert(b"cherry");

        assert!(filter.might_contain(b"apple"));
        assert!(filter.might_contain(b"banana"));
        assert!(filter.might_contain(b"cherry"));
        // Note: might_contain can return true for values not in filter (false positive)
        // but should rarely do so
    }

    #[test]
    fn test_bloom_serialization() {
        let mut filter = RowGroupBloomFilter::new(100, 0.01, 0, 100);
        filter.insert(b"test1");
        filter.insert(b"test2");

        let bytes = filter.to_bytes();
        let restored = RowGroupBloomFilter::from_bytes(&bytes).unwrap();

        assert!(restored.might_contain(b"test1"));
        assert!(restored.might_contain(b"test2"));
        assert_eq!(restored.start_row, 0);
        assert_eq!(restored.end_row, 100);
    }

    #[test]
    fn test_column_bloom_index() {
        let offsets = vec![0u32, 5, 11, 17, 23, 29]; // 5 strings
        let data = b"appleorangebananagrapemeloncoding";

        let index = ColumnBloomIndex::build_from_strings(
            "fruit", &offsets, data, 2, // 2 rows per group
            0.01,
        );

        // apple and orange in group 0
        assert!(index.filters[0].might_contain(b"apple"));
        // banana in group 1
        assert!(index.filters[1].might_contain(b"banana"));
    }
}
