# Release Notes

This page documents the major features and improvements in each version of LynseDB. Only versions with the `v` prefix are official releases.

## v0.4.0 (Upcoming)

**Key Features:**
- 🔧 **Refactored IVFIndex**: Shared KMeans implementation for improved code reusability and maintainability
- 🌐 **Enhanced LocalCollection**: Added wire_dtype parameter support for better HTTP API compatibility
- 🐍 **Improved Python Bindings**: Refactored bindings architecture and enhanced cluster functionality
- 📚 **Enhanced Documentation**: Expanded index documentation with more naming variants and examples

**Improvements:**
- Better resource management in indexing
- Improved cluster operation capabilities
- More comprehensive index documentation

---

## v0.3.0

**Major Release - Float16 Support and Advanced Indexing**

**Key Features:**
- ✨ **Float16 Support**: Added native float16 (f16) vector storage for 50% memory reduction
- 📊 **PolarVec Enhancement**: Metric-aware auxiliary storage for improved quantization
- 🔍 **Enhanced Index Management**: Improved field-specific indexing options
- 📖 **Documentation Updates**: New tutorials and comprehensive guides

**Improvements:**
- Better resource management in vector storage
- Enhanced snapshot functionality
- Improved index building and search documentation
- Optimized metadata filtering

---

## v0.2.0

**Major Release - Rust Architecture and Advanced Features**

**Key Features:**
- 🏗️ **Rust-First Architecture**: Complete restructuring to Rust-first with simplified Python API
- 🎯 **Advanced Quantization**: Added PQ (Product Quantization), RaBitQ, and PolarVec index modes with two-pass search
- 📦 **Mmap-backed Storage**: Memory-mapped vector storage for efficient large-scale data handling
- 🗂️ **IVF_FLAT Indexing**: Inverted File index with flat quantization support
- ✍️ **Write-Ahead Logging (WAL)**: Durability and crash recovery support
- 🔎 **Approximate Search**: Approximate search capabilities with configurable precision
- 📊 **Optimized Field Indexing**: In-memory field index and apex_id_map for fast queries
- 🎯 **Optimized Filtered Search**: Dual-strategy filtered search for better performance

**Improvements:**
- Removed Python fallback implementations for better performance
- Enhanced query capabilities with numeric equality and IN filters
- Improved context management and resource handling
- Better integration between Python and Rust components
- Comprehensive testing for metadata indexing and search functionality
- API method signature improvements and documentation enhancements
- Observability improvements: slow query alerts and audit logging capabilities

---

## v0.1.6

**Final Stability Release - Pure Python**

**Key Features:**
- Finalized v0.1 series with Python implementation
- Performance optimizations and bug fixes
- Resource management improvements

---

## v0.1.5

**Maintenance Release - Pure Python**

**Key Features:**
- Workflow improvements
- Testing enhancements
- Documentation updates

---

## v0.1.4

**Performance Release - Pure Python**

**Key Features:**
- 🛡️ **Safer MMAP Reading**: Improved memory-mapped file reading strategy for better stability

---

## v0.1.3

**Bug Fix Release - Pure Python**

**Key Features:**
- 🛡️ **Enhanced MMAP Reading**: Improved memory-mapped file reading strategy
- Better error handling

---

## v0.1.2

**Compatibility Release - Pure Python**

**Key Features:**
- 🐛 **Warning Fixes**: Fixed deprecation warnings
- Improved Python compatibility

---

## v0.1.1

**Stability Release - Pure Python**

**Key Features:**
- 🔒 **Resource Management**: Used context managers to prevent unclosed mmap file errors
- ⚙️ **File Handling**: Added logic to handle PermissionError during file deletion
- 🔄 **Thread Safety**: Implemented thread locks to prevent race conditions
- Windows compatibility improvements

---

## v0.1.0

**Major Release - Python Implementation with HNSW**

**Key Features:**
- 🐍 **Python Core**: Pure Python implementation of vector search
- 🌐 **HNSW Algorithm**: Hierarchical Navigable Small World index implementation
- 🗄️ **Storage Backend**: Efficient vector storage and retrieval
- 📦 **File-based Storage**: Support for memory-mapped file access
- 🔍 **Metadata Filtering**: Support for filtering during search
- 📐 **Common Metrics**: L2, Cosine, and other distance metrics

**Improvements:**
- Foundation of vector database functionality
- Python-first API design
- Support for multiprocessing

---

## v0.0.2

**Early Release - Bug Fixes**

**Key Features:**
- 🐛 **FileNotFoundError Fix**: Fixed file not found error (#9)
- Initial stability improvements

---

## v0.0.1

**Initial Release - Foundation**

**Features:**
- Basic vector database functionality
- Python-first API design
- Support for common distance metrics (L2, Cosine)
- Metadata filtering capabilities
- Initial documentation and examples

---

## Migration Guide

### From v0.3.0 to v0.4.0
- IVFIndex refactoring is internal; no API changes required
- HTTP API compatibility improved; wire_dtype parameter now supported
- Cluster functionality enhanced; no breaking changes

### From v0.2.0 to v0.3.0
- Float16 vectors are now available; specify `dtype=float16` when creating collections
- PolarVec index is improved; consider rebuilding indices for better performance
- All existing APIs remain compatible

### From v0.1.x to v0.2.0
- **Major architectural change**: Migration from pure Python to Rust backend
- HNSW index remains the default; API remains similar for backwards compatibility
- Significant performance improvements (2.4x faster HNSW)
- Some API changes due to architecture shift; please review migration guide

### From v0.0.x to v0.1.0
- Complete rewrite with Rust core; some API changes may occur
- HNSW index replaces previous indexing methods
- Major performance improvements with improved resource management

---

## Support

- For issues or questions about a specific version, please check the [Troubleshooting Guide](tutorials/troubleshooting.md)
- For API documentation, see the [Python Client Reference](client.md) and [HTTP API Documentation](http_api/serve_api/app.md)
- For upcoming features, see the [Production Roadmap](production_roadmap.md)
