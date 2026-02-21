# Changelog

All notable changes to VecGrep are documented here.

---

## [1.0.0] — 2026-02-22

First stable release.

### Added

**Core features**
- AST-based code chunking via `tree-sitter-languages` for Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, Ruby, Swift, Kotlin, and C#
- Local embeddings using `all-MiniLM-L6-v2` (384-dim, ~80 MB one-time download, no API key required)
- SQLite-backed vector store at `~/.vecgrep/<project-hash>/index.db`
- Cosine similarity search with in-memory embedding cache (eliminates per-query full-table scan)
- Incremental indexing via SHA-256 file hashing — unchanged files are skipped on re-index
- Orphan cleanup — chunks for deleted files are removed on the next index run
- Three MCP tools: `index_codebase`, `search_code`, `get_index_status`

**Robustness**
- Per-path threading lock prevents concurrent indexing corruption
- Atomic file updates via `replace_file_chunks()` (DELETE + INSERT in a single transaction)
- Context-manager support on `VectorStore` guarantees connection closure on exceptions
- `top_k` clamped to `max(1, min(top_k, 20))` — negative values no longer crash
- Query validation: empty queries and queries over 500 characters return a clean error string
- `followlinks=False` in directory walk prevents symlink loops
- Try/except on all MCP tool functions — errors surface as readable strings, not tracebacks

**Tooling**
- 52 tests across `test_store`, `test_chunker`, `test_embedder`, and `test_server`
- CI with ruff lint, pytest with coverage upload to Codecov, and non-blocking pyright
- Published to PyPI — install with `pip install vecgrep` or run directly with `uvx vecgrep`

### Changed

- `pyproject.toml`: tightened dependency version ranges, added dev extras (`pytest`, `ruff`, `pyright`)

---

## [0.1.0] — 2026-02-22 *(pre-release, not published)*

Initial working prototype — basic indexing and search functional but without hardening, tests, or PyPI packaging.
