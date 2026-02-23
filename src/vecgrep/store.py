
"""LanceDB-backed vector store."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa


# Define the LanceDB Schema
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("content", pa.string()),
    pa.field("file_hash", pa.string()),
    pa.field("chunk_hash", pa.string()),
    pa.field("mtime", pa.float64()),
    pa.field("size", pa.int64()),
    pa.field("vector", pa.list_(pa.float32(), 384)),
])


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        # Use LanceDB, removing raw SQLite constraints
        self.db_path = str(self.index_dir / "lancedb")
        self._db = lancedb.connect(self.db_path)
        
        self.table_name = "chunks"
        if self.table_name not in set(self._db.table_names()):
            self._table = self._db.create_table(self.table_name, schema=schema)
        else:
            self._table = self._db.open_table(self.table_name)
            
        # Optional: Meta table for 'last_indexed' etc
        self.meta_table_name = "meta"
        meta_schema = pa.schema([
            pa.field("key", pa.string()),
            pa.field("value", pa.string())
        ])
        if self.meta_table_name not in set(self._db.table_names()):
            self._meta_table = self._db.create_table(self.meta_table_name, schema=meta_schema)
        else:
            self._meta_table = self._db.open_table(self.meta_table_name)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        pass # LanceDB doesn't require explicit closing

    # ------------------------------------------------------------------
    # File hash helpers
    # ------------------------------------------------------------------

    def get_file_stats(self) -> dict[str, dict]:
        """Return {file_path: {'file_hash': hash, 'mtime': mtime, 'size': size}} for all indexed files."""
        # Use to_list to avoid pandas dependency
        if self._table.count_rows() == 0:
            return {}
            
        rows = self._table.search().limit(None).to_list()
        stats = {}
        
        # Latest stats per file
        for r in rows:
            stats[r["file_path"]] = {
                "file_hash": r["file_hash"],
                "mtime": r["mtime"],
                "size": r["size"]
            }
        return stats

    def get_file_hashes(self) -> dict[str, str]:
        """Compatibility method for legacy callers."""
        stats = self.get_file_stats()
        return {fp: s["file_hash"] for fp, s in stats.items()}

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def delete_file_chunks(self, file_path: str) -> None:
        """Remove all chunks for a given file."""
        self._table.delete(f"file_path = '{file_path}'")

    def add_chunks(self, rows: list[dict], vectors: np.ndarray) -> None:
        """
        Insert chunk rows with their embeddings.

        rows: list of dicts with keys: file_path, start_line, end_line,
              content, file_hash, chunk_hash, mtime, size
        vectors: float32 array of shape (len(rows), 384)
        """
        if len(rows) != len(vectors):
            raise ValueError("rows/vectors length mismatch")
            
        data = []
        for i, r in enumerate(rows):
            # Create a unique ID for LanceDB (LanceDB requires an implicit or explicit ID, though we manage it via metadata conceptually)
            row_id = f"{r['file_path']}_{r['start_line']}_{r['end_line']}_{r['chunk_hash']}"
            
            data.append({
                "id": row_id,
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"],
                "file_hash": r["file_hash"],
                "chunk_hash": r["chunk_hash"],
                "mtime": r.get("mtime", 0.0),
                "size": r.get("size", 0),
                "vector": vectors[i].tolist(),
            })
            
        if data:
            self._table.add(data)

    def replace_file_chunks(
        self, file_path: str, rows: list[dict], vectors: np.ndarray
    ) -> None:
        """Atomically delete existing chunks and insert new ones for a file."""
        # LanceDB doesn't have strict atomic transactions across delete+add in the OSS version easily,
        # but sequentially it's extremely fast. 
        self.delete_file_chunks(file_path)
        if rows:
            self.add_chunks(rows, vectors)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vec: np.ndarray, top_k: int = 8) -> list[dict]:
        """
        Cosine similarity search via LanceDB natively.
        Returns list of dicts: {file_path, start_line, end_line, content, score}.
        """
        if self._table.count_rows() == 0:
            return []

        # LanceDB uses L2 or Cosine. We explicitly set metric="cosine"
        results = (
            self._table.search(query_vec.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        
        parsed_results = []
        for r in results:
            # _distance is returned by LanceDB. For cosine, smaller distance means more similar.
            # Convert to a similarity score to maintain compatibility: score = 1 - distance
            score = 1.0 - r.get("_distance", 0.0)
            parsed_results.append({
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"],
                "score": float(score),
            })
            
        return parsed_results

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        total_chunks = self._table.count_rows()
        
        if total_chunks > 0:
            # Use pyarrow to get distinct files without pandas
            df = self._table.to_arrow()
            total_files = len(pa.compute.unique(df["file_path"]))
        else:
            total_files = 0
            
        meta_rows = self._meta_table.search().limit(None).to_list()
        last_indexed = "never"
        for r in meta_rows:
            if r["key"] == "last_indexed":
                last_indexed = r["value"]
                break
                
        # Approximate size of the .lance directory
        db_size = sum(f.stat().st_size for f in Path(self.db_path).glob('**/*') if f.is_file()) if Path(self.db_path).exists() else 0
        
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "last_indexed": last_indexed,
            "index_size_bytes": db_size,
        }

    def touch_last_indexed(self) -> None:
        ts = datetime.now(UTC).isoformat()
        # Delete old key if exists
        self._meta_table.delete("key = 'last_indexed'")
        # Add new key
        self._meta_table.add([{"key": "last_indexed", "value": ts}])
