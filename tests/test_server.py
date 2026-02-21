"""Integration tests for VecGrep MCP server tools."""

from __future__ import annotations

import threading
from pathlib import Path

from vecgrep.server import (
    _do_index,
    _get_index_lock,
    _get_store,
    get_index_status,
    index_codebase,
    search_code,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_py(path: Path, name: str, content: str) -> Path:
    f = path / name
    f.write_text(content, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexCodebase:
    def test_indexes_python_files(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo():\n    return 42\n")
        result = _do_index(str(tmp_path))
        assert "1 file(s)" in result
        assert "Error" not in result

    def test_nonexistent_path_returns_error(self):
        result = _do_index("/nonexistent/path/xyzzy12345")
        assert result.startswith("Error")

    def test_incremental_skips_unchanged(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result2 = _do_index(str(tmp_path))
        assert "skipped" in result2

    def test_force_reindexes_all(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result = _do_index(str(tmp_path), force=True)
        assert "1 file(s)" in result
        assert "Error" not in result

    def test_index_codebase_tool_wraps_do_index(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        result = index_codebase(str(tmp_path))
        assert "Error" not in result


class TestOrphanCleanup:
    def test_deleted_file_chunks_removed_on_reindex(self, tmp_path):
        f1 = _write_py(tmp_path, "a.py", "def foo(): pass\n")
        f2 = _write_py(tmp_path, "b.py", "def bar(): pass\n")

        _do_index(str(tmp_path))

        root = str(tmp_path.resolve())
        with _get_store(root) as store:
            hashes = store.get_file_hashes()
        assert str(f1.resolve()) in hashes
        assert str(f2.resolve()) in hashes

        f2.unlink()
        result = _do_index(str(tmp_path))
        assert "1 orphan" in result

        with _get_store(root) as store:
            hashes_after = store.get_file_hashes()
        assert str(f2.resolve()) not in hashes_after


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearchCode:
    def test_returns_relevant_result(self, tmp_path):
        _write_py(tmp_path, "auth.py", "def authenticate_user(username, password):\n    pass\n")
        _do_index(str(tmp_path))
        result = search_code("user authentication", str(tmp_path), top_k=5)
        assert "auth.py" in result
        assert "Error" not in result

    def test_negative_top_k_does_not_crash(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result = search_code("foo", str(tmp_path), top_k=-5)
        # max(1, min(-5, 20)) = 1 — should return 1 result without error
        assert "Error" not in result

    def test_empty_query_returns_error(self, tmp_path):
        result = search_code("   ", str(tmp_path), top_k=5)
        assert "Error" in result

    def test_long_query_returns_error(self, tmp_path):
        result = search_code("x" * 501, str(tmp_path), top_k=5)
        assert "Error" in result
        assert "long" in result.lower() or "500" in result

    def test_auto_indexes_on_first_search(self, tmp_path):
        _write_py(tmp_path, "a.py", "def compute(): return 1\n")
        # No explicit index call — search_code should trigger it
        result = search_code("compute function", str(tmp_path), top_k=5)
        assert "Error" not in result


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentLock:
    def test_second_index_call_returns_in_progress(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        root_str = str(tmp_path.resolve())

        lock = _get_index_lock(root_str)
        assert lock.acquire(blocking=False)

        captured: list[str] = []

        def run():
            captured.append(_do_index(str(tmp_path)))

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)
        lock.release()

        assert len(captured) == 1
        assert "already in progress" in captured[0]


# ---------------------------------------------------------------------------
# get_index_status
# ---------------------------------------------------------------------------


class TestGetIndexStatus:
    def test_returns_expected_fields(self, tmp_path):
        _write_py(tmp_path, "a.py", "def foo(): pass\n")
        _do_index(str(tmp_path))
        result = get_index_status(str(tmp_path))
        assert "Files indexed" in result
        assert "Total chunks" in result
        assert "Last indexed" in result
        assert "Index size" in result

    def test_nonexistent_path_shows_zero_files(self, tmp_path):
        # No indexing — status should still return without raising
        result = get_index_status(str(tmp_path))
        assert "Files indexed:  0" in result
