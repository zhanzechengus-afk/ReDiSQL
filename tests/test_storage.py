import copy
import json
import sqlite3

import pytest

from redisql import storage
from redisql.storage import RepairCache, atomic_write, canonical_json, file_fingerprint, fingerprint, read_jsonl, sqlite_fingerprint, write_json, write_jsonl


def test_fingerprints_are_deterministic_and_reject_nonfinite_values():
    assert canonical_json({"b": 2, "a": "汉字"}) == '{"a":"汉字","b":2}'
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    with pytest.raises(ValueError):
        fingerprint({"value": float("nan")})


def test_json_and_jsonl_writes_are_utf8_and_newline_terminated(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    values = [{"id": 1, "question": "多少账户？"}, {"id": 2, "question": "Which account?"}]
    write_jsonl(path, values)
    assert read_jsonl(path) == values
    assert path.read_bytes().endswith(b"\n")
    assert len(path.read_text().splitlines()) == 2
    single = tmp_path / "config.json"
    write_json(single, {"value": 7})
    assert json.loads(single.read_text()) == {"value": 7}
    assert single.read_bytes().endswith(b"\n")


def test_failed_atomic_replace_preserves_old_file_and_cleans_temporary_files(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("old\n")
    def fail_replace(source, target):
        raise OSError("replace failed")
    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write(path, "new\n")
    assert path.read_text() == "old\n"
    assert list(tmp_path.iterdir()) == [path]


def context():
    return {
        "dataset": "spider/train",
        "example": "42",
        "schema": {"table": "account", "column": "id"},
        "database": "database-digest",
        "student": "student-digest",
        "teacher": {"model": "CodeLlama", "weights": "teacher-digest"},
        "prompt": "prompt-template",
        "decoding": {"greedy": True, "max_new_tokens": 256},
        "draft": "SELECT id FROM account",
        "feedback": {"category": "denotation", "message": "Executable query; result mismatch"},
    }


@pytest.mark.parametrize("field", ["dataset", "example", "schema", "database", "student", "teacher", "prompt", "decoding", "draft", "feedback"])
def test_cache_requires_exact_context_for_every_recorded_component(tmp_path, field):
    original = context()
    with RepairCache(tmp_path / "cache.sqlite") as cache:
        assert cache.get(original) is None
        cache.put(original, "SELECT COUNT(*) FROM account")
        assert cache.get(copy.deepcopy(original)) == "SELECT COUNT(*) FROM account"
        changed = copy.deepcopy(original)
        changed[field] = {"changed": True}
        assert cache.get(changed) is None


def test_cache_detects_serialized_context_corruption(tmp_path):
    value = context()
    with RepairCache(tmp_path / "cache.sqlite") as cache:
        cache.put(value, "SELECT 1")
        cache.connection.execute("UPDATE repairs SET context=? WHERE key=?", (canonical_json({"corrupt": True}), fingerprint(value)))
        cache.connection.commit()
        with pytest.raises(RuntimeError, match="context mismatch"):
            cache.get(value)


def test_database_mutation_invalidates_a_content_addressed_cache_context(tmp_path):
    path = tmp_path / "source.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE account (id INTEGER)")
        connection.execute("INSERT INTO account VALUES (1)")
    first = context()
    first["database"] = file_fingerprint(path)
    with RepairCache(tmp_path / "cache.sqlite") as cache:
        cache.put(first, "SELECT id FROM account")
        with sqlite3.connect(path) as connection:
            connection.execute("INSERT INTO account VALUES (2)")
        second = copy.deepcopy(first)
        second["database"] = file_fingerprint(path)
        assert first["database"] != second["database"]
        assert cache.get(second) is None


def test_sqlite_fingerprint_includes_committed_wal_content(tmp_path):
    path = tmp_path / "wal.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE account (id INTEGER)")
        connection.execute("INSERT INTO account VALUES (1)")
        connection.commit()
        file_before = file_fingerprint(path)
        content_before = sqlite_fingerprint(path)
        connection.execute("INSERT INTO account VALUES (2)")
        connection.commit()
        assert file_fingerprint(path) == file_before
        assert sqlite_fingerprint(path) != content_before
        assert connection.execute("SELECT id FROM account ORDER BY id").fetchall() == [(1,), (2,)]
    finally:
        connection.close()


def test_cache_persists_and_rejects_nontext_responses(tmp_path):
    path = tmp_path / "cache.sqlite"
    with RepairCache(path) as cache:
        cache.put(context(), "SELECT 1")
        with pytest.raises(TypeError, match="SQL strings"):
            cache.put(context(), {"sql": "SELECT 1"})
    with RepairCache(path) as cache:
        assert cache.get(context()) == "SELECT 1"
