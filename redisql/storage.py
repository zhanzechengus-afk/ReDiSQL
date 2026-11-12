from pathlib import Path
import hashlib
import json
import os
import sqlite3
import tempfile
from urllib.parse import quote


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sqlite_fingerprint(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with sqlite3.connect("file:" + quote(str(path)) + "?mode=ro", uri=True) as connection:
        connection.execute("BEGIN")
        for line in connection.iterdump():
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def state_fingerprint(module):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(canonical_json([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(b"\0")
        value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    atomic_write(path, "".join(canonical_json(record) + "\n" for record in records))


class RepairCache:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS repairs (key TEXT PRIMARY KEY, context TEXT NOT NULL, response TEXT NOT NULL)")
        self.connection.commit()

    def get(self, context):
        serialized = canonical_json(context)
        row = self.connection.execute("SELECT context, response FROM repairs WHERE key=?", (fingerprint(context),)).fetchone()
        if row is None:
            return None
        if row[0] != serialized:
            raise RuntimeError("repair cache context mismatch")
        return row[1]

    def put(self, context, response):
        if not isinstance(response, str):
            raise TypeError("repair responses must be SQL strings")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO repairs VALUES (?, ?, ?)", (fingerprint(context), canonical_json(context), response))

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()

