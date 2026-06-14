from __future__ import annotations

"""Small LynseDB cluster coordinator.

This module intentionally keeps clustering modest: one active coordinator
process owns metadata, shards are ordinary LynseDB HTTP servers, and replicas
are maintained by coordinator-side mirroring. It avoids external services while
giving the common happy path automatic shard failover.
"""

import argparse
import array
import hashlib
import json
import os
import socket
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx


DEFAULT_BUCKET_COUNT = 4096
DEFAULT_HEALTH_INTERVAL_SECS = 1.0
DEFAULT_HEALTH_FAILURES = 3
DEFAULT_REQUEST_TIMEOUT_SECS = 30.0
REPLICA_ACTIVE = "active"
REPLICA_STALE = "stale"

RPC_OP_PING = 1
RPC_OP_SEARCH = 2
RPC_OP_BATCH_SEARCH = 3
RPC_OP_BULK_ADD_BINARY_IDS = 4
RPC_OP_UPSERT_BINARY_IDS = 5
RPC_OP_DELETE_ITEMS = 6
RPC_OP_RESTORE_ITEMS = 7
FIELDS_BINARY_MAGIC = b"LDBF1"


def _normalize_uri(uri: str) -> str:
    uri = str(uri).strip()
    if uri.endswith("/"):
        uri = uri[:-1]
    return uri


def _derive_rpc_target(uri: str) -> tuple[str, int]:
    parsed = urlparse(_normalize_uri(uri))
    if not parsed.hostname:
        raise ValueError(f"cannot derive RPC target from URI: {uri}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    rpc_port = port + 10000 if port <= 55535 else port - 10000
    return parsed.hostname, rpc_port


def _json_success(params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "success", "params": params or {}}


def _hash_u64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _external_id_key(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid LynseDB ID")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("integer IDs must be non-negative")
        return f"int:{value}"
    if isinstance(value, str):
        if not value:
            raise ValueError("string IDs cannot be empty")
        return f"str:{value}"
    raise TypeError("IDs must be strings or non-negative integers")


def _is_ascending_index(index_mode: str | None) -> bool:
    upper = (index_mode or "FLAT").upper()
    return any(token in upper for token in ("L2", "COS", "HAMMING", "JACCARD"))


def _split_search_binary(buf: bytes, offset: int = 0):
    n = int.from_bytes(buf[offset:offset + 4], "little")
    offset += 4
    ids = [
        int.from_bytes(buf[offset + i * 8:offset + (i + 1) * 8], "little")
        for i in range(n)
    ]
    offset += n * 8
    distances = []
    for i in range(n):
        raw = buf[offset + i * 4:offset + (i + 1) * 4]
        distances.append(float(struct.unpack("<f", raw)[0]))
    offset += n * 4
    fields_len = int.from_bytes(buf[offset:offset + 4], "little")
    offset += 4
    fields = json.loads(buf[offset:offset + fields_len]) if fields_len else []
    offset += fields_len
    return ids, distances, fields, offset


def _split_vectors_binary(buf: bytes):
    offset = 0
    n = int.from_bytes(buf[offset:offset + 4], "little")
    offset += 4
    dim = int.from_bytes(buf[offset:offset + 4], "little")
    offset += 4
    vectors = []
    for _ in range(n):
        row = []
        for _ in range(dim):
            raw = buf[offset:offset + 4]
            row.append(float(struct.unpack("<f", raw)[0]))
            offset += 4
        vectors.append(row)
    ids = [
        int.from_bytes(buf[offset + i * 8:offset + (i + 1) * 8], "little")
        for i in range(n)
    ]
    offset += n * 8
    fields_len = int.from_bytes(buf[offset:offset + 4], "little")
    offset += 4
    fields = json.loads(buf[offset:offset + fields_len]) if fields_len else []
    offset += fields_len
    return vectors, ids, fields, dim, offset


def _encode_search_binary(
    ids: list[int],
    distances: list[float],
    fields: list[dict[str, Any]] | None = None,
) -> bytes:
    fields_json = b"" if not fields else json.dumps(fields, separators=(",", ":")).encode("utf-8")
    parts = [len(ids).to_bytes(4, "little")]
    parts.extend(int(i).to_bytes(8, "little", signed=False) for i in ids)
    parts.extend(struct.pack("<f", float(d)) for d in distances)
    parts.append(len(fields_json).to_bytes(4, "little"))
    parts.append(fields_json)
    return b"".join(parts)


def _encode_vectors_binary(
    vectors: list[list[float]],
    ids: list[int],
    fields: list[dict[str, Any]] | None = None,
) -> bytes:
    dim = len(vectors[0]) if vectors else 0
    fields_json = b"" if not fields else json.dumps(fields, separators=(",", ":")).encode("utf-8")
    parts = [len(vectors).to_bytes(4, "little"), dim.to_bytes(4, "little")]
    for row in vectors:
        parts.extend(struct.pack("<f", float(value)) for value in row)
    parts.extend(int(i).to_bytes(8, "little", signed=False) for i in ids)
    parts.append(len(fields_json).to_bytes(4, "little"))
    parts.append(fields_json)
    return b"".join(parts)


def _vectors_to_f32_bytes(vectors: list[Any]) -> tuple[bytes, int, int]:
    if not vectors:
        return b"", 0, 0
    dim = len(vectors[0])
    flat = array.array("f")
    for vector in vectors:
        if len(vector) != dim:
            raise ValueError("all vectors in an RPC batch must have the same dimension")
        flat.extend(float(value) for value in vector)
    if sys.byteorder != "little":
        flat.byteswap()
    return flat.tobytes(), len(vectors), dim


def _encode_ids_for_wire(ids: list[int]) -> tuple[bytes, dict[str, Any]]:
    ids = [int(item_id) for item_id in ids]
    if ids:
        start = ids[0]
        if all(item_id == start + idx for idx, item_id in enumerate(ids)):
            return b"", {"ids_encoding": "range", "ids_start": start}

    raw = b"".join(struct.pack("<Q", item_id) for item_id in ids)
    return raw, {"ids_encoding": "raw"}


def _normalize_vector_encoding(vector_encoding: str | None = None) -> str:
    value = (vector_encoding or "float32").lower()
    if value in {"float32", "f32"}:
        return "float32"
    if value in {"float16", "f16", "fp16"}:
        return "float16"
    raise ValueError(f"unsupported vector_encoding {vector_encoding!r}")


def _vector_wire_width(vector_encoding: str | None = None) -> int:
    return 2 if _normalize_vector_encoding(vector_encoding) == "float16" else 4


def _pack_u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _pack_i64(value: int) -> bytes:
    return struct.pack("<q", int(value))


def _pack_u64(value: int) -> bytes:
    return struct.pack("<Q", int(value))


def _pack_f64(value: float) -> bytes:
    return struct.pack("<d", float(value))


def _pack_string(value: Any) -> bytes:
    raw = str(value).encode("utf-8")
    return _pack_u32(len(raw)) + raw


def _encode_binary_value(value: Any) -> bytes:
    if value is None:
        return b"\x00"
    if value is False:
        return b"\x01"
    if value is True:
        return b"\x02"
    if isinstance(value, int):
        if value >= 0:
            return b"\x04" + _pack_u64(value)
        return b"\x03" + _pack_i64(value)
    if isinstance(value, float):
        return b"\x05" + _pack_f64(value)
    if isinstance(value, str):
        return b"\x06" + _pack_string(value)
    if isinstance(value, (list, tuple)):
        parts = [b"\x07", _pack_u32(len(value))]
        parts.extend(_encode_binary_value(item) for item in value)
        return b"".join(parts)
    if isinstance(value, dict):
        parts = [b"\x08", _pack_u32(len(value))]
        for key, item in value.items():
            parts.append(_pack_string(key))
            parts.append(_encode_binary_value(item))
        return b"".join(parts)
    return b"\x06" + _pack_string(value)


def _encode_fields_binary(fields: list[Any]) -> bytes:
    parts = [FIELDS_BINARY_MAGIC, _pack_u32(len(fields))]
    for field in fields:
        if field is None:
            parts.append(b"\x00")
            continue
        if not isinstance(field, dict):
            raise TypeError("field payload entries must be dict or None")
        parts.append(b"\x01")
        parts.append(_pack_u32(len(field)))
        for key, value in field.items():
            parts.append(_pack_string(key))
            parts.append(_encode_binary_value(value))
    return b"".join(parts)


class _BinaryFieldCursor:
    def __init__(self, data: bytes | memoryview):
        self.data = memoryview(data)
        self.pos = 0

    def read(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise ValueError("unexpected end of binary fields payload")
        out = self.data[self.pos:end].tobytes()
        self.pos = end
        return out

    def read_u8(self) -> int:
        return self.read(1)[0]

    def read_u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def read_i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def read_f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def read_string(self) -> str:
        size = self.read_u32()
        return self.read(size).decode("utf-8")

    def ensure_finished(self) -> None:
        if self.pos != len(self.data):
            raise ValueError("trailing bytes in binary fields payload")


def _decode_binary_object(cursor: _BinaryFieldCursor, depth: int = 0) -> dict[str, Any]:
    if depth > 128:
        raise ValueError("binary fields payload is nested too deeply")
    count = cursor.read_u32()
    out: dict[str, Any] = {}
    for _ in range(count):
        key = cursor.read_string()
        out[key] = _decode_binary_value(cursor, depth + 1)
    return out


def _decode_binary_value(cursor: _BinaryFieldCursor, depth: int = 0) -> Any:
    if depth > 128:
        raise ValueError("binary fields payload is nested too deeply")
    tag = cursor.read_u8()
    if tag == 0:
        return None
    if tag == 1:
        return False
    if tag == 2:
        return True
    if tag == 3:
        return cursor.read_i64()
    if tag == 4:
        return cursor.read_u64()
    if tag == 5:
        return cursor.read_f64()
    if tag == 6:
        return cursor.read_string()
    if tag == 7:
        return [_decode_binary_value(cursor, depth + 1) for _ in range(cursor.read_u32())]
    if tag == 8:
        return _decode_binary_object(cursor, depth + 1)
    raise ValueError(f"unknown binary value tag {tag}")


def _decode_fields_binary(raw: bytes | memoryview) -> list[Any]:
    view = memoryview(raw)
    if not view.tobytes().startswith(FIELDS_BINARY_MAGIC):
        raise ValueError("invalid binary fields payload magic")
    cursor = _BinaryFieldCursor(view[len(FIELDS_BINARY_MAGIC):])
    count = cursor.read_u32()
    fields: list[Any] = []
    for _ in range(count):
        present = cursor.read_u8()
        if present == 0:
            fields.append(None)
        elif present == 1:
            fields.append(_decode_binary_object(cursor))
        else:
            raise ValueError(f"invalid field presence tag {present}")
    cursor.ensure_finished()
    return fields


def _split_binary_items_payload(
    buf: bytes,
    n_vectors: int,
    dim: int,
    vector_encoding: str | None = None,
    ids_encoding: str | None = None,
    ids_start: int | str | None = None,
) -> tuple[memoryview, list[int], list[Any] | None, int]:
    vector_bytes = int(n_vectors) * int(dim) * _vector_wire_width(vector_encoding)
    if len(buf) < vector_bytes:
        raise ValueError(f"expected at least {vector_bytes} vector bytes, got {len(buf)}")
    view = memoryview(buf)
    offset = vector_bytes
    encoding = (ids_encoding or "raw").lower()
    if encoding == "range":
        if ids_start is None:
            raise ValueError("ids_start is required for range id encoding")
        start = int(ids_start)
        ids = list(range(start, start + int(n_vectors)))
    elif encoding in {"raw", ""}:
        id_bytes = int(n_vectors) * 8
        if len(buf) < offset + id_bytes:
            got = len(buf) - offset
            raise ValueError(f"expected at least {id_bytes} id bytes after vectors, got {got}")
        ids = [
            struct.unpack("<Q", view[offset + i * 8:offset + (i + 1) * 8])[0]
            for i in range(int(n_vectors))
        ]
        offset += id_bytes
    else:
        raise ValueError(f"unsupported ids_encoding {ids_encoding!r}")

    fields = _decode_fields_binary(view[offset:]) if offset < len(buf) else None
    if fields is not None and len(fields) != int(n_vectors):
        raise ValueError(f"fields length ({len(fields)}) must match n_vectors ({n_vectors})")
    return view[:vector_bytes], ids, fields, offset


def _merge_pairs(
    results: list[tuple[list[Any], list[float], list[dict[str, Any]]]],
    k: int,
    ascending: bool,
    return_fields: bool,
) -> tuple[list[Any], list[float], list[dict[str, Any]]]:
    merged: list[tuple[Any, float, dict[str, Any] | None]] = []
    for ids, scores, fields in results:
        for idx, (item_id, score) in enumerate(zip(ids, scores)):
            field = fields[idx] if return_fields and idx < len(fields) else None
            merged.append((item_id, float(score), field))
    merged.sort(key=lambda item: item[1], reverse=not ascending)
    top = merged[:k]
    ids = [item[0] for item in top]
    distances = [item[1] for item in top]
    out_fields = [item[2] or {} for item in top] if return_fields else []
    return ids, distances, out_fields


class ClusterState:
    def __init__(self, path: Path, seed_config: dict[str, Any] | None = None):
        self.path = Path(path)
        self._lock = threading.RLock()
        if self.path.exists():
            self.data = self._read_state()
        else:
            if not seed_config:
                raise ValueError("cluster state does not exist and no cluster config was provided")
            self.data = self._initial_state(seed_config)
            self.save()

    def _read_state(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._normalize_state(data)
        return data

    @staticmethod
    def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
        groups = []
        raw_groups = config.get("shard_groups") or config.get("shards") or []
        if not raw_groups:
            raise ValueError("cluster config requires at least one shard group")
        for idx, group in enumerate(raw_groups):
            name = str(group.get("name") or f"sg{idx}")
            primary = _normalize_uri(group["primary"])
            replicas = []
            for replica in group.get("replicas", []):
                if isinstance(replica, str):
                    replicas.append({"uri": _normalize_uri(replica), "state": REPLICA_ACTIVE})
                else:
                    replicas.append({
                        "uri": _normalize_uri(replica["uri"]),
                        "state": replica.get("state", REPLICA_ACTIVE),
                    })
            groups.append({
                "name": name,
                "primary": primary,
                "primary_epoch": int(group.get("primary_epoch", 1)),
                "replicas": replicas,
            })

        return {
            "format": "lynsedb-cluster-state",
            "version": 1,
            "meta_epoch": 1,
            "bucket_count": int(config.get("bucket_count", DEFAULT_BUCKET_COUNT)),
            "write_mirror_replicas": bool(config.get("write_mirror_replicas", True)),
            "databases": list(config.get("databases", [])),
            "collections": dict(config.get("collections", {})),
            "shard_groups": groups,
        }

    @staticmethod
    def _normalize_state(data: dict[str, Any]) -> None:
        data.setdefault("format", "lynsedb-cluster-state")
        data.setdefault("version", 1)
        data.setdefault("meta_epoch", 1)
        data.setdefault("bucket_count", DEFAULT_BUCKET_COUNT)
        data.setdefault("write_mirror_replicas", True)
        data.setdefault("databases", [])
        data.setdefault("collections", {})
        for group in data.setdefault("shard_groups", []):
            group["primary"] = _normalize_uri(group["primary"])
            group.setdefault("primary_epoch", 1)
            normalized = []
            for replica in group.get("replicas", []):
                if isinstance(replica, str):
                    normalized.append({"uri": _normalize_uri(replica), "state": REPLICA_ACTIVE})
                else:
                    normalized.append({
                        "uri": _normalize_uri(replica["uri"]),
                        "state": replica.get("state", REPLICA_ACTIVE),
                    })
            group["replicas"] = normalized

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            payload = json.dumps(self.data, indent=2, sort_keys=True).encode("utf-8")
            with tmp.open("wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)

    def bump_epoch(self) -> None:
        self.data["meta_epoch"] = int(self.data.get("meta_epoch", 0)) + 1

    def databases(self) -> list[str]:
        with self._lock:
            return sorted(set(self.data.get("databases", [])))

    def ensure_database(self, name: str) -> None:
        with self._lock:
            dbs = set(self.data.setdefault("databases", []))
            if name not in dbs:
                dbs.add(name)
                self.data["databases"] = sorted(dbs)
                self.bump_epoch()
                self.save()

    def remove_database(self, name: str) -> None:
        with self._lock:
            self.data["databases"] = [db for db in self.data.get("databases", []) if db != name]
            prefix = f"{name}/"
            for key in list(self.data.get("collections", {}).keys()):
                if key.startswith(prefix):
                    del self.data["collections"][key]
            self.bump_epoch()
            self.save()

    def collection_key(self, db_name: str, coll_name: str) -> str:
        return f"{db_name}/{coll_name}"

    def get_collection(self, db_name: str, coll_name: str) -> dict[str, Any] | None:
        with self._lock:
            coll = self.data.get("collections", {}).get(self.collection_key(db_name, coll_name))
            return dict(coll) if coll else None

    def upsert_collection(
        self,
        db_name: str,
        coll_name: str,
        dim: int,
        description: str | None,
        dtypes: str,
        drop_if_exists: bool,
    ) -> dict[str, Any]:
        with self._lock:
            key = self.collection_key(db_name, coll_name)
            group_names = [group["name"] for group in self.data["shard_groups"]]
            bucket_count = int(self.data.get("bucket_count", DEFAULT_BUCKET_COUNT))
            existing = self.data["collections"].get(key)
            if existing and not drop_if_exists:
                return existing
            bucket_to_group = [group_names[i % len(group_names)] for i in range(bucket_count)]
            coll = {
                "dim": int(dim or 0),
                "chunk_size": 100000,
                "description": description,
                "dtypes": dtypes or "float32",
                "index_mode": "FLAT",
                "next_id": 0,
                "bucket_count": bucket_count,
                "bucket_to_group": bucket_to_group,
            }
            self.data["collections"][key] = coll
            dbs = set(self.data.setdefault("databases", []))
            dbs.add(db_name)
            self.data["databases"] = sorted(dbs)
            self.bump_epoch()
            self.save()
            return coll

    def drop_collection(self, db_name: str, coll_name: str) -> None:
        with self._lock:
            self.data.get("collections", {}).pop(self.collection_key(db_name, coll_name), None)
            self.bump_epoch()
            self.save()

    def update_collection_index(self, db_name: str, coll_name: str, index_mode: str | None) -> None:
        with self._lock:
            coll = self.data.get("collections", {}).get(self.collection_key(db_name, coll_name))
            if coll is not None:
                coll["index_mode"] = index_mode
                self.bump_epoch()
                self.save()

    def update_collection_dim_if_unset(self, db_name: str, coll_name: str, dim: int) -> None:
        if not dim:
            return
        with self._lock:
            coll = self.data.get("collections", {}).get(self.collection_key(db_name, coll_name))
            if coll is not None and int(coll.get("dim") or 0) == 0:
                coll["dim"] = int(dim)
                self.bump_epoch()
                self.save()

    def allocate_ids(self, db_name: str, coll_name: str, count: int) -> list[int]:
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            start = int(coll.get("next_id", 0))
            ids = list(range(start, start + count))
            coll["next_id"] = start + count
            self.bump_epoch()
            self.save()
            return ids

    def group_for_id(self, db_name: str, coll_name: str, item_id: int) -> dict[str, Any]:
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            bucket_count = int(coll.get("bucket_count", self.data.get("bucket_count", DEFAULT_BUCKET_COUNT)))
            bucket = _hash_u64(f"{db_name}/{coll_name}/{int(item_id)}") % bucket_count
            group_name = coll["bucket_to_group"][bucket]
            return self.group_by_name(group_name)

    def group_for_external_id(self, db_name: str, coll_name: str, item_id: Any) -> dict[str, Any]:
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            bucket_count = int(coll.get("bucket_count", self.data.get("bucket_count", DEFAULT_BUCKET_COUNT)))
            bucket = _hash_u64(f"{db_name}/{coll_name}/{_external_id_key(item_id)}") % bucket_count
            group_name = coll["bucket_to_group"][bucket]
            return self.group_by_name(group_name)

    def group_by_name(self, name: str) -> dict[str, Any]:
        for group in self.data["shard_groups"]:
            if group["name"] == name:
                return group
        raise KeyError(f"unknown shard group: {name}")

    def groups(self) -> list[dict[str, Any]]:
        with self._lock:
            return [group for group in self.data["shard_groups"]]

    def all_primary_uris(self) -> list[str]:
        with self._lock:
            return [group["primary"] for group in self.data["shard_groups"]]

    def writable_uris_for_group(self, group: dict[str, Any]) -> list[tuple[str, bool]]:
        with self._lock:
            uris = [(group["primary"], True)]
            if self.data.get("write_mirror_replicas", True):
                for replica in group.get("replicas", []):
                    if replica.get("state") == REPLICA_ACTIVE:
                        uris.append((replica["uri"], False))
            return uris

    def mark_replica_stale(self, uri: str) -> None:
        with self._lock:
            changed = False
            uri = _normalize_uri(uri)
            for group in self.data["shard_groups"]:
                for replica in group.get("replicas", []):
                    if _normalize_uri(replica["uri"]) == uri and replica.get("state") != REPLICA_STALE:
                        replica["state"] = REPLICA_STALE
                        changed = True
            if changed:
                self.bump_epoch()
                self.save()

    def promote(self, group_name: str, replica_uri: str) -> None:
        with self._lock:
            group = self.group_by_name(group_name)
            replica_uri = _normalize_uri(replica_uri)
            old_primary = group["primary"]
            new_replicas = []
            found = False
            for replica in group.get("replicas", []):
                if _normalize_uri(replica["uri"]) == replica_uri:
                    found = True
                    continue
                new_replicas.append(replica)
            if not found:
                raise ValueError(f"{replica_uri} is not a replica of {group_name}")
            group["primary"] = replica_uri
            group["primary_epoch"] = int(group.get("primary_epoch", 1)) + 1
            new_replicas.append({"uri": old_primary, "state": REPLICA_STALE})
            group["replicas"] = new_replicas
            self.bump_epoch()
            self.save()


class ClusterCoordinator:
    def __init__(
        self,
        state: ClusterState,
        timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        health_interval_secs: float = DEFAULT_HEALTH_INTERVAL_SECS,
        health_failures: int = DEFAULT_HEALTH_FAILURES,
        shard_api_key: str | None = None,
    ):
        self.state = state
        self.timeout_secs = float(timeout_secs)
        self.health_interval_secs = float(health_interval_secs)
        self.health_failures = int(health_failures)
        self.shard_api_key = shard_api_key
        self.client = httpx.Client(timeout=self.timeout_secs)
        self._health_lock = threading.Lock()
        self._failures: dict[str, int] = {}
        self._healthy: dict[str, bool] = {}
        self._rpc_available: dict[str, bool] = {}
        self._stop = threading.Event()
        self._health_thread: threading.Thread | None = None

    def start_health_loop(self) -> None:
        if self._health_thread and self._health_thread.is_alive():
            return
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._health_thread:
            self._health_thread.join(timeout=2)
        self.client.close()

    def _headers(self) -> dict[str, str]:
        if self.shard_api_key:
            return {"Authorization": f"Bearer {self.shard_api_key}"}
        return {}

    def _request(
        self,
        method: str,
        uri: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        content: bytes | None = None,
        content_type: str | None = None,
    ) -> httpx.Response:
        headers = self._headers()
        if content_type:
            headers["Content-Type"] = content_type
        url = f"{_normalize_uri(uri)}{path}"
        return self.client.request(
            method,
            url,
            json=json_body,
            params=params,
            content=content,
            headers=headers,
        )

    def _json_post(self, uri: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self._request("POST", uri, path, json_body=body)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("error") or response.text)
        return payload

    def _rpc_request(
        self,
        uri: str,
        op: int,
        meta: dict[str, Any] | None = None,
        raw: bytes = b"",
    ) -> tuple[dict[str, Any], bytes]:
        uri = _normalize_uri(uri)
        meta = dict(meta or {})
        if self.shard_api_key:
            meta["api_key"] = self.shard_api_key
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        payload = bytes([int(op)]) + struct.pack("<I", len(meta_bytes)) + meta_bytes + raw
        host, port = _derive_rpc_target(uri)

        try:
            with socket.create_connection((host, port), timeout=self.timeout_secs) as sock:
                sock.settimeout(self.timeout_secs)
                sock.sendall(struct.pack("<I", len(payload)) + payload)
                header = self._recv_exact(sock, 4)
                frame_len = struct.unpack("<I", header)[0]
                frame = self._recv_exact(sock, frame_len)
        except Exception:
            self._rpc_available[uri] = False
            raise

        if len(frame) < 5:
            self._rpc_available[uri] = False
            raise RuntimeError("internal RPC response frame is too short")
        status = frame[0]
        meta_len = struct.unpack("<I", frame[1:5])[0]
        if len(frame) < 5 + meta_len:
            raise RuntimeError("internal RPC response metadata length exceeds frame")
        response_meta = json.loads(frame[5:5 + meta_len]) if meta_len else {}
        response_raw = frame[5 + meta_len:]
        if status != 0:
            self._rpc_available[uri] = False
            raise RuntimeError(response_meta.get("error") or "internal RPC request failed")
        self._rpc_available[uri] = True
        return response_meta, response_raw

    @staticmethod
    def _recv_exact(sock: socket.socket, n_bytes: int) -> bytes:
        chunks = []
        remaining = n_bytes
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise RuntimeError("internal RPC connection closed early")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _can_rpc(self, uri: str) -> bool:
        uri = _normalize_uri(uri)
        if self._rpc_available.get(uri) is True:
            return True
        try:
            self._rpc_request(uri, RPC_OP_PING, {})
            return True
        except Exception:
            return False

    def _is_uri_healthy(self, uri: str) -> bool:
        with self._health_lock:
            return self._healthy.get(_normalize_uri(uri), True)

    def _set_health(self, uri: str, ok: bool) -> None:
        uri = _normalize_uri(uri)
        with self._health_lock:
            if ok:
                self._failures[uri] = 0
                self._healthy[uri] = True
            else:
                failures = self._failures.get(uri, 0) + 1
                self._failures[uri] = failures
                if failures >= self.health_failures:
                    self._healthy[uri] = False

    def _probe(self, uri: str) -> bool:
        try:
            response = self._request("GET", uri, "/")
            ok = response.status_code == 200
        except Exception:
            ok = False
        self._set_health(uri, ok)
        return ok

    def refresh_health_once(self) -> None:
        uris = set()
        for group in self.state.groups():
            uris.add(group["primary"])
            for replica in group.get("replicas", []):
                uris.add(replica["uri"])
        for uri in sorted(uris):
            self._probe(uri)

        for group in self.state.groups():
            primary = group["primary"]
            if self._is_uri_healthy(primary):
                continue
            for replica in group.get("replicas", []):
                replica_uri = replica["uri"]
                if replica.get("state") == REPLICA_ACTIVE and self._is_uri_healthy(replica_uri):
                    self.state.promote(group["name"], replica_uri)
                    break

    def _health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_health_once()
            except Exception:
                pass
            self._stop.wait(self.health_interval_secs)

    def read_uri_for_group(self, group: dict[str, Any]) -> str:
        if self._is_uri_healthy(group["primary"]):
            return group["primary"]
        for replica in group.get("replicas", []):
            if replica.get("state") == REPLICA_ACTIVE and self._is_uri_healthy(replica["uri"]):
                return replica["uri"]
        return group["primary"]

    def broadcast_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        primaries_only: bool = False,
        require_primary: bool = True,
    ) -> dict[str, Any]:
        targets: list[tuple[str, bool]] = []
        if primaries_only:
            targets = [(uri, True) for uri in self.state.all_primary_uris()]
        else:
            for group in self.state.groups():
                targets.extend(self.state.writable_uris_for_group(group))
        if not targets:
            return {}

        first_payload: dict[str, Any] | None = None
        primary_errors = []
        with ThreadPoolExecutor(max_workers=min(len(targets), 32)) as pool:
            future_to_target = {
                pool.submit(self._json_post, uri, path, body): (uri, is_primary)
                for uri, is_primary in targets
            }
            for future in as_completed(future_to_target):
                uri, is_primary = future_to_target[future]
                try:
                    payload = future.result()
                    if first_payload is None:
                        first_payload = payload
                except Exception as exc:
                    if is_primary:
                        primary_errors.append(f"{uri}: {exc}")
                    else:
                        self.state.mark_replica_stale(uri)
        if require_primary and primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success()

    def write_group_json(self, group: dict[str, Any], path: str, body: dict[str, Any]) -> dict[str, Any]:
        targets = self.state.writable_uris_for_group(group)
        first_payload: dict[str, Any] | None = None
        primary_errors = []
        with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as pool:
            future_to_target = {
                pool.submit(self._json_post, uri, path, body): (uri, is_primary)
                for uri, is_primary in targets
            }
            for future in as_completed(future_to_target):
                uri, is_primary = future_to_target[future]
                try:
                    payload = future.result()
                    if first_payload is None and is_primary:
                        first_payload = payload
                except Exception as exc:
                    if is_primary:
                        primary_errors.append(f"{uri}: {exc}")
                    else:
                        self.state.mark_replica_stale(uri)
        if primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success()

    def write_group_binary_items(
        self,
        group: dict[str, Any],
        path: str,
        body: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        op = RPC_OP_UPSERT_BINARY_IDS if path == "/upsert" else RPC_OP_BULK_ADD_BINARY_IDS
        vectors = [item["vector"] for item in items]
        raw, n_vectors, dim = _vectors_to_f32_bytes(vectors)
        ids = [int(item["id"]) for item in items]
        id_raw, id_meta = _encode_ids_for_wire(ids)
        meta = {
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "dim": dim,
            "n_vectors": n_vectors,
            **id_meta,
        }
        raw += id_raw
        fields = self._fields_for_binary_items(path, items)
        if fields is not None:
            raw += _encode_fields_binary(fields)
        http_payload = {
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "items": items,
        }

        first_payload: dict[str, Any] | None = None
        primary_errors = []
        for uri, is_primary in self.state.writable_uris_for_group(group):
            try:
                if self._can_rpc(uri):
                    rpc_meta, _raw = self._rpc_request(uri, op, meta, raw)
                    payload = _json_success({
                        "database_name": body["database_name"],
                        "collection_name": body["collection_name"],
                        "ids": rpc_meta.get("ids", ids),
                    })
                else:
                    payload = self._json_post(uri, path, http_payload)
                if is_primary:
                    first_payload = payload
            except Exception as exc:
                try:
                    payload = self._json_post(uri, path, http_payload)
                    if is_primary:
                        first_payload = payload
                except Exception as fallback_exc:
                    if is_primary:
                        primary_errors.append(f"{uri}: {fallback_exc}")
                    else:
                        self.state.mark_replica_stale(uri)
                else:
                    if not is_primary:
                        continue
                    if first_payload is None:
                        primary_errors.append(f"{uri}: {exc}")
        if primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success({"ids": ids})

    def write_group_binary_payload(
        self,
        group: dict[str, Any],
        path: str,
        database_name: str,
        collection_name: str,
        dim: int,
        vector_encoding: str,
        vector_raw: bytes | memoryview,
        ids: list[int],
        fields: list[Any] | None,
    ) -> dict[str, Any]:
        op = RPC_OP_UPSERT_BINARY_IDS if path == "/upsert_binary" else RPC_OP_BULK_ADD_BINARY_IDS
        id_raw, id_meta = _encode_ids_for_wire(ids)
        raw = bytes(vector_raw) + id_raw
        if fields is not None:
            raw += _encode_fields_binary(fields)
        meta = {
            "database_name": database_name,
            "collection_name": collection_name,
            "dim": int(dim),
            "n_vectors": len(ids),
            "vector_encoding": _normalize_vector_encoding(vector_encoding),
            **id_meta,
        }
        params = dict(meta)
        params["return_ids"] = "false"

        first_payload: dict[str, Any] | None = None
        primary_errors = []
        for uri, is_primary in self.state.writable_uris_for_group(group):
            try:
                if self._can_rpc(uri):
                    rpc_meta, _raw = self._rpc_request(uri, op, meta, raw)
                    payload = _json_success({
                        "database_name": database_name,
                        "collection_name": collection_name,
                        "ids": rpc_meta.get("ids", ids),
                    })
                else:
                    response = self._request(
                        "POST",
                        uri,
                        path,
                        params=params,
                        content=raw,
                        content_type="application/octet-stream",
                    )
                    if response.status_code != 200:
                        raise RuntimeError(response.text)
                    payload = response.json()
                if is_primary:
                    first_payload = payload
            except Exception as exc:
                if is_primary:
                    primary_errors.append(f"{uri}: {exc}")
                else:
                    self.state.mark_replica_stale(uri)
        if primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success({"ids": ids})

    @staticmethod
    def _fields_for_binary_items(path: str, items: list[dict[str, Any]]) -> list[Any] | None:
        if path == "/upsert":
            if not any("field" in item for item in items):
                return None
            return [item["field"] if "field" in item else None for item in items]

        if not any(item.get("field") for item in items):
            return None
        return [item.get("field") or {} for item in items]

    def write_group_ids(self, group: dict[str, Any], path: str, body: dict[str, Any]) -> dict[str, Any]:
        primary_errors = []
        first_payload: dict[str, Any] | None = None
        for uri, is_primary in self.state.writable_uris_for_group(group):
            try:
                payload = self._json_post(uri, path, body)
                if is_primary:
                    first_payload = payload
            except Exception:
                try:
                    payload = self._json_post(uri, path, body)
                    if is_primary:
                        first_payload = payload
                except Exception as fallback_exc:
                    if is_primary:
                        primary_errors.append(f"{uri}: {fallback_exc}")
                    else:
                        self.state.mark_replica_stale(uri)
        if primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success()

    def create_database(self, body: dict[str, Any]) -> dict[str, Any]:
        name = body["database_name"]
        if body.get("drop_if_exists"):
            self.drop_database({"database_name": name})
        self.broadcast_json("/create_database", body)
        self.state.ensure_database(name)
        return _json_success({"database_name": name})

    def drop_database(self, body: dict[str, Any]) -> dict[str, Any]:
        name = body["database_name"]
        self.broadcast_json("/delete_database", body, require_primary=False)
        self.state.remove_database(name)
        return _json_success({"database_name": name})

    def require_collection(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        self.broadcast_json("/required_collection", body)
        dim = body.get("dim")
        coll = self.state.upsert_collection(
            db_name,
            coll_name,
            int(dim or 0),
            body.get("description"),
            body.get("dtypes") or "float32",
            bool(body.get("drop_if_exists", False)),
        )
        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "config": coll,
        })

    def drop_collection(self, body: dict[str, Any]) -> dict[str, Any]:
        self.broadcast_json("/drop_collection", body, require_primary=False)
        self.state.drop_collection(body["database_name"], body["collection_name"])
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
        })

    def collection_config_response(self, body: dict[str, Any]) -> dict[str, Any]:
        coll = self.state.get_collection(body["database_name"], body["collection_name"])
        if not coll:
            raise KeyError(f"Collection '{body['collection_name']}' not found in config")
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "config": {
                "dim": coll.get("dim"),
                "chunk_size": coll.get("chunk_size", 100000),
                "description": coll.get("description"),
                "dtypes": coll.get("dtypes", "float32"),
            },
        })

    def collection_exists_response(self, body: dict[str, Any]) -> dict[str, Any]:
        exists = self.state.get_collection(body["database_name"], body["collection_name"]) is not None
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "exists": exists,
        })

    def _route_items(self, body: dict[str, Any], endpoint: str) -> list[int]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        items = [dict(item) for item in body.get("items", [])]
        missing = [idx for idx, item in enumerate(items) if item.get("id") is None]
        allocated = self.state.allocate_ids(db_name, coll_name, len(missing)) if missing else []
        for idx, item_id in zip(missing, allocated):
            items[idx]["id"] = item_id

        grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for item in items:
            group = self.state.group_for_id(db_name, coll_name, int(item["id"]))
            grouped.setdefault(group["name"], (group, []))[1].append(item)

        with ThreadPoolExecutor(max_workers=min(len(grouped), 32) or 1) as pool:
            futures = []
            for group, group_items in grouped.values():
                payload = {
                    "database_name": db_name,
                    "collection_name": coll_name,
                    "items": group_items,
                }
                futures.append(pool.submit(
                    self.write_group_binary_items,
                    group,
                    endpoint,
                    payload,
                    group_items,
                ))
            for future in as_completed(futures):
                future.result()
        return [int(item["id"]) for item in items]

    def add_records(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        ids = list(body.get("ids") or [])
        vectors = list(body.get("vectors") or [])
        fields = body.get("fields")
        if not ids:
            raise ValueError("ids cannot be empty")
        if len(ids) != len(vectors):
            raise ValueError("ids length must match vectors length")
        if fields is not None and len(fields) != len(ids):
            raise ValueError("fields length must match ids length")
        if vectors:
            self.state.update_collection_dim_if_unset(db_name, coll_name, len(vectors[0]))

        grouped: dict[str, dict[str, Any]] = {}
        for idx, item_id in enumerate(ids):
            group = self.state.group_for_external_id(db_name, coll_name, item_id)
            bucket = grouped.setdefault(
                group["name"],
                {"group": group, "ids": [], "vectors": [], "fields": [] if fields is not None else None},
            )
            bucket["ids"].append(item_id)
            bucket["vectors"].append(vectors[idx])
            if fields is not None:
                bucket["fields"].append(fields[idx])

        with ThreadPoolExecutor(max_workers=min(len(grouped), 32) or 1) as pool:
            futures = []
            for bucket in grouped.values():
                payload = {
                    "database_name": db_name,
                    "collection_name": coll_name,
                    "ids": bucket["ids"],
                    "vectors": bucket["vectors"],
                    "fields": bucket["fields"],
                }
                futures.append(pool.submit(self.write_group_json, bucket["group"], "/add", payload))
            for future in as_completed(futures):
                future.result()

        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "ids": ids,
        })

    def upsert_records(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        ids = list(body.get("ids") or [])
        vectors = list(body.get("vectors") or [])
        fields = body.get("fields")
        if not ids:
            raise ValueError("ids cannot be empty")
        if len(ids) != len(vectors):
            raise ValueError("ids length must match vectors length")
        if fields is not None and len(fields) != len(ids):
            raise ValueError("fields length must match ids length")
        if vectors:
            self.state.update_collection_dim_if_unset(db_name, coll_name, len(vectors[0]))

        grouped: dict[str, dict[str, Any]] = {}
        for idx, item_id in enumerate(ids):
            group = self.state.group_for_external_id(db_name, coll_name, item_id)
            bucket = grouped.setdefault(
                group["name"],
                {"group": group, "ids": [], "vectors": [], "fields": [] if fields is not None else None},
            )
            bucket["ids"].append(item_id)
            bucket["vectors"].append(vectors[idx])
            if fields is not None:
                bucket["fields"].append(fields[idx])

        with ThreadPoolExecutor(max_workers=min(len(grouped), 32) or 1) as pool:
            futures = []
            for bucket in grouped.values():
                payload = {
                    "database_name": db_name,
                    "collection_name": coll_name,
                    "ids": bucket["ids"],
                    "vectors": bucket["vectors"],
                    "fields": bucket["fields"],
                }
                futures.append(pool.submit(self.write_group_json, bucket["group"], "/upsert", payload))
            for future in as_completed(futures):
                future.result()

        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "ids": ids,
        })

    def route_binary_items(self, params: dict[str, Any], body: bytes, path: str) -> dict[str, Any]:
        db_name = str(params["database_name"])
        coll_name = str(params["collection_name"])
        dim = int(params["dim"])
        n_vectors = int(params["n_vectors"])
        vector_encoding = _normalize_vector_encoding(params.get("vector_encoding"))
        self.state.update_collection_dim_if_unset(db_name, coll_name, dim)
        vector_raw, ids, fields, _ = _split_binary_items_payload(
            body,
            n_vectors,
            dim,
            vector_encoding,
            params.get("ids_encoding"),
            params.get("ids_start"),
        )
        stride = dim * _vector_wire_width(vector_encoding)
        grouped: dict[str, dict[str, Any]] = {}
        for idx, item_id in enumerate(ids):
            group = self.state.group_for_id(db_name, coll_name, int(item_id))
            bucket = grouped.setdefault(
                group["name"],
                {"group": group, "parts": [], "ids": [], "fields": [] if fields is not None else None},
            )
            start = idx * stride
            bucket["parts"].append(vector_raw[start:start + stride])
            bucket["ids"].append(int(item_id))
            if fields is not None:
                bucket["fields"].append(fields[idx])

        with ThreadPoolExecutor(max_workers=min(len(grouped), 32) or 1) as pool:
            futures = []
            for bucket in grouped.values():
                futures.append(pool.submit(
                    self.write_group_binary_payload,
                    bucket["group"],
                    path,
                    db_name,
                    coll_name,
                    dim,
                    vector_encoding,
                    b"".join(bucket["parts"]),
                    bucket["ids"],
                    bucket["fields"],
                ))
            for future in as_completed(futures):
                future.result()

        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            **({"ids": ids} if str(params.get("return_ids", "true")).lower() == "true" else {"n_vectors": n_vectors}),
        })

    def route_ids_write(self, body: dict[str, Any], endpoint: str) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        ids = list(body.get("ids", []))
        grouped: dict[str, tuple[dict[str, Any], list[Any]]] = {}
        for item_id in ids:
            group = self.state.group_for_external_id(db_name, coll_name, item_id)
            grouped.setdefault(group["name"], (group, []))[1].append(item_id)
        with ThreadPoolExecutor(max_workers=min(len(grouped), 32) or 1) as pool:
            futures = []
            for group, group_ids in grouped.values():
                futures.append(pool.submit(self.write_group_ids, group, endpoint, {
                    "database_name": db_name,
                    "collection_name": coll_name,
                    "ids": group_ids,
                }))
            for future in as_completed(futures):
                future.result()
        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "status": "ok",
        })

    def search_json(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        coll = self.state.get_collection(db_name, coll_name)
        if not coll:
            raise KeyError(f"Collection '{coll_name}' not found")
        k = int(body.get("k") or 10)
        return_fields = bool(body.get("return_fields", False))
        ascending = _is_ascending_index(coll.get("index_mode"))
        results = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            future_to_group = {}
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                future_to_group[pool.submit(self._json_post, uri, "/search", body)] = group
            for future in as_completed(future_to_group):
                payload = future.result()
                items = payload.get("params", {}).get("items", {})
                results.append((
                    list(items.get("ids", [])),
                    [float(x) for x in items.get("scores", [])],
                    items.get("fields", []) or [],
                ))
        ids, scores, fields = _merge_pairs(results, k, ascending, return_fields)
        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "items": {
                "k": k,
                "ids": ids,
                "scores": scores,
                "fields": fields,
                "vector_field": body.get("vector_field", "default"),
            },
        })

    def bm25_search_json(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        k = int(body.get("k") or 10)
        return_fields = bool(body.get("return_fields", False))
        results = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(self._json_post, uri, "/bm25_search", body))
            for future in as_completed(futures):
                payload = future.result()
                items = payload.get("params", {}).get("items", {})
                results.append((
                    list(items.get("ids", [])),
                    [float(x) for x in items.get("scores", [])],
                    items.get("fields", []) or [],
                ))
        ids, scores, fields = _merge_pairs(results, k, False, return_fields)
        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "items": {
                "k": k,
                "ids": ids,
                "scores": scores,
                "fields": fields,
                "index": "BM25-SCAN",
            },
        })

    def search_binary(
        self,
        params: dict[str, Any],
        body: bytes,
    ) -> bytes:
        db_name = str(params["database_name"])
        coll_name = str(params["collection_name"])
        coll = self.state.get_collection(db_name, coll_name)
        if not coll:
            raise KeyError(f"Collection '{coll_name}' not found")
        k = int(params.get("k") or 10)
        return_fields = str(params.get("return_fields", "false")).lower() == "true"
        ascending = _is_ascending_index(coll.get("index_mode"))
        results = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(
                    self._binary_call,
                    uri,
                    RPC_OP_SEARCH,
                    "/search_binary",
                    params,
                    body,
                ))
            for future in as_completed(futures):
                raw = future.result()
                ids, distances, fields, _ = _split_search_binary(raw)
                results.append((ids, distances, fields))
        ids, distances, fields = _merge_pairs(results, k, ascending, return_fields)
        return _encode_search_binary(ids, distances, fields)

    def _binary_call(
        self,
        uri: str,
        rpc_op: int,
        http_path: str,
        params: dict[str, Any],
        body: bytes,
    ) -> bytes:
        if self._can_rpc(uri):
            try:
                _meta, raw = self._rpc_request(uri, rpc_op, params, body)
                return raw
            except Exception:
                pass
        response = self._request(
            "POST",
            uri,
            http_path,
            params=params,
            content=body,
            content_type="application/octet-stream",
        )
        if response.status_code != 200:
            raise RuntimeError(response.text)
        return response.content

    def batch_search_binary(self, params: dict[str, Any], body: bytes) -> bytes:
        db_name = str(params["database_name"])
        coll_name = str(params["collection_name"])
        coll = self.state.get_collection(db_name, coll_name)
        if not coll:
            raise KeyError(f"Collection '{coll_name}' not found")
        k = int(params.get("k") or 10)
        n_queries = int(params.get("n_queries") or 0)
        return_fields = str(params.get("return_fields", "false")).lower() == "true"
        ascending = _is_ascending_index(coll.get("index_mode"))
        per_query: list[list[tuple[list[int], list[float], list[dict[str, Any]]]]] = [
            [] for _ in range(n_queries)
        ]

        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(
                    self._binary_call,
                    uri,
                    RPC_OP_BATCH_SEARCH,
                    "/batch_search_binary",
                    params,
                    body,
                ))
            for future in as_completed(futures):
                buf = future.result()
                offset = 0
                count = int.from_bytes(buf[offset:offset + 4], "little")
                offset += 4
                if count != n_queries:
                    raise RuntimeError(f"batch_search_binary returned {count} queries, expected {n_queries}")
                for query_idx in range(count):
                    ids, distances, fields, offset = _split_search_binary(buf, offset)
                    per_query[query_idx].append((ids, distances, fields))

        parts = [n_queries.to_bytes(4, "little")]
        for query_results in per_query:
            ids, distances, fields = _merge_pairs(query_results, k, ascending, return_fields)
            parts.append(_encode_search_binary(ids, distances, fields))
        return b"".join(parts)

    def head_tail_binary(self, endpoint: str, params: dict[str, Any]) -> bytes:
        n = int(params.get("n") or 5)
        rows: list[tuple[list[float], int, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(
                    self._request,
                    "GET",
                    uri,
                    endpoint,
                    params=params,
                ))
            for future in as_completed(futures):
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(response.text)
                vectors, ids, fields, _dim, _offset = _split_vectors_binary(response.content)
                for idx, (vector, item_id) in enumerate(zip(vectors, ids)):
                    field = fields[idx] if idx < len(fields) else {}
                    rows.append((vector, item_id, field))

        rows.sort(key=lambda item: item[1], reverse=endpoint == "/tail_binary")
        rows = rows[:n]
        if endpoint == "/tail_binary":
            rows.reverse()
        vectors = [row[0] for row in rows]
        ids = [row[1] for row in rows]
        fields = [row[2] for row in rows]
        return _encode_vectors_binary(vectors, ids, fields)

    def head_tail_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        n = int(body.get("n") or 5)
        result_key = "tail" if endpoint == "/tail" else "head"
        rows: list[tuple[list[float], Any, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(self._json_post, uri, endpoint, body))
            for future in as_completed(futures):
                payload = future.result()
                result = payload.get("params", {}).get(result_key, [[], [], []])
                vectors, ids, fields = result[0] or [], result[1] or [], result[2] or []
                for idx, (vector, item_id) in enumerate(zip(vectors, ids)):
                    field = fields[idx] if idx < len(fields) else {}
                    rows.append((vector, item_id, field))

        rows.sort(key=lambda item: (type(item[1]).__name__, str(item[1])), reverse=endpoint == "/tail")
        rows = rows[:n]
        if endpoint == "/tail":
            rows.reverse()
        vectors = [row[0] for row in rows]
        ids = [row[1] for row in rows]
        fields = [row[2] for row in rows]
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            result_key: [vectors, ids, fields],
        })

    def query_all_json(self, body: dict[str, Any], endpoint: str) -> dict[str, Any]:
        results = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            filter_ids = body.get("filter_ids") or []
            if filter_ids and body.get("where") is None:
                grouped: dict[str, tuple[dict[str, Any], list[Any]]] = {}
                for item_id in filter_ids:
                    group = self.state.group_for_external_id(
                        body["database_name"],
                        body["collection_name"],
                        item_id,
                    )
                    grouped.setdefault(group["name"], (group, []))[1].append(item_id)
                for group, group_ids in grouped.values():
                    uri = self.read_uri_for_group(group)
                    payload = {**body, "filter_ids": group_ids}
                    futures.append(pool.submit(self._json_post, uri, endpoint, payload))
            else:
                for group in self.state.groups():
                    uri = self.read_uri_for_group(group)
                    futures.append(pool.submit(self._json_post, uri, endpoint, body))
            for future in as_completed(futures):
                payload = future.result()
                results.append(payload.get("params", {}).get("result"))

        if endpoint == "/query_vectors":
            vectors: list[Any] = []
            ids: list[Any] = []
            fields: list[Any] = []
            for result in results:
                if not result:
                    continue
                vectors.extend(result[0] or [])
                ids.extend(result[1] or [])
                fields.extend(result[2] or [])
            result_data: Any = [vectors, ids, fields]
        else:
            if body.get("return_ids_only"):
                merged_ids = []
                for result in results:
                    merged_ids.extend(result or [])
                result_data = merged_ids
            else:
                records = []
                for result in results:
                    records.extend(result or [])
                result_data = records
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "result": result_data,
        })

    def show_collections(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body.get("database_name")
        names = []
        prefix = f"{db_name}/"
        for key in self.state.data.get("collections", {}):
            if key.startswith(prefix):
                names.append(key[len(prefix):])
        return _json_success({"database_name": db_name, "collections": sorted(names)})

    def show_collections_details(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body.get("database_name")
        details = {}
        prefix = f"{db_name}/"
        for key, coll in self.state.data.get("collections", {}).items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            details[name] = {
                "name": name,
                "dim": coll.get("dim"),
                "dtypes": coll.get("dtypes", "float32"),
                "description": coll.get("description"),
                "index_mode": coll.get("index_mode"),
            }
        return _json_success({"database_name": db_name, "collections": details})

    def is_id_exists(self, body: dict[str, Any]) -> dict[str, Any]:
        group = self.state.group_for_external_id(
            body["database_name"],
            body["collection_name"],
            body["id"],
        )
        uri = self.read_uri_for_group(group)
        payload = self._json_post(uri, "/is_id_exists", body)
        exists = bool(payload.get("params", {}).get("is_id_exists", False))
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "is_id_exists": exists,
        })

    def search_range(self, body: dict[str, Any]) -> dict[str, Any]:
        coll = self.state.get_collection(body["database_name"], body["collection_name"]) or {}
        max_results = int(body.get("max_results") or 1000)
        ascending = _is_ascending_index(coll.get("index_mode"))
        results = []
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(self._json_post, uri, "/search_range", body))
            for future in as_completed(futures):
                payload = future.result()
                result = payload.get("params", {}).get("result", {})
                results.append((
                    list(result.get("ids", [])),
                    [float(x) for x in result.get("distances", [])],
                    [],
                ))
        ids, distances, _ = _merge_pairs(results, max_results, ascending, False)
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "result": {"ids": ids, "distances": distances},
        })

    def list_deleted_ids(self, body: dict[str, Any]) -> dict[str, Any]:
        ids = set()
        with ThreadPoolExecutor(max_workers=min(len(self.state.groups()), 32)) as pool:
            futures = []
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(pool.submit(self._json_post, uri, "/list_deleted_ids", body))
            for future in as_completed(futures):
                payload = future.result()
                ids.update(payload.get("params", {}).get("ids", []))
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "ids": sorted(ids, key=lambda value: (type(value).__name__, str(value))),
        })


class ClusterRequestHandler(BaseHTTPRequestHandler):
    coordinator: ClusterCoordinator

    server_version = "LynseDBCluster/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _read_json(self) -> dict[str, Any]:
        body = self._read_body()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, message: str, status: int = 500) -> None:
        self._send_json({"status": "error", "error": message}, status=status)

    def _send_binary(self, data: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_json(_json_success({"message": "LynseDB cluster coordinator"}))
            elif parsed.path == "/list_databases":
                self._send_json(_json_success({"databases": self.coordinator.state.databases()}))
            elif parsed.path == "/cluster_info":
                self._send_json(_json_success(self.coordinator.state.data))
            elif parsed.path in {"/head_binary", "/tail_binary"}:
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                self._send_binary(self.coordinator.head_tail_binary(parsed.path, params))
            else:
                self._send_error_json(f"Unsupported cluster endpoint: {parsed.path}", status=404)
        except Exception as exc:
            self._send_error_json(str(exc))

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/search_binary":
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                buf = self.coordinator.search_binary(params, self._read_body())
                self._send_binary(buf)
                return
            if path == "/batch_search_binary":
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                buf = self.coordinator.batch_search_binary(params, self._read_body())
                self._send_binary(buf)
                return
            if path in {"/add_binary_ids", "/upsert_binary"}:
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                payload = self.coordinator.route_binary_items(params, self._read_body(), path)
                self._send_json(payload)
                return

            body = self._read_json()
            if path == "/create_database":
                payload = self.coordinator.create_database(body)
            elif path in {"/delete_database", "/drop_database"}:
                payload = self.coordinator.drop_database(body)
            elif path == "/database_exists":
                payload = _json_success({
                    "database_name": body["database_name"],
                    "exists": body["database_name"] in self.coordinator.state.databases(),
                })
            elif path == "/show_collections":
                payload = self.coordinator.show_collections(body)
            elif path == "/show_collections_details":
                payload = self.coordinator.show_collections_details(body)
            elif path == "/required_collection":
                payload = self.coordinator.require_collection(body)
            elif path == "/drop_collection":
                payload = self.coordinator.drop_collection(body)
            elif path == "/is_collection_exists":
                payload = self.coordinator.collection_exists_response(body)
            elif path == "/get_collection_config":
                payload = self.coordinator.collection_config_response(body)
            elif path == "/is_id_exists":
                payload = self.coordinator.is_id_exists(body)
            elif path == "/add":
                payload = self.coordinator.add_records(body)
            elif path == "/upsert":
                payload = self.coordinator.upsert_records(body)
            elif path == "/delete":
                payload = self.coordinator.route_ids_write(body, "/delete")
            elif path == "/restore":
                payload = self.coordinator.route_ids_write(body, "/restore")
            elif path == "/search":
                payload = self.coordinator.search_json(body)
            elif path == "/bm25_search":
                payload = self.coordinator.bm25_search_json(body)
            elif path == "/search_range":
                payload = self.coordinator.search_range(body)
            elif path == "/list_deleted_ids":
                payload = self.coordinator.list_deleted_ids(body)
            elif path in {"/head", "/tail"}:
                payload = self.coordinator.head_tail_json(path, body)
            elif path in {"/query", "/query_vectors"}:
                payload = self.coordinator.query_all_json(body, path)
            elif path in {"/commit", "/flush", "/checkpoint", "/close_collection"}:
                payload = self.coordinator.broadcast_json(path, body)
            elif path == "/build_index":
                payload = self.coordinator.broadcast_json(path, body)
                self.coordinator.state.update_collection_index(
                    body["database_name"],
                    body["collection_name"],
                    body.get("index_mode") or "FLAT",
                )
            elif path == "/remove_index":
                payload = self.coordinator.broadcast_json(path, body)
                self.coordinator.state.update_collection_index(
                    body["database_name"],
                    body["collection_name"],
                    None,
                )
            elif path in {
                "/collection_shape",
                "/max_id",
                "/index_mode",
                "/list_fields",
                "/read_by_only_id",
            }:
                # These endpoints are safe to answer from all shards and merge only
                # where needed. For now, use shard fan-out for id reads and query-like
                # endpoints; shape/max are computed locally enough for client checks.
                payload = self._fallback_read(path, body)
            else:
                self._send_error_json(f"Unsupported cluster endpoint: {path}", status=404)
                return
            self._send_json(payload)
        except KeyError as exc:
            self._send_error_json(str(exc), status=400)
        except Exception as exc:
            self._send_error_json(str(exc))

    def _fallback_read(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        coord = self.coordinator
        if path == "/collection_shape":
            total = 0
            dim = 0
            for group in coord.state.groups():
                uri = coord.read_uri_for_group(group)
                payload = coord._json_post(uri, path, body)
                shape = payload.get("params", {}).get("shape", [0, 0])
                total += int(shape[0])
                dim = int(shape[1] or dim)
            return _json_success({
                "database_name": body["database_name"],
                "collection_name": body["collection_name"],
                "shape": [total, dim],
            })
        if path == "/max_id":
            max_id = -1
            for group in coord.state.groups():
                uri = coord.read_uri_for_group(group)
                payload = coord._json_post(uri, path, body)
                max_id = max(max_id, int(payload.get("params", {}).get("max_id", -1)))
            return _json_success({
                "database_name": body["database_name"],
                "collection_name": body["collection_name"],
                "max_id": max_id,
            })
        if path == "/index_mode":
            coll = coord.state.get_collection(body["database_name"], body["collection_name"]) or {}
            return _json_success({
                "database_name": body["database_name"],
                "collection_name": body["collection_name"],
                "index_mode": coll.get("index_mode"),
            })
        if path == "/list_fields":
            fields = set()
            for group in coord.state.groups():
                uri = coord.read_uri_for_group(group)
                payload = coord._json_post(uri, path, body)
                fields.update(payload.get("params", {}).get("fields", []))
            return _json_success({
                "database_name": body["database_name"],
                "collection_name": body["collection_name"],
                "fields": sorted(fields),
            })
        if path == "/read_by_only_id":
            raw_id = body.get("id")
            ids = raw_id if isinstance(raw_id, list) else [raw_id]
            by_group: dict[str, tuple[dict[str, Any], list[Any]]] = {}
            for item_id in ids:
                group = coord.state.group_for_external_id(body["database_name"], body["collection_name"], item_id)
                by_group.setdefault(group["name"], (group, []))[1].append(item_id)
            vectors = []
            returned_ids = []
            fields = []
            for group, group_ids in by_group.values():
                uri = coord.read_uri_for_group(group)
                payload = coord._json_post(uri, path, {**body, "id": group_ids})
                item = payload.get("params", {}).get("item", [[], [], []])
                vectors.extend(item[0] or [])
                returned_ids.extend(item[1] or [])
                fields.extend(item[2] or [])
            return _json_success({
                "database_name": body["database_name"],
                "collection_name": body["collection_name"],
                "item": [vectors, returned_ids, fields],
            })
        raise ValueError(f"Unsupported fallback endpoint: {path}")


def load_cluster_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_coordinator(
    host: str,
    port: int,
    *,
    cluster_config: str | None = None,
    cluster_state: str | None = None,
    shard_api_key: str | None = None,
    request_timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
    health_interval_secs: float = DEFAULT_HEALTH_INTERVAL_SECS,
    health_failures: int = DEFAULT_HEALTH_FAILURES,
) -> None:
    config = load_cluster_config(cluster_config)
    state_path = Path(
        cluster_state
        or config.get("state_path")
        or os.environ.get("LYNSE_CLUSTER_STATE")
        or "cluster_state.json"
    )
    state = ClusterState(state_path, seed_config=config)
    coordinator = ClusterCoordinator(
        state,
        timeout_secs=request_timeout_secs,
        health_interval_secs=health_interval_secs,
        health_failures=health_failures,
        shard_api_key=shard_api_key or config.get("shard_api_key"),
    )
    coordinator.start_health_loop()

    class Handler(ClusterRequestHandler):
        pass

    Handler.coordinator = coordinator
    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    try:
        httpd.serve_forever()
    finally:
        coordinator.stop()
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LynseDB lightweight cluster coordinator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7637)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--cluster-state")
    parser.add_argument("--shard-api-key")
    parser.add_argument("--request-timeout-secs", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECS)
    parser.add_argument("--health-interval-secs", type=float, default=DEFAULT_HEALTH_INTERVAL_SECS)
    parser.add_argument("--health-failures", type=int, default=DEFAULT_HEALTH_FAILURES)
    args = parser.parse_args(argv)
    run_coordinator(
        args.host,
        args.port,
        cluster_config=args.cluster_config,
        cluster_state=args.cluster_state,
        shard_api_key=args.shard_api_key,
        request_timeout_secs=args.request_timeout_secs,
        health_interval_secs=args.health_interval_secs,
        health_failures=args.health_failures,
    )


if __name__ == "__main__":
    main()
