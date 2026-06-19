from __future__ import annotations

"""Small LynseDB cluster coordinator.

This module intentionally keeps clustering modest: one active coordinator
process owns metadata, shards are ordinary LynseDB HTTP servers, and replicas
are maintained by coordinator-side mirroring. It avoids external services while
giving the common happy path automatic shard failover.
"""

import argparse
import array
import asyncio
import heapq
import hashlib
import importlib
import json
import os
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .utils.poster import RustRemoteSession

if not getattr(ThreadingHTTPServer.serve_forever, "_lynse_fast_shutdown", False):
    _threading_http_serve_forever = ThreadingHTTPServer.serve_forever

    def _lynse_fast_serve_forever(self, poll_interval: float = 0.05):
        return _threading_http_serve_forever(self, poll_interval=poll_interval)

    _lynse_fast_serve_forever._lynse_fast_shutdown = True
    ThreadingHTTPServer.serve_forever = _lynse_fast_serve_forever


DEFAULT_BUCKET_COUNT = 4096
DEFAULT_HEALTH_INTERVAL_SECS = 1.0
DEFAULT_HEALTH_FAILURES = 3
DEFAULT_REQUEST_TIMEOUT_SECS = 30.0
DEFAULT_COORDINATOR_LEASE_SECS = 5.0
REPLICA_ACTIVE = "active"
REPLICA_STALE = "stale"

RPC_OP_PING = 1
RPC_OP_SEARCH = 2
RPC_OP_BATCH_SEARCH = 3
RPC_OP_BULK_ADD_BINARY_IDS = 4
RPC_OP_UPSERT_BINARY_IDS = 5
RPC_OP_DELETE_ITEMS = 6
RPC_OP_RESTORE_ITEMS = 7
RPC_OP_COLLECTION_CONTROL = 8
RPC_OP_METADATA_GET = 9
RPC_OP_METADATA_CAS = 10
FIELDS_BINARY_MAGIC = b"LDBF1"
CLUSTER_STATE_METADATA_KEY = "cluster_state"
COORDINATOR_LEASE_METADATA_KEY = "coordinator_lease"
METADATA_RECORD_FORMAT = "lynsedb-metadata-record"


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


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(value)


def _default_coordinator_uri(host: str, port: int) -> str:
    host = str(host or "127.0.0.1")
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(port)}"


def _lease_record_is_expired(record: dict[str, Any], now: float | None = None) -> bool:
    if not record.get("leader_id") or not record.get("leader_uri"):
        return True
    return float(record.get("expires_at", 0.0)) <= (time.time() if now is None else now)


def _metadata_value_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metadata_record(key: str, logical_version: int, value: Any) -> dict[str, Any]:
    return {
        "format": METADATA_RECORD_FORMAT,
        "key": key,
        "version": 1,
        "logical_version": int(logical_version),
        "value_hash": _metadata_value_hash(value),
        "value": value,
    }


def _unwrap_metadata_record(key: str, version: int, value: Any | None) -> tuple[int, Any | None, str, bool]:
    if value is None:
        return 0, None, "", True
    if isinstance(value, dict) and value.get("format") == METADATA_RECORD_FORMAT:
        if value.get("key") not in {None, key}:
            raise ValueError(f"metadata record key mismatch: expected {key}, got {value.get('key')}")
        inner = value.get("value")
        logical_version = int(value.get("logical_version", version) or 0)
        value_hash = str(value.get("value_hash") or _metadata_value_hash(inner))
        return logical_version, inner, value_hash, True
    return int(version), value, _metadata_value_hash(value), False


def _hash_u64(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _hash_u64_prefixed(prefix_hasher, value: int) -> int:
    hasher = prefix_hasher.copy()
    hasher.update(str(int(value)).encode("utf-8"))
    digest = hasher.digest()
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
    upper = (index_mode or "FLAT-IP").upper()
    return any(token in upper for token in ("L2", "COS", "HAMMING", "JACCARD"))


def _shard_artifact_path(base_path: Any, group_name: str) -> str:
    value = str(base_path)
    safe_group = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(group_name)
    ) or "shard"
    slash = max(value.rfind("/"), value.rfind("\\"))
    prefix = value[:slash + 1] if slash >= 0 else ""
    name = value[slash + 1:] if slash >= 0 else value
    dot = name.rfind(".")
    if dot > 0:
        return f"{prefix}{name[:dot]}.{safe_group}{name[dot:]}"
    return f"{value}.{safe_group}"


def _split_search_binary(buf: bytes, offset: int = 0):
    n = struct.unpack_from("<I", buf, offset)[0]
    offset += 4
    ids = list(struct.unpack_from(f"<{n}Q", buf, offset)) if n else []
    offset += n * 8
    distances = list(struct.unpack_from(f"<{n}f", buf, offset)) if n else []
    offset += n * 4
    fields_len = struct.unpack_from("<I", buf, offset)[0]
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
    n = len(ids)
    parts = [struct.pack("<I", n)]
    if n:
        parts.append(struct.pack(f"<{n}Q", *(int(item_id) for item_id in ids)))
        parts.append(struct.pack(f"<{n}f", *(float(distance) for distance in distances)))
    parts.append(struct.pack("<I", len(fields_json)))
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

    raw_ids = array.array("Q", ids)
    if sys.byteorder != "little":
        raw_ids.byteswap()
    raw = raw_ids.tobytes()
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
        ids = list(struct.unpack_from(f"<{int(n_vectors)}Q", view, offset)) if int(n_vectors) else []
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
    if k <= 0:
        return [], [], []
    if not return_fields:
        merged: list[tuple[Any, float]] = []
        for ids, scores, _fields in results:
            merged.extend((item_id, float(score)) for item_id, score in zip(ids, scores))
        if not merged:
            return [], [], []
        key = lambda item: item[1]
        if len(merged) <= max(k * 4, 64):
            top = sorted(merged, key=key, reverse=not ascending)[:k]
        elif len(merged) > k:
            top = heapq.nsmallest(k, merged, key=key) if ascending else heapq.nlargest(k, merged, key=key)
        else:
            top = sorted(merged, key=key, reverse=not ascending)
        return [item[0] for item in top], [item[1] for item in top], []

    merged: list[tuple[Any, float, dict[str, Any] | None]] = []
    for ids, scores, fields in results:
        for idx, (item_id, score) in enumerate(zip(ids, scores)):
            field = fields[idx] if return_fields and idx < len(fields) else None
            merged.append((item_id, float(score), field))
    if not merged:
        return [], [], []
    if len(merged) <= max(k * 4, 64):
        top = sorted(merged, key=lambda item: item[1], reverse=not ascending)[:k]
    elif len(merged) > k:
        if ascending:
            top = heapq.nsmallest(k, merged, key=lambda item: item[1])
        else:
            top = heapq.nlargest(k, merged, key=lambda item: item[1])
    else:
        top = sorted(merged, key=lambda item: item[1], reverse=not ascending)
    ids = [item[0] for item in top]
    distances = [item[1] for item in top]
    out_fields = [item[2] or {} for item in top] if return_fields else []
    return ids, distances, out_fields


class MetadataConflict(RuntimeError):
    pass


class MetadataStore:
    cache_path: Path

    def get(self, key: str) -> tuple[int, Any | None]:
        raise NotImplementedError

    def cas(self, key: str, expected_version: int, value: Any) -> int:
        raise NotImplementedError

    def metadata_status(self) -> dict[str, Any]:
        return {"mode": "local"}


class LocalMetadataStore(MetadataStore):
    def __init__(self, state_path: Path):
        self.cache_path = Path(state_path)

    def get(self, key: str) -> tuple[int, Any | None]:
        if key != CLUSTER_STATE_METADATA_KEY:
            raise KeyError(f"unsupported local metadata key: {key}")
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                value = json.load(f)
        except FileNotFoundError:
            return 0, None
        return int(value.get("meta_epoch", 0) or 0), value

    def cas(self, key: str, expected_version: int, value: Any) -> int:
        if key != CLUSTER_STATE_METADATA_KEY:
            raise KeyError(f"unsupported local metadata key: {key}")
        current_version, current = self.get(key)
        if current is not None and current_version != int(expected_version):
            raise MetadataConflict(
                f"metadata version conflict for {key}: expected {expected_version}, got {current_version}"
            )
        if current is None and int(expected_version) != 0:
            raise MetadataConflict(
                f"metadata version conflict for {key}: expected {expected_version}, got 0"
            )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.cache_path)
        return int(value.get("meta_epoch", 0) or 0)

    def metadata_status(self) -> dict[str, Any]:
        return {
            "mode": "local",
            "cache_path": str(self.cache_path),
            "replicated": False,
        }


class ShardMetadataStore(MetadataStore):
    def __init__(
        self,
        owner_uri: str,
        *,
        cache_path: Path,
        timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        api_key: str | None = None,
    ):
        self.owner_uri = _normalize_uri(owner_uri)
        self.cache_path = Path(cache_path)
        self.timeout_secs = float(timeout_secs)
        self.api_key = api_key
        self._core = None

    def close(self) -> None:
        core = self._core
        if core is None:
            try:
                core = importlib.import_module("lynse._core")
            except Exception:
                core = None
        close_rpc = getattr(core, "metadata_rpc_close", None) if core is not None else None
        if close_rpc is not None:
            try:
                close_rpc(self.owner_uri)
            except Exception:
                pass
        self._core = None

    def _metadata_rpc_core(self):
        if self._core is not None:
            return self._core
        try:
            core = importlib.import_module("lynse._core")
        except Exception as exc:
            raise RuntimeError(
                "metadata RPC requires the LynseDB Rust extension. "
                "Rebuild/reinstall LynseDB so lynse._core exposes metadata_rpc_get/cas."
            ) from exc
        if not hasattr(core, "metadata_rpc_get") or not hasattr(core, "metadata_rpc_cas"):
            raise RuntimeError(
                "metadata RPC requires a newer LynseDB Rust extension with "
                "metadata_rpc_get/cas support."
            )
        self._core = core
        return core

    def _raise_metadata_rpc_error(self, exc: Exception) -> None:
        message = str(exc)
        if "metadata version conflict" in message:
            raise MetadataConflict(message) from exc
        raise RuntimeError(f"metadata RPC request to {self.owner_uri} failed: {message}") from exc

    def get(self, key: str) -> tuple[int, Any | None]:
        try:
            version, raw = self._metadata_rpc_core().metadata_rpc_get(
                self.owner_uri,
                key,
                self.timeout_secs,
                self.api_key,
            )
        except Exception as exc:
            self._raise_metadata_rpc_error(exc)
        if raw is None:
            return int(version), None
        return int(version), json.loads(raw)

    def cas(self, key: str, expected_version: int, value: Any) -> int:
        raw = json.dumps(value, separators=(",", ":"))
        try:
            return int(
                self._metadata_rpc_core().metadata_rpc_cas(
                    self.owner_uri,
                    key,
                    int(expected_version),
                    raw,
                    self.timeout_secs,
                    self.api_key,
                )
            )
        except Exception as exc:
            self._raise_metadata_rpc_error(exc)

    def metadata_status(self) -> dict[str, Any]:
        return {
            "mode": "single",
            "owners": [self.owner_uri],
            "replicated": False,
            "rpc_client": "rust",
        }


class QuorumMetadataStore(MetadataStore):
    def __init__(
        self,
        owner_uris: list[str],
        *,
        cache_path: Path,
        timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
        api_key: str | None = None,
        owner_stores: list[MetadataStore] | None = None,
    ):
        if len(owner_uris) < 3:
            raise ValueError("quorum metadata requires at least 3 owners")
        self.owner_uris = [_normalize_uri(uri) for uri in owner_uris]
        if owner_stores is not None:
            if len(owner_stores) != len(owner_uris):
                raise ValueError("owner_stores length must match owner_uris")
            self.owners = list(owner_stores)
            self._owns_child_stores = False
        else:
            self.owners = [
                ShardMetadataStore(
                    uri,
                    cache_path=cache_path,
                    timeout_secs=timeout_secs,
                    api_key=api_key,
                )
                for uri in self.owner_uris
            ]
            self._owns_child_stores = True
        self.cache_path = Path(cache_path)
        self._pool = ThreadPoolExecutor(max_workers=min(16, len(self.owners)))
        self._status_lock = threading.Lock()
        self._last_owner_status: list[dict[str, Any]] = [
            {"uri": uri, "healthy": None, "version": 0, "logical_version": 0}
            for uri in self.owner_uris
        ]
        self._last_committed_version = 0
        self._last_degraded = False

    @property
    def majority(self) -> int:
        return len(self.owners) // 2 + 1

    def close(self) -> None:
        if self._owns_child_stores:
            for owner in self.owners:
                close = getattr(owner, "close", None)
                if close:
                    close()
        self._pool.shutdown(wait=True, cancel_futures=False)

    def _owner_read(self, idx: int, key: str) -> dict[str, Any]:
        owner = self.owners[idx]
        uri = self.owner_uris[idx]
        try:
            version, raw_value = owner.get(key)
            logical_version, value, value_hash, enveloped = _unwrap_metadata_record(key, int(version), raw_value)
            return {
                "idx": idx,
                "uri": uri,
                "healthy": True,
                "storage_version": int(version),
                "logical_version": logical_version,
                "value": value,
                "value_hash": value_hash,
                "exists": value is not None,
                "enveloped": enveloped,
                "error": None,
            }
        except Exception as exc:
            return {
                "idx": idx,
                "uri": uri,
                "healthy": False,
                "storage_version": 0,
                "logical_version": 0,
                "value": None,
                "value_hash": "",
                "exists": False,
                "enveloped": True,
                "error": str(exc),
            }

    def _read_all(self, key: str) -> list[dict[str, Any]]:
        futures = [self._pool.submit(self._owner_read, idx, key) for idx in range(len(self.owners))]
        reads = [future.result() for future in as_completed(futures)]
        reads.sort(key=lambda item: int(item["idx"]))
        self._update_owner_status(reads)
        return reads

    def _update_owner_status(self, reads: list[dict[str, Any]]) -> None:
        with self._status_lock:
            self._last_owner_status = [
                {
                    "uri": read["uri"],
                    "healthy": bool(read["healthy"]),
                    "version": int(read.get("storage_version", 0) or 0),
                    "logical_version": int(read.get("logical_version", 0) or 0),
                    "error": read.get("error"),
                }
                for read in reads
            ]

    def _select_committed(self, key: str, reads: list[dict[str, Any]]) -> dict[str, Any]:
        healthy = [read for read in reads if read.get("healthy")]
        if len(healthy) < self.majority:
            errors = [read.get("error") for read in reads if read.get("error")]
            raise RuntimeError(
                f"metadata quorum read failed: {len(healthy)}/{self.majority} owners available; "
                + "; ".join(str(error) for error in errors[:3])
            )

        non_empty = [read for read in healthy if read.get("exists")]
        if len(non_empty) == 1 and len(non_empty) + sum(1 for read in healthy if not read.get("exists")) >= self.majority:
            chosen = dict(non_empty[0])
            chosen["bootstrap_repair"] = True
            return chosen

        groups: dict[tuple[bool, int, str], list[dict[str, Any]]] = {}
        for read in healthy:
            group_key = (
                bool(read.get("exists")),
                int(read.get("logical_version", 0) or 0),
                str(read.get("value_hash") or ""),
            )
            groups.setdefault(group_key, []).append(read)

        majority_groups = [
            (group_key, group_reads)
            for group_key, group_reads in groups.items()
            if len(group_reads) >= self.majority
        ]
        if not majority_groups:
            summary = [
                {
                    "exists": key_part[0],
                    "logical_version": key_part[1],
                    "count": len(group_reads),
                }
                for key_part, group_reads in groups.items()
            ]
            raise RuntimeError(f"metadata quorum has no committed majority for {key}: {summary}")

        majority_groups.sort(key=lambda item: (item[0][1], item[0][0]), reverse=True)
        _group_key, group_reads = majority_groups[0]
        chosen = dict(group_reads[0])
        chosen["bootstrap_repair"] = False
        return chosen

    def _repair_to_chosen(self, key: str, chosen: dict[str, Any], reads: list[dict[str, Any]]) -> None:
        if not chosen.get("exists"):
            return
        logical_version = int(chosen.get("logical_version", 0) or 0)
        value = chosen.get("value")
        value_hash = str(chosen.get("value_hash") or _metadata_value_hash(value))
        envelope = _metadata_record(key, logical_version, value)
        repairs = []
        for read in reads:
            if not read.get("healthy"):
                continue
            needs_repair = (
                not read.get("exists")
                or int(read.get("logical_version", 0) or 0) != logical_version
                or str(read.get("value_hash") or "") != value_hash
                or not read.get("enveloped", True)
            )
            if needs_repair:
                repairs.append(self._pool.submit(
                    self.owners[int(read["idx"])].cas,
                    key,
                    int(read.get("storage_version", 0) or 0),
                    envelope,
                ))
        for future in repairs:
            try:
                future.result()
            except Exception:
                pass

    def _set_committed_status(self, chosen: dict[str, Any], reads: list[dict[str, Any]]) -> None:
        healthy_count = sum(1 for read in reads if read.get("healthy"))
        in_sync = 0
        chosen_version = int(chosen.get("logical_version", 0) or 0)
        chosen_hash = str(chosen.get("value_hash") or "")
        for read in reads:
            if (
                read.get("healthy")
                and bool(read.get("exists")) == bool(chosen.get("exists"))
                and int(read.get("logical_version", 0) or 0) == chosen_version
                and str(read.get("value_hash") or "") == chosen_hash
            ):
                in_sync += 1
        with self._status_lock:
            self._last_committed_version = chosen_version
            self._last_degraded = healthy_count < len(self.owners) or in_sync < len(self.owners)

    def get(self, key: str) -> tuple[int, Any | None]:
        reads = self._read_all(key)
        chosen = self._select_committed(key, reads)
        self._repair_to_chosen(key, chosen, reads)
        self._set_committed_status(chosen, reads)
        return int(chosen.get("logical_version", 0) or 0), chosen.get("value")

    def cas(self, key: str, expected_version: int, value: Any) -> int:
        reads = self._read_all(key)
        chosen = self._select_committed(key, reads)
        current_version = int(chosen.get("logical_version", 0) or 0)
        if current_version != int(expected_version):
            self._repair_to_chosen(key, chosen, reads)
            self._set_committed_status(chosen, reads)
            raise MetadataConflict(
                f"metadata version conflict for {key}: expected {expected_version}, got {current_version}"
            )

        next_version = current_version + 1
        envelope = _metadata_record(key, next_version, value)
        future_reads = {
            self._pool.submit(
                self.owners[int(read["idx"])].cas,
                key,
                int(read.get("storage_version", 0) or 0),
                envelope,
            ): read
            for read in reads
            if read.get("healthy")
        }
        versions = []
        success_reads = []
        conflicts = []
        errors = []
        for future in as_completed(list(future_reads.keys())):
            try:
                versions.append(int(future.result()))
                success_reads.append(future_reads[future])
            except MetadataConflict as exc:
                conflicts.append(exc)
            except Exception as exc:
                errors.append(exc)
        if len(versions) >= self.majority:
            value_hash = _metadata_value_hash(value)
            status_reads = []
            success_indices = {int(read["idx"]) for read in success_reads}
            for read in reads:
                updated = dict(read)
                if int(read["idx"]) in success_indices:
                    updated["storage_version"] = int(read.get("storage_version", 0) or 0) + 1
                    updated["logical_version"] = next_version
                    updated["value"] = value
                    updated["value_hash"] = value_hash
                    updated["exists"] = value is not None
                    updated["enveloped"] = True
                    updated["healthy"] = True
                    updated["error"] = None
                status_reads.append(updated)
            self._update_owner_status(status_reads)
            self._set_committed_status({
                "logical_version": next_version,
                "value": value,
                "value_hash": value_hash,
                "exists": value is not None,
            }, status_reads)
            return next_version
        if conflicts:
            raise MetadataConflict("; ".join(str(exc) for exc in conflicts[:3]))
        raise RuntimeError(
            f"metadata quorum write failed: {len(versions)}/{self.majority} owners accepted; "
            + "; ".join(str(error) for error in errors[:3])
        )

    def metadata_status(self) -> dict[str, Any]:
        with self._status_lock:
            owners = [dict(item) for item in self._last_owner_status]
            committed_version = self._last_committed_version
            degraded = self._last_degraded
        healthy_count = sum(1 for item in owners if item.get("healthy"))
        return {
            "mode": "replicated",
            "replicated": True,
            "owners": owners,
            "owner_count": len(self.owners),
            "healthy_owners": healthy_count,
            "quorum": self.majority,
            "degraded": bool(degraded or healthy_count < self.majority),
            "committed_version": committed_version,
        }


class MetadataCoordinatorLease:
    def __init__(
        self,
        store: MetadataStore,
        coordinator_id: str,
        coordinator_uri: str,
        lease_secs: float = DEFAULT_COORDINATOR_LEASE_SECS,
        key: str = COORDINATOR_LEASE_METADATA_KEY,
    ):
        self.store = store
        self.key = key
        self.coordinator_id = str(coordinator_id)
        self.coordinator_uri = _normalize_uri(coordinator_uri)
        self.lease_secs = float(lease_secs)
        if self.lease_secs <= 0:
            raise ValueError("coordinator lease seconds must be > 0")

    def read(self) -> dict[str, Any]:
        _version, value = self.store.get(self.key)
        return dict(value or {})

    def _new_record(self, previous: dict[str, Any], now: float) -> dict[str, Any]:
        previous_id = previous.get("leader_id")
        previous_epoch = int(previous.get("lease_epoch", 0) or 0)
        epoch = previous_epoch if previous_id == self.coordinator_id else previous_epoch + 1
        return {
            "format": "lynsedb-coordinator-lease",
            "version": 1,
            "leader_id": self.coordinator_id,
            "leader_uri": self.coordinator_uri,
            "lease_epoch": epoch,
            "updated_at": now,
            "expires_at": now + self.lease_secs,
        }

    def try_acquire(self, now: float | None = None) -> tuple[bool, dict[str, Any]]:
        now = time.time() if now is None else float(now)
        version, current = self.store.get(self.key)
        current = dict(current or {})
        if (
            not current
            or _lease_record_is_expired(current, now)
            or current.get("leader_id") == self.coordinator_id
        ):
            record = self._new_record(current, now)
            try:
                self.store.cas(self.key, version, record)
                return True, record
            except MetadataConflict:
                return False, self.read()
        return False, current

    def renew(self, now: float | None = None) -> tuple[bool, dict[str, Any]]:
        now = time.time() if now is None else float(now)
        version, current = self.store.get(self.key)
        current = dict(current or {})
        if current.get("leader_id") != self.coordinator_id:
            return False, current
        record = self._new_record(current, now)
        try:
            self.store.cas(self.key, version, record)
            return True, record
        except MetadataConflict:
            return False, self.read()

    def release(self) -> None:
        version, current = self.store.get(self.key)
        current = dict(current or {})
        if current.get("leader_id") != self.coordinator_id:
            return
        record = dict(current)
        record["updated_at"] = time.time()
        record["expires_at"] = 0.0
        try:
            self.store.cas(self.key, version, record)
        except MetadataConflict:
            pass


class ClusterState:
    def __init__(
        self,
        path: Path,
        seed_config: dict[str, Any] | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.store = metadata_store or LocalMetadataStore(Path(path))
        self.path = Path(getattr(self.store, "cache_path", Path(path)))
        self._lock = threading.RLock()
        version, value = self.store.get(CLUSTER_STATE_METADATA_KEY)
        if value is not None:
            self.data = value
            self._normalize_state(self.data)
            self._metadata_version = int(version)
            self._write_cache()
        else:
            if not seed_config:
                raise ValueError("cluster state does not exist and no cluster config was provided")
            self.data = self._initial_state(seed_config)
            try:
                self._metadata_version = self.store.cas(CLUSTER_STATE_METADATA_KEY, 0, self.data)
                self._write_cache()
            except MetadataConflict:
                version, value = self.store.get(CLUSTER_STATE_METADATA_KEY)
                if value is None:
                    raise
                self.data = value
                self._normalize_state(self.data)
                self._metadata_version = int(version)
                self._write_cache()

    def _read_state(self) -> dict[str, Any]:
        version, data = self.store.get(CLUSTER_STATE_METADATA_KEY)
        if data is None:
            raise ValueError("cluster state does not exist")
        self._normalize_state(data)
        self._metadata_version = int(version)
        self._write_cache(data)
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
            self._metadata_version = self.store.cas(
                CLUSTER_STATE_METADATA_KEY,
                int(self._metadata_version),
                self.data,
            )
            self._write_cache()

    def _write_cache(self, data: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(data or self.data, indent=2, sort_keys=True).encode("utf-8")
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def reload(self) -> None:
        with self._lock:
            self.data = self._read_state()

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
                "index_mode": "FLAT-IP",
                "next_id": 0,
                "bucket_count": bucket_count,
                "bucket_to_group": bucket_to_group,
                "integer_id_routing": None,
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

    def update_collection_description(self, db_name: str, coll_name: str, description: str | None) -> None:
        with self._lock:
            coll = self.data.get("collections", {}).get(self.collection_key(db_name, coll_name))
            if coll is not None:
                coll["description"] = description
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

    def integer_id_routing(self, db_name: str, coll_name: str) -> str | None:
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            return coll.get("integer_id_routing")

    def mark_integer_id_routing(self, db_name: str, coll_name: str, routing: str) -> None:
        if routing not in {"external", "internal"}:
            raise ValueError(f"invalid integer ID routing mode: {routing}")
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            current = coll.get("integer_id_routing")
            if current is None:
                coll["integer_id_routing"] = routing
                self.bump_epoch()
                self.save()
            elif current != routing:
                raise ValueError(
                    f"collection {db_name}/{coll_name} already uses {current} integer ID routing"
                )

    def allocate_ids(self, db_name: str, coll_name: str, count: int) -> list[int]:
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            start = int(coll.get("next_id", 0))
            ids = list(range(start, start + count))
            coll["next_id"] = start + count
            self.bump_epoch()
            self.save()
            return ids

    def allocate_id_range(self, db_name: str, coll_name: str, count: int) -> tuple[int, int]:
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            start = int(coll.get("next_id", 0))
            coll["next_id"] = start + int(count)
            self.bump_epoch()
            self.save()
            return start, int(count)

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

    def group_for_public_id(self, db_name: str, coll_name: str, item_id: Any) -> dict[str, Any]:
        if isinstance(item_id, int) and not isinstance(item_id, bool):
            if self.integer_id_routing(db_name, coll_name) == "internal":
                return self.group_for_id(db_name, coll_name, int(item_id))
        return self.group_for_external_id(db_name, coll_name, item_id)

    def internal_id_routing_snapshot(self, db_name: str, coll_name: str):
        with self._lock:
            coll = self.data["collections"][self.collection_key(db_name, coll_name)]
            bucket_count = int(coll.get("bucket_count", self.data.get("bucket_count", DEFAULT_BUCKET_COUNT)))
            bucket_to_group = list(coll["bucket_to_group"])
            groups_by_name = {
                group["name"]: group
                for group in self.data["shard_groups"]
            }
        prefix_hasher = hashlib.blake2b(
            f"{db_name}/{coll_name}/".encode("utf-8"),
            digest_size=8,
        )
        return bucket_count, bucket_to_group, groups_by_name, prefix_hasher

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
        coordinator_id: str | None = None,
        coordinator_uri: str | None = None,
        coordinator_lease_secs: float = DEFAULT_COORDINATOR_LEASE_SECS,
        coordinator_lease: Any | None = None,
    ):
        self.state = state
        self.timeout_secs = float(timeout_secs)
        self.health_interval_secs = float(health_interval_secs)
        self.health_failures = int(health_failures)
        self.shard_api_key = shard_api_key
        self.coordinator_uri = _normalize_uri(coordinator_uri or "")
        self.coordinator_id = str(coordinator_id or self.coordinator_uri or f"{socket.gethostname()}:{os.getpid()}")
        self.client = RustRemoteSession(api_key=shard_api_key)
        self._health_lock = threading.Lock()
        self._failures: dict[str, int] = {}
        self._healthy: dict[str, bool] = {}
        self._rpc_available: dict[str, bool] = {}
        self._rpc_pool: dict[str, list[socket.socket]] = {}
        self._rpc_pool_lock = threading.Lock()
        self._rpc_max_idle_per_uri = 8
        self._stop = threading.Event()
        self._health_thread: threading.Thread | None = None
        self._lease: Any | None = None
        self._lease_thread: threading.Thread | None = None
        self._lease_lock = threading.Lock()
        self._lease_record: dict[str, Any] = {}
        self._is_leader = coordinator_lease is None
        self._leader_valid_until = float("inf") if coordinator_lease is None else 0.0
        if coordinator_lease is not None:
            if not self.coordinator_uri:
                raise ValueError("coordinator_uri is required when coordinator failover is enabled")
            self._lease = coordinator_lease
            self._is_leader = False
            self._leader_valid_until = 0.0
        self._fanout_pool = ThreadPoolExecutor(max_workers=32)
        self._replica_pool = ThreadPoolExecutor(max_workers=32)
        self._async_ready = threading.Event()
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_client = self.client
        self._async_thread = threading.Thread(target=self._async_loop_main, daemon=True)
        self._async_thread.start()
        self._rust_read_lock = threading.Lock()
        self._rust_read = None
        self._rust_read_epoch: int | None = None

    def start_coordinator_lease_loop(self) -> None:
        if self._lease is None:
            return
        if self._lease_thread and self._lease_thread.is_alive():
            return
        self.try_become_leader_once()
        self._lease_thread = threading.Thread(target=self._coordinator_lease_loop, daemon=True)
        self._lease_thread.start()

    def start_health_loop(self) -> None:
        if self._health_thread and self._health_thread.is_alive():
            return
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._lease_thread:
            self._lease_thread.join(timeout=2)
        if self._health_thread:
            self._health_thread.join(timeout=2)
        was_leader = self.is_active_coordinator()
        if self._lease is not None:
            try:
                if was_leader:
                    self._lease.release()
                self._set_leader_view(self.current_leader_record(), is_self=False)
            except Exception:
                pass
        self._fanout_pool.shutdown(wait=True, cancel_futures=False)
        self._replica_pool.shutdown(wait=True, cancel_futures=False)
        self.client.close()
        self._close_rpc_pool()
        if self._async_thread.is_alive() and self._async_ready.wait(timeout=5):
            loop = self._async_loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
            self._async_thread.join(timeout=2)

    def _set_leader_view(
        self,
        record: dict[str, Any],
        *,
        is_self: bool,
        local_deadline: float | None = None,
    ) -> None:
        with self._lease_lock:
            self._lease_record = dict(record or {})
            self._is_leader = bool(is_self)
            if is_self:
                self._leader_valid_until = (
                    time.monotonic() + (self._lease.lease_secs if self._lease else 0.0)
                    if local_deadline is None
                    else local_deadline
                )
            else:
                self._leader_valid_until = 0.0

    def _coordinator_lease_loop(self) -> None:
        assert self._lease is not None
        interval = max(0.1, min(self._lease.lease_secs / 3.0, 1.0))
        while not self._stop.wait(interval):
            try:
                if self.is_active_coordinator():
                    ok, record = self._lease.renew()
                    self._set_leader_view(record, is_self=ok)
                else:
                    self.try_become_leader_once()
            except Exception:
                self._set_leader_view(self.current_leader_record(), is_self=False)

    def try_become_leader_once(self) -> bool:
        if self._lease is None:
            return True
        was_active = self.is_active_coordinator()
        try:
            acquired, record = self._lease.try_acquire()
        except Exception:
            return False
        self._set_leader_view(record, is_self=acquired)
        if acquired and not was_active:
            try:
                self.state.reload()
            except Exception:
                self._set_leader_view(record, is_self=False)
                return False
        return acquired

    def is_active_coordinator(self) -> bool:
        if self._lease is None:
            return True
        return bool(self._is_leader and time.monotonic() < self._leader_valid_until)

    def current_leader_record(self) -> dict[str, Any]:
        if self._lease is None:
            return {
                "leader_id": self.coordinator_id,
                "leader_uri": self.coordinator_uri,
                "lease_epoch": 0,
                "expires_at": float("inf"),
            }
        record = self._lease.read()
        with self._lease_lock:
            if record:
                self._lease_record = dict(record)
            elif self._lease_record:
                record = dict(self._lease_record)
        return record

    def coordinator_status(self) -> dict[str, Any]:
        leader = self.current_leader_record()
        active = self.is_active_coordinator()
        role = "leader" if active else "standby"
        if self._lease is None:
            role = "single"
        metadata_status = {}
        status_fn = getattr(self.state.store, "metadata_status", None)
        if callable(status_fn):
            try:
                metadata_status = status_fn()
            except Exception as exc:
                metadata_status = {"mode": "unknown", "error": str(exc)}
        return _json_success({
            "role": role,
            "coordinator_id": self.coordinator_id,
            "coordinator_uri": self.coordinator_uri,
            "leader": leader,
            "lease_enabled": self._lease is not None,
            "metadata": metadata_status,
        })

    def proxy_request_to_leader(
        self,
        method: str,
        target: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ):
        leader = self.current_leader_record()
        if not leader or _lease_record_is_expired(leader):
            if self.try_become_leader_once():
                raise RuntimeError("coordinator became leader; retry locally")
            leader = self.current_leader_record()

        leader_uri = _normalize_uri(str(leader.get("leader_uri") or ""))
        if not leader_uri:
            raise RuntimeError("no active coordinator leader is available")
        if self.coordinator_uri and leader_uri == self.coordinator_uri:
            raise RuntimeError("local coordinator is not ready to serve as leader")

        forwarded_headers: dict[str, str] = {"X-Lynse-Coordinator-Proxy": "1"}
        for name, value in (headers or {}).items():
            lower = name.lower()
            if lower in {
                "accept",
                "authorization",
                "content-type",
            }:
                forwarded_headers[name] = value

        return self.client.request(
            method,
            f"{leader_uri}{target}",
            content=body,
            headers=forwarded_headers,
        )

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
    ):
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

    def _async_loop_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._async_loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_start())
        self._async_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(self._async_stop())
            loop.close()

    async def _async_start(self) -> None:
        self._async_client = self.client

    async def _async_stop(self) -> None:
        self._async_client = None

    def _run_async(self, coro):
        if not self._async_ready.wait(timeout=5):
            raise RuntimeError("cluster async fan-out loop did not start")
        loop = self._async_loop
        if loop is None or loop.is_closed():
            raise RuntimeError("cluster async fan-out loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    async def _async_request(
        self,
        method: str,
        uri: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        content: bytes | None = None,
        content_type: str | None = None,
    ):
        if self._async_client is None:
            raise RuntimeError("cluster async HTTP client is not initialized")
        headers = self._headers()
        if content_type:
            headers["Content-Type"] = content_type
        url = f"{_normalize_uri(uri)}{path}"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._fanout_pool,
            lambda: self._async_client.request(
                method,
                url,
                json=json_body,
                params=params,
                content=content,
                headers=headers,
            ),
        )

    async def _async_json_post(self, uri: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._async_request("POST", uri, path, json_body=body)
        if response.status_code != 200:
            raise RuntimeError(response.text)
        payload = response.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("error") or response.text)
        return payload

    async def _async_json_post_many(
        self,
        calls: list[tuple[str, str, dict[str, Any]]],
        *,
        return_exceptions: bool = False,
    ) -> list[Any]:
        tasks = [
            self._async_json_post(uri, path, body)
            for uri, path, body in calls
        ]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    def _json_post_many(
        self,
        calls: list[tuple[str, str, dict[str, Any]]],
        *,
        return_exceptions: bool = False,
    ) -> list[Any]:
        if not calls:
            return []
        if type(self)._json_post is not ClusterCoordinator._json_post:
            results: list[Any] = []
            for uri, path, body in calls:
                try:
                    results.append(self._json_post(uri, path, body))
                except Exception as exc:
                    if not return_exceptions:
                        raise
                    results.append(exc)
            return results
        return self._run_async(self._async_json_post_many(
            calls,
            return_exceptions=return_exceptions,
        ))

    def _take_rpc_socket(self, uri: str) -> socket.socket:
        with self._rpc_pool_lock:
            pool = self._rpc_pool.get(uri)
            if pool:
                sock = pool.pop()
                if not pool:
                    self._rpc_pool.pop(uri, None)
                return sock
        host, port = _derive_rpc_target(uri)
        sock = socket.create_connection((host, port), timeout=self.timeout_secs)
        sock.settimeout(self.timeout_secs)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        return sock

    def _return_rpc_socket(self, uri: str, sock: socket.socket) -> None:
        with self._rpc_pool_lock:
            pool = self._rpc_pool.setdefault(uri, [])
            if len(pool) < self._rpc_max_idle_per_uri:
                pool.append(sock)
                return
        try:
            sock.close()
        except Exception:
            pass

    def _close_socket(self, sock: socket.socket | None) -> None:
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass

    def _close_rpc_pool(self) -> None:
        with self._rpc_pool_lock:
            sockets = [sock for pool in self._rpc_pool.values() for sock in pool]
            self._rpc_pool.clear()
        for sock in sockets:
            self._close_socket(sock)

    def _rpc_roundtrip(self, uri: str, payload: bytes) -> bytes:
        sock: socket.socket | None = None
        try:
            sock = self._take_rpc_socket(uri)
            sock.sendall(struct.pack("<I", len(payload)) + payload)
            header = self._recv_exact(sock, 4)
            frame_len = struct.unpack("<I", header)[0]
            frame = self._recv_exact(sock, frame_len)
            self._return_rpc_socket(uri, sock)
            sock = None
            return frame
        finally:
            self._close_socket(sock)

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

        last_error: Exception | None = None
        frame = b""
        for _attempt in range(2):
            try:
                frame = self._rpc_roundtrip(uri, payload)
                break
            except Exception as exc:
                last_error = exc
        else:
            self._rpc_available[uri] = False
            raise last_error or RuntimeError("internal RPC request failed")

        self._rpc_available[uri] = True
        try:
            if len(frame) < 5:
                raise RuntimeError("internal RPC response frame is too short")
            status = frame[0]
            meta_len = struct.unpack("<I", frame[1:5])[0]
            if len(frame) < 5 + meta_len:
                raise RuntimeError("internal RPC response metadata length exceeds frame")
            response_meta = json.loads(frame[5:5 + meta_len]) if meta_len else {}
            response_raw = frame[5 + meta_len:]
        except Exception:
            self._rpc_available[uri] = False
            raise

        if status != 0:
            raise RuntimeError(response_meta.get("error") or "internal RPC request failed")
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

    def _rust_read_coordinator(self):
        core = importlib.import_module("lynse._core")
        coordinator_cls = getattr(core, "ClusterReadCoordinator")

        epoch = int(self.state.data.get("meta_epoch", 0))
        with self._rust_read_lock:
            if self._rust_read is None or self._rust_read_epoch != epoch:
                self._rust_read = coordinator_cls(
                    str(self.state.path),
                    self.timeout_secs,
                    self.shard_api_key,
                )
                self._rust_read_epoch = epoch
            return self._rust_read

    def _rust_binary_read(
        self,
        method: str,
        params: dict[str, Any],
        body: bytes,
        coll: dict[str, Any],
    ) -> bytes:
        rust_read = self._rust_read_coordinator()

        meta = dict(params)
        if coll.get("index_mode") is not None:
            meta["index_mode"] = coll.get("index_mode")
        meta_json = json.dumps(meta, separators=(",", ":"))
        return bytes(getattr(rust_read, method)(meta_json, body))

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
        if not self.is_active_coordinator():
            return
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

        primary_errors = []
        first_payload: dict[str, Any] | None = None
        calls = [(uri, path, body) for uri, _is_primary in targets]
        for (uri, is_primary), result in zip(
            targets,
            self._json_post_many(calls, return_exceptions=True),
        ):
            if isinstance(result, Exception):
                if is_primary:
                    primary_errors.append(f"{uri}: {result}")
                else:
                    self.state.mark_replica_stale(uri)
            else:
                payload = result
                if first_payload is None:
                    first_payload = payload
        if require_primary and primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success()

    def broadcast_control(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        futures = [
            self._fanout_pool.submit(self.write_group_control, group, path, body)
            for group in self.state.groups()
        ]
        first_payload: dict[str, Any] | None = None
        for future in as_completed(futures):
            payload = future.result()
            if first_payload is None:
                first_payload = payload
        return first_payload or _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
        })

    def write_group_json(self, group: dict[str, Any], path: str, body: dict[str, Any]) -> dict[str, Any]:
        targets = self.state.writable_uris_for_group(group)
        first_payload: dict[str, Any] | None = None
        primary_errors = []
        calls = [(uri, path, body) for uri, _is_primary in targets]
        for (uri, is_primary), result in zip(
            targets,
            self._json_post_many(calls, return_exceptions=True),
        ):
            if isinstance(result, Exception):
                if is_primary:
                    primary_errors.append(f"{uri}: {result}")
                else:
                    self.state.mark_replica_stale(uri)
            else:
                payload = result
                if first_payload is None and is_primary:
                    first_payload = payload
        if primary_errors:
            raise RuntimeError("; ".join(primary_errors))
        return first_payload or _json_success()

    def write_group_control(self, group: dict[str, Any], path: str, body: dict[str, Any]) -> dict[str, Any]:
        action = path[1:] if path.startswith("/") else path
        meta = {
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "action": action,
        }
        http_body = {
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
        }
        first_payload: dict[str, Any] | None = None
        primary_errors = []
        for uri, is_primary in self.state.writable_uris_for_group(group):
            try:
                if self._can_rpc(uri):
                    self._rpc_request(uri, RPC_OP_COLLECTION_CONTROL, meta)
                    payload = _json_success({
                        "database_name": body["database_name"],
                        "collection_name": body["collection_name"],
                    })
                else:
                    payload = self._json_post(uri, path, http_body)
                if is_primary:
                    first_payload = payload
            except Exception:
                try:
                    payload = self._json_post(uri, path, http_body)
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
        if isinstance(vector_raw, memoryview):
            vector_bytes = vector_raw.tobytes()
        else:
            vector_bytes = bytes(vector_raw)
        raw = vector_bytes + id_raw
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

    def write_group_internal_ids(self, group: dict[str, Any], path: str, body: dict[str, Any]) -> dict[str, Any]:
        op = RPC_OP_RESTORE_ITEMS if path == "/restore" else RPC_OP_DELETE_ITEMS
        meta = {
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "ids": [int(item_id) for item_id in body.get("ids", [])],
        }
        primary_errors = []
        first_payload: dict[str, Any] | None = None
        for uri, is_primary in self.state.writable_uris_for_group(group):
            try:
                if not self._can_rpc(uri):
                    raise RuntimeError("internal ID writes require internal RPC")
                self._rpc_request(uri, op, meta)
                payload = _json_success({
                    "database_name": body["database_name"],
                    "collection_name": body["collection_name"],
                    "status": "ok",
                })
                if is_primary:
                    first_payload = payload
            except Exception as exc:
                if is_primary:
                    primary_errors.append(f"{uri}: {exc}")
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
                "integer_id_routing": coll.get("integer_id_routing"),
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

        futures = []
        for group, group_items in grouped.values():
            payload = {
                "database_name": db_name,
                "collection_name": coll_name,
                "items": group_items,
            }
            futures.append(self._fanout_pool.submit(
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
        if any(isinstance(item_id, int) and not isinstance(item_id, bool) for item_id in ids):
            self.state.mark_integer_id_routing(db_name, coll_name, "external")

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

        futures = []
        for bucket in grouped.values():
            payload = {
                "database_name": db_name,
                "collection_name": coll_name,
                "ids": bucket["ids"],
                "vectors": bucket["vectors"],
                "fields": bucket["fields"],
            }
            futures.append(self._fanout_pool.submit(self.write_group_json, bucket["group"], "/add", payload))
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
        if any(isinstance(item_id, int) and not isinstance(item_id, bool) for item_id in ids):
            self.state.mark_integer_id_routing(db_name, coll_name, "external")

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

        futures = []
        for bucket in grouped.values():
            payload = {
                "database_name": db_name,
                "collection_name": coll_name,
                "ids": bucket["ids"],
                "vectors": bucket["vectors"],
                "fields": bucket["fields"],
            }
            futures.append(self._fanout_pool.submit(self.write_group_json, bucket["group"], "/upsert", payload))
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
        self.state.mark_integer_id_routing(db_name, coll_name, "internal")
        vector_raw, ids, fields, _ = _split_binary_items_payload(
            body,
            n_vectors,
            dim,
            vector_encoding,
            params.get("ids_encoding"),
            params.get("ids_start"),
        )
        stride = dim * _vector_wire_width(vector_encoding)
        bucket_count, bucket_to_group, groups_by_name, prefix_hasher = (
            self.state.internal_id_routing_snapshot(db_name, coll_name)
        )
        grouped: dict[str, dict[str, Any]] = {}
        for idx, item_id in enumerate(ids):
            bucket = _hash_u64_prefixed(prefix_hasher, int(item_id)) % bucket_count
            group = groups_by_name[bucket_to_group[bucket]]
            bucket = grouped.setdefault(
                group["name"],
                {"group": group, "raw": bytearray(), "ids": [], "fields": [] if fields is not None else None},
            )
            start = idx * stride
            bucket["raw"].extend(vector_raw[start:start + stride])
            bucket["ids"].append(int(item_id))
            if fields is not None:
                bucket["fields"].append(fields[idx])

        futures = []
        for bucket in grouped.values():
            futures.append(self._fanout_pool.submit(
                self.write_group_binary_payload,
                bucket["group"],
                path,
                db_name,
                coll_name,
                dim,
                vector_encoding,
                bucket["raw"],
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

    def route_bulk_add_binary(self, params: dict[str, Any], body: bytes) -> dict[str, Any]:
        db_name = str(params["database_name"])
        coll_name = str(params["collection_name"])
        dim = int(params["dim"])
        n_vectors = int(params["n_vectors"])
        vector_encoding = _normalize_vector_encoding(params.get("vector_encoding"))
        expected_bytes = n_vectors * dim * _vector_wire_width(vector_encoding)
        if len(body) != expected_bytes:
            raise ValueError(
                f"expected {expected_bytes} encoded vector bytes "
                f"({n_vectors} vectors x {dim} dim), got {len(body)}"
            )
        self.state.update_collection_dim_if_unset(db_name, coll_name, dim)
        self.state.mark_integer_id_routing(db_name, coll_name, "internal")
        start_id, id_count = self.state.allocate_id_range(db_name, coll_name, n_vectors)
        return_ids = str(params.get("return_ids", "false")).lower() == "true"
        response_ids = [] if return_ids else None

        vector_raw = memoryview(body)
        stride = dim * _vector_wire_width(vector_encoding)
        bucket_count, bucket_to_group, groups_by_name, prefix_hasher = (
            self.state.internal_id_routing_snapshot(db_name, coll_name)
        )
        grouped: dict[str, dict[str, Any]] = {}
        for idx, item_id in enumerate(range(start_id, start_id + id_count)):
            if response_ids is not None:
                response_ids.append(item_id)
            routing_bucket = _hash_u64_prefixed(prefix_hasher, item_id) % bucket_count
            group = groups_by_name[bucket_to_group[routing_bucket]]
            bucket = grouped.setdefault(
                group["name"],
                {"group": group, "raw": bytearray(), "ids": []},
            )
            start = idx * stride
            bucket["raw"].extend(vector_raw[start:start + stride])
            bucket["ids"].append(int(item_id))

        futures = []
        for bucket in grouped.values():
            futures.append(self._fanout_pool.submit(
                self.write_group_binary_payload,
                bucket["group"],
                "/add_binary_ids",
                db_name,
                coll_name,
                dim,
                vector_encoding,
                bucket["raw"],
                bucket["ids"],
                None,
            ))
        for future in as_completed(futures):
            future.result()

        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            **({"ids": response_ids or []} if return_ids else {"n_vectors": n_vectors}),
        })

    def route_ids_write(self, body: dict[str, Any], endpoint: str) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        ids = list(body.get("ids", []))
        internal_integer_ids = (
            self.state.integer_id_routing(db_name, coll_name) == "internal"
            and all(isinstance(item_id, int) and not isinstance(item_id, bool) for item_id in ids)
        )
        grouped: dict[str, tuple[dict[str, Any], list[Any]]] = {}
        for item_id in ids:
            group = self.state.group_for_public_id(db_name, coll_name, item_id)
            grouped.setdefault(group["name"], (group, []))[1].append(item_id)
        futures = []
        for group, group_ids in grouped.values():
            writer = self.write_group_internal_ids if internal_integer_ids else self.write_group_ids
            futures.append(self._fanout_pool.submit(writer, group, endpoint, {
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

    def fanout_json_read(self, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        calls = [
            (self.read_uri_for_group(group), path, body)
            for group in self.state.groups()
        ]
        return self._json_post_many(calls)

    def merge_search_endpoint(
        self,
        path: str,
        body: dict[str, Any],
        *,
        ascending: bool,
        index: str | None = None,
        fusion: str | None = None,
        include_profile: bool = False,
    ) -> dict[str, Any]:
        k = int(body.get("k") or 10)
        return_fields = bool(body.get("return_fields", False))
        results = []
        profiles = []
        for payload in self.fanout_json_read(path, body):
            params = payload.get("params", {})
            items = params.get("items", {})
            results.append((
                list(items.get("ids", [])),
                [float(x) for x in items.get("scores", [])],
                items.get("fields", []) or [],
            ))
            if include_profile:
                profiles.append(params.get("profile", {}))
        ids, scores, fields = _merge_pairs(results, k, ascending, return_fields)
        items = {
            "k": k,
            "ids": ids,
            "scores": scores,
            "fields": fields,
        }
        if index is not None:
            items["index"] = index
        if fusion is not None:
            items["fusion"] = fusion
        payload = _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "items": items,
        })
        if include_profile:
            payload["params"]["profile"] = {"shards": profiles}
        return payload

    def search_json(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        coll = self.state.get_collection(db_name, coll_name)
        if not coll:
            raise KeyError(f"Collection '{coll_name}' not found")
        groups = self.state.groups()
        if len(groups) == 1:
            uri = self.read_uri_for_group(groups[0])
            return self._json_post(uri, "/search", body)
        k = int(body.get("k") or 10)
        return_fields = bool(body.get("return_fields", False))
        ascending = _is_ascending_index(coll.get("index_mode"))
        results = []
        calls = [
            (self.read_uri_for_group(group), "/search", body)
            for group in groups
        ]
        for payload in self._json_post_many(calls):
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

    def batch_search_json(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        coll = self.state.get_collection(db_name, coll_name)
        if not coll:
            raise KeyError(f"Collection '{coll_name}' not found")
        groups = self.state.groups()
        if len(groups) == 1:
            uri = self.read_uri_for_group(groups[0])
            return self._json_post(uri, "/batch_search", body)
        k = int(body.get("k") or 10)
        vectors = body.get("vectors") or []
        return_fields = bool(body.get("return_fields", False))
        ascending = _is_ascending_index(coll.get("index_mode"))
        per_query: list[list[tuple[list[Any], list[float], list[dict[str, Any]]]]] = [
            [] for _ in range(len(vectors))
        ]

        calls = [
            (self.read_uri_for_group(group), "/batch_search", body)
            for group in groups
        ]
        for payload in self._json_post_many(calls):
            shard_results = payload.get("params", {}).get("results", [])
            if len(shard_results) != len(vectors):
                raise RuntimeError(f"batch_search returned {len(shard_results)} queries, expected {len(vectors)}")
            for query_idx, items in enumerate(shard_results):
                per_query[query_idx].append((
                    list(items.get("ids", [])),
                    [float(x) for x in items.get("scores", [])],
                    items.get("fields", []) or [],
                ))

        results = []
        for query_results in per_query:
            ids, scores, fields = _merge_pairs(query_results, k, ascending, return_fields)
            results.append({
                "ids": ids,
                "scores": scores,
                "fields": fields,
            })
        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "results": results,
        })

    def bm25_search_json(self, body: dict[str, Any]) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        groups = self.state.groups()
        if len(groups) == 1:
            uri = self.read_uri_for_group(groups[0])
            return self._json_post(uri, "/bm25_search", body)
        k = int(body.get("k") or 10)
        return_fields = bool(body.get("return_fields", False))
        results = []
        calls = [
            (self.read_uri_for_group(group), "/bm25_search", body)
            for group in groups
        ]
        for payload in self._json_post_many(calls):
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
        groups = self.state.groups()
        if len(groups) == 1:
            uri = self.read_uri_for_group(groups[0])
            return self._binary_call(
                uri,
                RPC_OP_SEARCH,
                "/search_binary",
                params,
                body,
            )
        return self._rust_binary_read("search_binary", params, body, coll)

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
        groups = self.state.groups()
        if len(groups) == 1:
            uri = self.read_uri_for_group(groups[0])
            return self._binary_call(
                uri,
                RPC_OP_BATCH_SEARCH,
                "/batch_search_binary",
                params,
                body,
            )
        return self._rust_binary_read("batch_search_binary", params, body, coll)

    def head_tail_binary(self, endpoint: str, params: dict[str, Any]) -> bytes:
        n = int(params.get("n") or 5)
        rows: list[tuple[list[float], int, dict[str, Any]]] = []
        futures = []
        for group in self.state.groups():
            uri = self.read_uri_for_group(group)
            futures.append(self._fanout_pool.submit(
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
        futures = []
        for group in self.state.groups():
            uri = self.read_uri_for_group(group)
            futures.append(self._fanout_pool.submit(self._json_post, uri, endpoint, body))
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
        futures = []
        filter_ids = body.get("filter_ids") or []
        if filter_ids and body.get("where") is None:
            grouped: dict[str, tuple[dict[str, Any], list[Any]]] = {}
            for item_id in filter_ids:
                group = self.state.group_for_public_id(
                    body["database_name"],
                    body["collection_name"],
                    item_id,
                )
                grouped.setdefault(group["name"], (group, []))[1].append(item_id)
            for group, group_ids in grouped.values():
                uri = self.read_uri_for_group(group)
                payload = {**body, "filter_ids": group_ids}
                futures.append(self._fanout_pool.submit(self._json_post, uri, endpoint, payload))
        else:
            for group in self.state.groups():
                uri = self.read_uri_for_group(group)
                futures.append(self._fanout_pool.submit(self._json_post, uri, endpoint, body))
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
        group = self.state.group_for_public_id(
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
        futures = []
        for group in self.state.groups():
            uri = self.read_uri_for_group(group)
            futures.append(self._fanout_pool.submit(self._json_post, uri, "/search_range", body))
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
        futures = []
        for group in self.state.groups():
            uri = self.read_uri_for_group(group)
            futures.append(self._fanout_pool.submit(self._json_post, uri, "/list_deleted_ids", body))
        for future in as_completed(futures):
            payload = future.result()
            ids.update(payload.get("params", {}).get("ids", []))
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "ids": sorted(ids, key=lambda value: (type(value).__name__, str(value))),
        })

    def list_vector_fields(self, body: dict[str, Any]) -> dict[str, Any]:
        by_name: dict[str, dict[str, Any]] = {}
        for payload in self.fanout_json_read("/list_vector_fields", body):
            for field in payload.get("params", {}).get("fields", []):
                name = field.get("name")
                if name is not None and name not in by_name:
                    by_name[name] = field
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "fields": list(by_name.values()),
        })

    def route_vector_payloads(self, body: dict[str, Any], endpoint: str) -> dict[str, Any]:
        db_name = body["database_name"]
        coll_name = body["collection_name"]
        ids = list(body.get("ids") or [])
        vectors = list(body.get("vectors") or [])
        if len(ids) != len(vectors):
            raise ValueError("ids length must match vectors length")
        grouped: dict[str, dict[str, Any]] = {}
        for idx, item_id in enumerate(ids):
            group = self.state.group_for_public_id(db_name, coll_name, item_id)
            bucket = grouped.setdefault(
                group["name"],
                {"group": group, "ids": [], "vectors": []},
            )
            bucket["ids"].append(item_id)
            bucket["vectors"].append(vectors[idx])
        futures = []
        for bucket in grouped.values():
            payload = {
                **body,
                "ids": bucket["ids"],
                "vectors": bucket["vectors"],
            }
            futures.append(self._fanout_pool.submit(self.write_group_json, bucket["group"], endpoint, payload))
        for future in as_completed(futures):
            future.result()
        return _json_success({
            "database_name": db_name,
            "collection_name": coll_name,
            "ids": ids,
        })

    def compact(self, body: dict[str, Any]) -> dict[str, Any]:
        removed = 0
        for payload in self.fanout_json_read("/compact", body):
            removed += int(payload.get("params", {}).get("vectors_removed", 0))
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "vectors_removed": removed,
        })

    def stats(self, body: dict[str, Any]) -> dict[str, Any]:
        totals = {
            "n_vectors": 0,
            "n_live": 0,
            "n_tombstoned": 0,
            "dimension": 0,
            "index_mode": None,
            "max_id": -1,
        }
        for payload in self.fanout_json_read("/stats", body):
            stats = payload.get("params", {}).get("stats", {})
            totals["n_vectors"] += int(stats.get("n_vectors", 0))
            totals["n_live"] += int(stats.get("n_live", 0))
            totals["n_tombstoned"] += int(stats.get("n_tombstoned", 0))
            totals["dimension"] = int(stats.get("dimension") or totals["dimension"])
            totals["max_id"] = max(int(totals["max_id"]), int(stats.get("max_id", -1)))
        coll = self.state.get_collection(body["database_name"], body["collection_name"]) or {}
        totals["index_mode"] = coll.get("index_mode")
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "stats": totals,
        })

    def artifact_fanout(
        self,
        path: str,
        body: dict[str, Any],
        path_key: str,
    ) -> dict[str, Any]:
        base_path = body[path_key]
        futures = []
        for group in self.state.groups():
            uri = group["primary"]
            shard_path = _shard_artifact_path(base_path, group["name"])
            payload = {**body, path_key: shard_path}
            futures.append((
                group["name"],
                uri,
                shard_path,
                self._fanout_pool.submit(self._json_post, uri, path, payload),
            ))

        shards = []
        for group_name, uri, shard_path, future in futures:
            payload = future.result()
            shards.append({
                "group": group_name,
                "uri": uri,
                path_key: shard_path,
                "params": payload.get("params", {}),
            })

        params = {
            "database_name": body["database_name"],
            path_key: str(base_path),
            "shards": shards,
        }
        if "collection_name" in body:
            params["collection_name"] = body["collection_name"]
        return _json_success(params)

    def collection_paths(self, body: dict[str, Any]) -> dict[str, Any]:
        shards = []
        for group in self.state.groups():
            uri = self.read_uri_for_group(group)
            payload = self._json_post(uri, "/get_collection_path", body)
            shards.append({
                "group": group["name"],
                "uri": uri,
                "collection_path": payload.get("params", {}).get("collection_path"),
            })
        return _json_success({
            "database_name": body["database_name"],
            "collection_name": body["collection_name"],
            "collection_path": None,
            "shards": shards,
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
        return self._json_from_body(body)

    def _json_from_body(self, body: bytes) -> dict[str, Any]:
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

    def _send_proxy_response(self, response) -> None:
        data = response.content
        self.send_response(response.status_code)
        content_type = response.headers.get("Content-Type")
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy_if_standby(self, method: str, body: bytes = b"") -> bool:
        parsed = urlparse(self.path)
        if method == "GET" and parsed.path in {"/", "/coordinator_status"}:
            return False
        coord = self.coordinator
        if coord.is_active_coordinator() or coord.try_become_leader_once():
            return False
        if self.headers.get("X-Lynse-Coordinator-Proxy") == "1":
            self._send_error_json("coordinator is standby and cannot proxy this request", status=503)
            return True
        try:
            response = coord.proxy_request_to_leader(
                method,
                self.path,
                headers=dict(self.headers.items()),
                body=body,
            )
        except RuntimeError as exc:
            if "retry locally" in str(exc) and coord.is_active_coordinator():
                return False
            self._send_error_json(str(exc), status=503)
            return True
        except OSError as exc:
            self._send_error_json(f"failed to proxy request to active coordinator: {exc}", status=503)
            return True
        self._send_proxy_response(response)
        return True

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if self._proxy_if_standby("GET"):
                return
            if parsed.path == "/":
                self._send_json(_json_success({"message": "LynseDB cluster coordinator"}))
            elif parsed.path == "/list_databases":
                self._send_json(_json_success({"databases": self.coordinator.state.databases()}))
            elif parsed.path == "/cluster_info":
                self._send_json(_json_success(self.coordinator.state.data))
            elif parsed.path == "/coordinator_status":
                self._send_json(self.coordinator.coordinator_status())
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
            raw_body = self._read_body()
            if self._proxy_if_standby("POST", raw_body):
                return
            if path == "/search_binary":
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                buf = self.coordinator.search_binary(params, raw_body)
                self._send_binary(buf)
                return
            if path == "/batch_search_binary":
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                buf = self.coordinator.batch_search_binary(params, raw_body)
                self._send_binary(buf)
                return
            if path == "/bulk_add_binary":
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                payload = self.coordinator.route_bulk_add_binary(params, raw_body)
                self._send_json(payload)
                return
            if path in {"/add_binary_ids", "/upsert_binary"}:
                params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                payload = self.coordinator.route_binary_items(params, raw_body, path)
                self._send_json(payload)
                return

            body = self._json_from_body(raw_body)
            if path == "/create_database":
                payload = self.coordinator.create_database(body)
            elif path in {"/delete_database", "/drop_database"}:
                payload = self.coordinator.drop_database(body)
            elif path == "/database_exists":
                payload = _json_success({
                    "database_name": body["database_name"],
                    "exists": body["database_name"] in self.coordinator.state.databases(),
                })
            elif path == "/snapshot_database":
                payload = self.coordinator.artifact_fanout(path, body, "snapshot_path")
            elif path == "/restore_database":
                self._send_error_json("restore_database is not supported by the cluster coordinator yet", status=501)
                return
            elif path == "/show_collections":
                payload = self.coordinator.show_collections(body)
            elif path == "/show_collections_details":
                payload = self.coordinator.show_collections_details(body)
            elif path == "/required_collection":
                payload = self.coordinator.require_collection(body)
            elif path == "/drop_collection":
                payload = self.coordinator.drop_collection(body)
            elif path == "/snapshot_collection":
                payload = self.coordinator.artifact_fanout(path, body, "snapshot_path")
            elif path == "/export_collection":
                payload = self.coordinator.artifact_fanout(path, body, "export_path")
            elif path in {"/restore_collection", "/import_collection"}:
                self._send_error_json(f"{path[1:]} is not supported by the cluster coordinator yet", status=501)
                return
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
            elif path == "/batch_search":
                payload = self.coordinator.batch_search_json(body)
            elif path == "/search_profile":
                coll = self.coordinator.state.get_collection(body["database_name"], body["collection_name"]) or {}
                payload = self.coordinator.merge_search_endpoint(
                    "/search_profile",
                    body,
                    ascending=_is_ascending_index(coll.get("index_mode")),
                    index=coll.get("index_mode") or "FLAT-IP",
                    include_profile=True,
                )
            elif path == "/bm25_search":
                payload = self.coordinator.bm25_search_json(body)
            elif path == "/sparse_search":
                payload = self.coordinator.merge_search_endpoint(
                    "/sparse_search",
                    body,
                    ascending=False,
                    index="SPARSE-FLAT-IP",
                )
            elif path == "/hybrid_search":
                payload = self.coordinator.merge_search_endpoint(
                    "/hybrid_search",
                    body,
                    ascending=False,
                    index="HYBRID-RRF",
                    fusion=body.get("fusion") or "rrf",
                )
            elif path == "/search_range":
                payload = self.coordinator.search_range(body)
            elif path == "/list_deleted_ids":
                payload = self.coordinator.list_deleted_ids(body)
            elif path in {"/head", "/tail"}:
                payload = self.coordinator.head_tail_json(path, body)
            elif path in {"/query", "/query_vectors"}:
                payload = self.coordinator.query_all_json(body, path)
            elif path == "/create_vector_field":
                payload = self.coordinator.broadcast_json(path, body)
            elif path == "/list_vector_fields":
                payload = self.coordinator.list_vector_fields(body)
            elif path in {"/add_named_vectors", "/add_sparse_vectors"}:
                payload = self.coordinator.route_vector_payloads(body, path)
            elif path in {"/commit", "/flush", "/checkpoint", "/close_collection"}:
                payload = self.coordinator.broadcast_control(path, body)
            elif path == "/build_index":
                payload = self.coordinator.broadcast_json(path, body)
                self.coordinator.state.update_collection_index(
                    body["database_name"],
                    body["collection_name"],
                    body.get("index_mode") or "FLAT-IP",
                )
            elif path == "/build_vector_field_index":
                payload = self.coordinator.broadcast_json(path, body)
            elif path == "/remove_index":
                payload = self.coordinator.broadcast_json(path, body)
                self.coordinator.state.update_collection_index(
                    body["database_name"],
                    body["collection_name"],
                    None,
                )
            elif path == "/remove_vector_field_index":
                payload = self.coordinator.broadcast_json(path, body)
            elif path == "/update_collection_description":
                payload = self.coordinator.broadcast_json(path, body)
                self.coordinator.state.update_collection_description(
                    body["database_name"],
                    body["collection_name"],
                    body.get("description"),
                )
            elif path == "/compact":
                payload = self.coordinator.compact(body)
            elif path == "/stats":
                payload = self.coordinator.stats(body)
            elif path in {
                "/collection_shape",
                "/max_id",
                "/index_mode",
                "/list_fields",
                "/read_by_only_id",
                "/get_collection_path",
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
        if path == "/get_collection_path":
            return coord.collection_paths(body)
        if path == "/read_by_only_id":
            raw_id = body.get("id")
            ids = raw_id if isinstance(raw_id, list) else [raw_id]
            by_group: dict[str, tuple[dict[str, Any], list[Any]]] = {}
            for item_id in ids:
                group = coord.state.group_for_public_id(body["database_name"], body["collection_name"], item_id)
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


def _split_csv(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _metadata_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("metadata") or config.get("metadata_store") or {}
    if isinstance(value, str):
        return {"owners": value}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _default_metadata_owners_from_config(config: dict[str, Any]) -> list[str]:
    raw_groups = config.get("shard_groups") or config.get("shards") or []
    if not raw_groups:
        return []
    primaries = [
        _normalize_uri(group["primary"])
        for group in raw_groups
        if isinstance(group, dict) and group.get("primary")
    ]
    if len(primaries) >= 3:
        return primaries[:3]
    return primaries[:1]


def build_metadata_store(
    *,
    owners: list[str] | None,
    cache_path: Path,
    timeout_secs: float,
    api_key: str | None,
) -> MetadataStore:
    owners = owners or []
    if len(owners) == 1:
        return ShardMetadataStore(
            owners[0],
            cache_path=cache_path,
            timeout_secs=timeout_secs,
            api_key=api_key,
        )
    if len(owners) >= 3:
        return QuorumMetadataStore(
            owners,
            cache_path=cache_path,
            timeout_secs=timeout_secs,
            api_key=api_key,
        )
    if not owners:
        raise ValueError(
            "cluster metadata owner could not be inferred. Provide --cluster-config "
            "with at least one shard group, or pass --metadata-owners."
        )
    raise ValueError("metadata owners must contain exactly 1 URI, or 3+ URIs for replicated metadata")


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
    coordinator_id: str | None = None,
    coordinator_uri: str | None = None,
    coordinator_lease_secs: float | None = None,
    metadata_owners: list[str] | None = None,
) -> None:
    config = load_cluster_config(cluster_config)
    metadata_cfg = _metadata_config(config)
    state_path = Path(
        cluster_state
        or config.get("state_path")
        or os.environ.get("LYNSE_CLUSTER_STATE")
        or "cluster_state.cache.json"
    )
    advertised_uri = _normalize_uri(
        coordinator_uri
        or config.get("coordinator_uri")
        or os.environ.get("LYNSE_COORDINATOR_URI")
        or _default_coordinator_uri(host, int(port))
    )
    lease_secs = float(
        coordinator_lease_secs
        if coordinator_lease_secs is not None
        else config.get(
            "coordinator_lease_secs",
            os.environ.get("LYNSE_COORDINATOR_LEASE_SECS", DEFAULT_COORDINATOR_LEASE_SECS),
        )
    )
    shard_key = shard_api_key or config.get("shard_api_key")
    effective_metadata_owners = (
        metadata_owners
        or _split_csv(config.get("metadata_owners") or config.get("metadata_owner"))
        or _split_csv(metadata_cfg.get("owners") or metadata_cfg.get("owner"))
        or _split_csv(os.environ.get("LYNSE_CLUSTER_METADATA_OWNERS"))
        or _default_metadata_owners_from_config(config)
    )
    metadata_store = build_metadata_store(
        owners=effective_metadata_owners,
        cache_path=state_path,
        timeout_secs=request_timeout_secs,
        api_key=shard_key,
    )
    state = ClusterState(state_path, seed_config=config, metadata_store=metadata_store)
    coordinator_lease = MetadataCoordinatorLease(
        metadata_store,
        coordinator_id=(
            coordinator_id
            or config.get("coordinator_id")
            or os.environ.get("LYNSE_COORDINATOR_ID")
            or advertised_uri
        ),
        coordinator_uri=advertised_uri,
        lease_secs=lease_secs,
    )
    coordinator = ClusterCoordinator(
        state,
        timeout_secs=request_timeout_secs,
        health_interval_secs=health_interval_secs,
        health_failures=health_failures,
        shard_api_key=shard_key,
        coordinator_id=(
            coordinator_id
            or config.get("coordinator_id")
            or os.environ.get("LYNSE_COORDINATOR_ID")
            or advertised_uri
        ),
        coordinator_uri=advertised_uri,
        coordinator_lease_secs=lease_secs,
        coordinator_lease=coordinator_lease,
    )
    coordinator.start_coordinator_lease_loop()
    coordinator.start_health_loop()

    class Handler(ClusterRequestHandler):
        pass

    Handler.coordinator = coordinator
    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    try:
        httpd.serve_forever()
    finally:
        coordinator.stop()
        close = getattr(metadata_store, "close", None)
        if close:
            close()
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LynseDB lightweight cluster coordinator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7637)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument(
        "--cluster-state",
        help=(
            "Local coordinator metadata cache path. Authoritative metadata is stored "
            "on the metadata owner shard(s)."
        ),
    )
    parser.add_argument("--shard-api-key")
    parser.add_argument("--request-timeout-secs", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECS)
    parser.add_argument("--health-interval-secs", type=float, default=DEFAULT_HEALTH_INTERVAL_SECS)
    parser.add_argument("--health-failures", type=int, default=DEFAULT_HEALTH_FAILURES)
    parser.add_argument("--coordinator-id")
    parser.add_argument("--coordinator-uri")
    parser.add_argument("--coordinator-lease-secs", type=float, default=None)
    parser.add_argument(
        "--metadata-owners",
        help=(
            "Comma-separated metadata owner shard HTTP URIs. Omit to infer from "
            "shard primaries; provide 3+ URIs for replicated metadata."
        ),
    )
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
        coordinator_id=args.coordinator_id,
        coordinator_uri=args.coordinator_uri,
        coordinator_lease_secs=args.coordinator_lease_secs,
        metadata_owners=_split_csv(args.metadata_owners),
    )


if __name__ == "__main__":
    main()
