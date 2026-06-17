import json
import queue
import socket
import struct
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lynse.cluster import (
    FIELDS_BINARY_MAGIC,
    REPLICA_ACTIVE,
    REPLICA_STALE,
    RPC_OP_COLLECTION_CONTROL,
    ClusterCoordinator,
    ClusterState,
    _derive_rpc_target,
    _encode_fields_binary,
    _encode_ids_for_wire,
    _encode_search_binary,
    _is_ascending_index,
    _merge_pairs,
    _shard_artifact_path,
    _split_binary_items_payload,
    _split_search_binary,
    _vectors_to_f32_bytes,
)
from lynse.api.http_api.client_api import Collection


def _seed_config():
    return {
        "bucket_count": 16,
        "shard_groups": [
            {
                "name": "sg0",
                "primary": "http://127.0.0.1:8101",
                "replicas": ["http://127.0.0.1:8102"],
            },
            {
                "name": "sg1",
                "primary": "http://127.0.0.1:8201",
                "replicas": [{"uri": "http://127.0.0.1:8202", "state": REPLICA_ACTIVE}],
            },
        ],
    }


def test_spann_cluster_merge_order_uses_metric_tokens():
    assert _is_ascending_index("SPANN-L2")
    assert _is_ascending_index("SPANN-COS")
    assert not _is_ascending_index("SPANN")


def _single_seed_config():
    return {
        "bucket_count": 16,
        "shard_groups": [
            {
                "name": "sg0",
                "primary": "http://127.0.0.1:8101",
                "replicas": [],
            },
        ],
    }


def _find_rpc_test_ports():
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            rpc_port = sock.getsockname()[1]
        if rpc_port > 10000:
            return rpc_port - 10000, rpc_port


def _read_exact(conn, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise RuntimeError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def test_cluster_state_initializes_and_routes_ids(tmp_path):
    state_path = tmp_path / "cluster_state.json"
    state = ClusterState(state_path, seed_config=_seed_config())

    coll = state.upsert_collection(
        "db",
        "docs",
        dim=4,
        description="test",
        dtypes="float32",
        drop_if_exists=False,
    )
    assert coll["bucket_count"] == 16
    assert state.databases() == ["db"]

    ids = state.allocate_ids("db", "docs", 3)
    assert ids == [0, 1, 2]
    assert state.get_collection("db", "docs")["next_id"] == 3

    group = state.group_for_id("db", "docs", ids[0])
    assert group["name"] in {"sg0", "sg1"}
    assert json.loads(state_path.read_text(encoding="utf-8"))["meta_epoch"] >= 3


def test_cluster_promote_marks_old_primary_stale(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())

    state.promote("sg0", "http://127.0.0.1:8102")

    group = state.group_by_name("sg0")
    assert group["primary"] == "http://127.0.0.1:8102"
    assert group["primary_epoch"] == 2
    assert group["replicas"] == [
        {"uri": "http://127.0.0.1:8101", "state": REPLICA_STALE}
    ]


def test_mark_replica_stale_persists(tmp_path):
    state_path = tmp_path / "cluster_state.json"
    state = ClusterState(state_path, seed_config=_seed_config())

    state.mark_replica_stale("http://127.0.0.1:8202")

    reloaded = ClusterState(state_path)
    group = reloaded.group_by_name("sg1")
    assert group["replicas"][0]["state"] == REPLICA_STALE


def test_search_merge_obeys_metric_order():
    ids, distances, fields = _merge_pairs(
        [
            ([1, 2], [0.9, 0.2], [{"a": 1}, {"a": 2}]),
            ([3, 4], [0.5, 0.8], [{"a": 3}, {"a": 4}]),
        ],
        k=3,
        ascending=False,
        return_fields=True,
    )
    assert ids == [1, 4, 3]
    assert distances == [0.9, 0.8, 0.5]
    assert fields == [{"a": 1}, {"a": 4}, {"a": 3}]

    ids, distances, _ = _merge_pairs(
        [([1], [3.0], []), ([2], [1.0], []), ([3], [2.0], [])],
        k=2,
        ascending=True,
        return_fields=False,
    )
    assert ids == [2, 3]
    assert distances == [1.0, 2.0]


def test_search_binary_roundtrip():
    encoded = _encode_search_binary([10, 20], [0.5, 0.25], [{"x": 1}, {"x": 2}])
    ids, distances, fields, offset = _split_search_binary(encoded)

    assert ids == [10, 20]
    assert distances == [0.5, 0.25]
    assert fields == [{"x": 1}, {"x": 2}]
    assert offset == len(encoded)


def test_search_binary_uses_rust_read_coordinator_when_available(tmp_path, monkeypatch):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)
    raw_response = _encode_search_binary([11], [0.75])
    calls = []

    class FakeRustReadCoordinator:
        def __init__(self, state_path, timeout_secs, api_key):
            calls.append(("init", state_path, timeout_secs, api_key))

        def search_binary(self, meta_json, body):
            calls.append(("search_binary", json.loads(meta_json), body))
            return raw_response

    monkeypatch.setitem(
        sys.modules,
        "lynse._core",
        types.SimpleNamespace(ClusterReadCoordinator=FakeRustReadCoordinator),
    )

    coord = ClusterCoordinator(state, timeout_secs=3, shard_api_key="secret")
    try:
        payload = coord.search_binary(
            {
                "database_name": "db",
                "collection_name": "docs",
                "dim": "2",
                "k": "1",
            },
            b"query",
        )
    finally:
        coord.stop()

    assert payload == raw_response
    assert calls[0] == ("init", str(state.path), 3.0, "secret")
    assert calls[1][0] == "search_binary"
    assert calls[1][1]["index_mode"] == "FLAT-IP"
    assert calls[1][2] == b"query"


def test_batch_search_binary_does_not_fallback_when_rust_read_fails(tmp_path, monkeypatch):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)
    fallback_calls = []

    class FailingRustReadCoordinator:
        def __init__(self, state_path, timeout_secs, api_key):
            pass

        def batch_search_binary(self, meta_json, body):
            raise RuntimeError("rust read failed")

    class NoFallbackCoordinator(ClusterCoordinator):
        def _binary_call(self, uri, rpc_op, http_path, params, body):
            fallback_calls.append((uri, rpc_op, http_path, params, body))
            return b""

    monkeypatch.setitem(
        sys.modules,
        "lynse._core",
        types.SimpleNamespace(ClusterReadCoordinator=FailingRustReadCoordinator),
    )

    coord = NoFallbackCoordinator(state)
    try:
        try:
            coord.batch_search_binary(
                {
                    "database_name": "db",
                    "collection_name": "docs",
                    "dim": "2",
                    "n_queries": "1",
                    "k": "1",
                },
                b"query",
            )
        except RuntimeError as exc:
            assert str(exc) == "rust read failed"
        else:
            raise AssertionError("expected Rust read error")
    finally:
        coord.stop()

    assert fallback_calls == []


def test_rpc_target_is_derived_from_http_uri():
    assert _derive_rpc_target("http://127.0.0.1:7638") == ("127.0.0.1", 17638)
    assert _derive_rpc_target("http://example.com:60000") == ("example.com", 50000)


def test_rpc_request_reuses_idle_socket(tmp_path):
    http_port, rpc_port = _find_rpc_test_ports()
    ready = threading.Event()
    accepted = []

    def serve():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", rpc_port))
            srv.listen(1)
            ready.set()
            conn, _addr = srv.accept()
            accepted.append(1)
            with conn:
                for _ in range(2):
                    size = struct.unpack("<I", _read_exact(conn, 4))[0]
                    _read_exact(conn, size)
                    meta = b'{"ok":true}'
                    frame = bytes([0]) + struct.pack("<I", len(meta)) + meta
                    conn.sendall(struct.pack("<I", len(frame)) + frame)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)

    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    coord = ClusterCoordinator(state, timeout_secs=2)
    try:
        uri = f"http://127.0.0.1:{http_port}"
        assert coord._rpc_request(uri, 1, {})[0] == {"ok": True}
        assert coord._rpc_request(uri, 1, {})[0] == {"ok": True}
    finally:
        coord.stop()

    thread.join(timeout=2)
    assert accepted == [1]


def test_vectors_to_f32_bytes_reports_shape():
    raw, n_vectors, dim = _vectors_to_f32_bytes([[1, 2], [3, 4]])

    assert len(raw) == 4 * 4
    assert n_vectors == 2
    assert dim == 2


def test_binary_item_fields_preserve_add_and_upsert_semantics():
    add_fields = ClusterCoordinator._fields_for_binary_items(
        "/add",
        [
            {"id": 1, "vector": [1.0], "field": {}},
            {"id": 2, "vector": [2.0], "field": {"tag": "x"}},
        ],
    )
    assert add_fields == [{}, {"tag": "x"}]

    no_add_fields = ClusterCoordinator._fields_for_binary_items(
        "/add",
        [
            {"id": 1, "vector": [1.0], "field": {}},
            {"id": 2, "vector": [2.0]},
        ],
    )
    assert no_add_fields is None

    upsert_fields = ClusterCoordinator._fields_for_binary_items(
        "/upsert",
        [
            {"id": 1, "vector": [1.0]},
            {"id": 2, "vector": [2.0], "field": {}},
            {"id": 3, "vector": [3.0], "field": {"tag": "x"}},
        ],
    )
    assert upsert_fields == [None, {}, {"tag": "x"}]


def test_fields_binary_payload_is_not_json():
    payload = _encode_fields_binary([{"tag": "x", "score": 1.5}, None, {}])

    assert payload.startswith(FIELDS_BINARY_MAGIC)
    assert not payload.startswith(b"[")


def test_binary_items_payload_supports_range_ids_and_fields():
    vectors_raw, n_vectors, dim = _vectors_to_f32_bytes([[1, 2], [3, 4], [5, 6]])
    id_raw, id_params = _encode_ids_for_wire([10, 11, 12])
    fields = [{"tag": "a"}, None, {}]
    payload = vectors_raw + id_raw + _encode_fields_binary(fields)

    vector_part, ids, decoded_fields, offset = _split_binary_items_payload(
        payload,
        n_vectors,
        dim,
        "float32",
        id_params["ids_encoding"],
        id_params.get("ids_start"),
    )

    assert bytes(vector_part) == vectors_raw
    assert ids == [10, 11, 12]
    assert decoded_fields == fields
    assert offset == len(vectors_raw)


def test_binary_items_payload_supports_raw_ids():
    vectors_raw, n_vectors, dim = _vectors_to_f32_bytes([[1], [2], [3]])
    id_raw, id_params = _encode_ids_for_wire([3, 9, 4])
    payload = vectors_raw + id_raw

    _vector_part, ids, fields, offset = _split_binary_items_payload(
        payload,
        n_vectors,
        dim,
        "float32",
        id_params["ids_encoding"],
        id_params.get("ids_start"),
    )

    assert id_params == {"ids_encoding": "raw"}
    assert ids == [3, 9, 4]
    assert fields is None
    assert offset == len(vectors_raw) + len(id_raw)


def test_bulk_add_binary_allocates_global_ids_and_routes_vectors(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)

    class RecordingCoordinator(ClusterCoordinator):
        def __init__(self, state):
            super().__init__(state)
            self.calls = []

        def write_group_binary_payload(
            self,
            group,
            path,
            database_name,
            collection_name,
            dim,
            vector_encoding,
            vector_raw,
            ids,
            fields,
        ):
            self.calls.append({
                "group": group["name"],
                "path": path,
                "database_name": database_name,
                "collection_name": collection_name,
                "dim": dim,
                "vector_encoding": vector_encoding,
                "vector_raw": bytes(vector_raw),
                "ids": list(ids),
                "fields": fields,
            })
            return {"status": "success", "params": {"ids": ids}}

    coord = RecordingCoordinator(state)
    try:
        raw, n_vectors, dim = _vectors_to_f32_bytes([[1, 2], [3, 4], [5, 6], [7, 8]])
        payload = coord.route_bulk_add_binary(
            {
                "database_name": "db",
                "collection_name": "docs",
                "dim": dim,
                "n_vectors": n_vectors,
                "return_ids": "true",
            },
            raw,
        )
    finally:
        coord.stop()

    assert payload["params"]["ids"] == [0, 1, 2, 3]
    assert sorted(item_id for call in coord.calls for item_id in call["ids"]) == [0, 1, 2, 3]
    assert sum(len(call["vector_raw"]) for call in coord.calls) == len(raw)
    assert {call["path"] for call in coord.calls} == {"/add_binary_ids"}


def test_batch_search_json_merges_each_query(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)

    class SearchCoordinator(ClusterCoordinator):
        def _json_post(self, uri, path, body):
            assert path == "/batch_search"
            if uri.endswith("8101"):
                results = [
                    {"ids": [1], "scores": [0.9], "fields": [{"s": 1}]},
                    {"ids": [2], "scores": [0.1], "fields": [{"s": 2}]},
                ]
            elif uri.endswith("8201"):
                results = [
                    {"ids": [3], "scores": [0.8], "fields": [{"s": 3}]},
                    {"ids": [4], "scores": [0.7], "fields": [{"s": 4}]},
                ]
            else:
                results = []
            return {"status": "success", "params": {"results": results}}

    coord = SearchCoordinator(state)
    try:
        payload = coord.batch_search_json({
            "database_name": "db",
            "collection_name": "docs",
            "vectors": [[1, 0], [0, 1]],
            "k": 1,
            "return_fields": True,
        })
    finally:
        coord.stop()

    assert payload["params"]["results"] == [
        {"ids": [1], "scores": [0.9], "fields": [{"s": 1}]},
        {"ids": [4], "scores": [0.7], "fields": [{"s": 4}]},
    ]


def test_async_json_fanout_reads_real_http_shards(tmp_path):
    class SearchHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_POST(self):
            assert self.path == "/search"
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            payload = json.dumps({
                "status": "success",
                "params": {
                    "items": {
                        "ids": [self.server.item_id],
                        "scores": [self.server.score],
                        "fields": [],
                    }
                },
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    servers = []
    threads = []
    groups = []
    for idx, score in [(10, 0.9), (20, 0.8)]:
        server = ThreadingHTTPServer(("127.0.0.1", 0), SearchHandler)
        server.item_id = idx
        server.score = score
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        groups.append({
            "name": f"sg{idx}",
            "primary": f"http://127.0.0.1:{server.server_address[1]}",
            "replicas": [],
        })

    state = ClusterState(
        tmp_path / "cluster_state.json",
        seed_config={"bucket_count": 16, "shard_groups": groups},
    )
    state.upsert_collection("db", "docs", 2, None, "float32", False)
    coord = ClusterCoordinator(state, timeout_secs=2)
    try:
        payload = coord.search_json({
            "database_name": "db",
            "collection_name": "docs",
            "vector": [1, 0],
            "k": 2,
            "return_fields": False,
        })
    finally:
        coord.stop()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)

    assert payload["params"]["items"]["ids"] == [10, 20]
    assert payload["params"]["items"]["scores"] == [0.9, 0.8]


def test_single_shard_binary_search_passthrough(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_single_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)
    raw_response = _encode_search_binary([42], [0.25])

    class PassthroughCoordinator(ClusterCoordinator):
        def __init__(self, state):
            super().__init__(state)
            self.calls = []

        def _binary_call(self, uri, rpc_op, http_path, params, body):
            self.calls.append((uri, rpc_op, http_path, params, body))
            return raw_response

    coord = PassthroughCoordinator(state)
    try:
        payload = coord.search_binary(
            {
                "database_name": "db",
                "collection_name": "docs",
                "dim": "2",
                "k": "1",
            },
            b"query",
        )
    finally:
        coord.stop()

    assert payload == raw_response
    assert coord.calls == [
        (
            "http://127.0.0.1:8101",
            2,
            "/search_binary",
            {
                "database_name": "db",
                "collection_name": "docs",
                "dim": "2",
                "k": "1",
            },
            b"query",
        )
    ]


def test_internal_integer_ids_delete_routes_by_internal_hash(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)
    state.mark_integer_id_routing("db", "docs", "internal")

    class DeleteCoordinator(ClusterCoordinator):
        def __init__(self, state):
            super().__init__(state)
            self.rpc_calls = []

        def _can_rpc(self, uri):
            return True

        def _rpc_request(self, uri, op, meta=None, raw=b""):
            self.rpc_calls.append((uri, op, dict(meta or {}), raw))
            return {"ok": True}, b""

    coord = DeleteCoordinator(state)
    try:
        payload = coord.route_ids_write(
            {"database_name": "db", "collection_name": "docs", "ids": [0, 1, 2]},
            "/delete",
        )
    finally:
        coord.stop()

    assert payload["status"] == "success"
    routed_ids = {item_id for _uri, _op, meta, _raw in coord.rpc_calls for item_id in meta["ids"]}
    assert routed_ids == {0, 1, 2}
    assert {op for _uri, op, _meta, _raw in coord.rpc_calls} == {6}


def test_write_group_control_uses_internal_rpc(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    group = state.group_by_name("sg0")

    class ControlCoordinator(ClusterCoordinator):
        def __init__(self, state):
            super().__init__(state)
            self.rpc_calls = []
            self.http_calls = []

        def _can_rpc(self, uri):
            return True

        def _rpc_request(self, uri, op, meta=None, raw=b""):
            self.rpc_calls.append((uri, op, dict(meta or {}), raw))
            return {"ok": True}, b""

        def _json_post(self, uri, path, body):
            self.http_calls.append((uri, path, body))
            return {"status": "success", "params": {}}

    coord = ControlCoordinator(state)
    try:
        payload = coord.write_group_control(
            group,
            "/checkpoint",
            {"database_name": "db", "collection_name": "docs"},
        )
    finally:
        coord.stop()

    assert payload["status"] == "success"
    assert not coord.http_calls
    assert {op for _uri, op, _meta, _raw in coord.rpc_calls} == {RPC_OP_COLLECTION_CONTROL}
    assert {meta["action"] for _uri, _op, meta, _raw in coord.rpc_calls} == {"checkpoint"}
    assert {meta["database_name"] for _uri, _op, meta, _raw in coord.rpc_calls} == {"db"}
    assert {meta["collection_name"] for _uri, _op, meta, _raw in coord.rpc_calls} == {"docs"}


def test_list_vector_fields_merges_by_name(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)

    class FieldCoordinator(ClusterCoordinator):
        def _json_post(self, uri, path, body):
            assert path == "/list_vector_fields"
            fields = [{"name": "default", "dim": 2}, {"name": "image", "dim": 3}]
            if uri.endswith("8201"):
                fields.append({"name": "text", "dim": 4})
            return {"status": "success", "params": {"fields": fields}}

    coord = FieldCoordinator(state)
    try:
        payload = coord.list_vector_fields({"database_name": "db", "collection_name": "docs"})
    finally:
        coord.stop()

    assert {field["name"] for field in payload["params"]["fields"]} == {"default", "image", "text"}


def test_internal_integer_ids_route_vector_payloads_by_internal_hash(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)
    state.mark_integer_id_routing("db", "docs", "internal")

    class VectorCoordinator(ClusterCoordinator):
        def __init__(self, state):
            super().__init__(state)
            self.calls = []

        def write_group_json(self, group, path, body, require_primary=True):
            self.calls.append((group["name"], path, list(body["ids"])))
            return {"status": "success", "params": {}}

    coord = VectorCoordinator(state)
    try:
        payload = coord.route_vector_payloads(
            {
                "database_name": "db",
                "collection_name": "docs",
                "field_name": "image",
                "ids": [0, 1, 2],
                "vectors": [[1.0], [2.0], [3.0]],
            },
            "/add_named_vectors",
        )
    finally:
        coord.stop()

    expected = {}
    for item_id in [0, 1, 2]:
        group = state.group_for_id("db", "docs", item_id)["name"]
        expected.setdefault(group, []).append(item_id)

    assert payload["params"]["ids"] == [0, 1, 2]
    assert {group: ids for group, _path, ids in coord.calls} == expected
    assert {path for _group, path, _ids in coord.calls} == {"/add_named_vectors"}


def test_stats_merges_shard_counts(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())
    state.upsert_collection("db", "docs", 2, None, "float32", False)

    class StatsCoordinator(ClusterCoordinator):
        def _json_post(self, uri, path, body):
            assert path == "/stats"
            if uri.endswith("8101"):
                stats = {"n_vectors": 2, "n_live": 1, "n_tombstoned": 1, "dimension": 2, "max_id": 8}
            else:
                stats = {"n_vectors": 3, "n_live": 3, "n_tombstoned": 0, "dimension": 2, "max_id": 9}
            return {"status": "success", "params": {"stats": stats}}

    coord = StatsCoordinator(state)
    try:
        payload = coord.stats({"database_name": "db", "collection_name": "docs"})
    finally:
        coord.stop()

    assert payload["params"]["stats"]["n_vectors"] == 5
    assert payload["params"]["stats"]["n_live"] == 4
    assert payload["params"]["stats"]["n_tombstoned"] == 1
    assert payload["params"]["stats"]["max_id"] == 9


def test_artifact_fanout_uses_shard_specific_paths(tmp_path):
    assert _shard_artifact_path("/tmp/docs.snap", "sg0") == "/tmp/docs.sg0.snap"

    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())

    class SnapshotCoordinator(ClusterCoordinator):
        def __init__(self, state):
            super().__init__(state)
            self.paths = []

        def _json_post(self, uri, path, body):
            assert path == "/snapshot_database"
            self.paths.append(body["snapshot_path"])
            return {"status": "success", "params": {"snapshot_path": body["snapshot_path"]}}

    coord = SnapshotCoordinator(state)
    try:
        payload = coord.artifact_fanout(
            "/snapshot_database",
            {"database_name": "db", "snapshot_path": "/tmp/db.snap"},
            "snapshot_path",
        )
    finally:
        coord.stop()

    assert sorted(coord.paths) == ["/tmp/db.sg0.snap", "/tmp/db.sg1.snap"]
    assert {shard["group"] for shard in payload["params"]["shards"]} == {"sg0", "sg1"}


def test_collection_paths_return_shard_manifest(tmp_path):
    state = ClusterState(tmp_path / "cluster_state.json", seed_config=_seed_config())

    class PathCoordinator(ClusterCoordinator):
        def _json_post(self, uri, path, body):
            assert path == "/get_collection_path"
            return {
                "status": "success",
                "params": {"collection_path": f"/data/{uri.rsplit(':', 1)[-1]}/docs"},
            }

    coord = PathCoordinator(state)
    try:
        payload = coord.collection_paths({"database_name": "db", "collection_name": "docs"})
    finally:
        coord.stop()

    assert payload["params"]["collection_path"] is None
    assert {item["group"] for item in payload["params"]["shards"]} == {"sg0", "sg1"}
    assert all(item["collection_path"].endswith("/docs") for item in payload["params"]["shards"])


def test_client_binary_item_preparer_omits_contiguous_ids():
    payload, n_vectors, dim, ids, id_params = Collection._prepare_binary_items(
        [([1.0, 2.0], 7), ([3.0, 4.0], 8)],
        upsert=False,
    )

    assert n_vectors == 2
    assert dim == 2
    assert ids == [7, 8]
    assert id_params == {
        "ids_encoding": "range",
        "ids_start": 7,
        "vector_encoding": "float32",
    }
    assert len(payload) == n_vectors * dim * 4


def test_client_external_integer_routing_disables_binary_fast_path():
    coll = object.__new__(Collection)
    coll._cluster_mode = True
    coll._integer_id_routing = "external"
    coll._binary_integer_id_safe = False

    assert not coll._can_write_cluster_binary_integer_ids([1, 2, 3])


def test_client_batch_search_uses_binary_when_integer_ids_are_safe():
    raw = (1).to_bytes(4, "little") + _encode_search_binary([7], [0.5])

    class Response:
        status_code = 200
        content = raw

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, uri, **kwargs):
            self.calls.append((uri, kwargs))
            return Response()

    coll = object.__new__(Collection)
    coll._uri = "http://127.0.0.1:7637"
    coll._database_name = "db"
    coll._collection_name = "docs"
    coll._session = Session()
    coll._binary_integer_id_safe = True
    coll._result_index_mode = lambda vector_field="default": "FLAT-IP"

    results = Collection.batch_search(coll, [[1.0, 2.0]], k=1)

    assert len(results) == 1
    assert results[0].ids.tolist() == [7]
    assert coll._session.calls[0][0] == "http://127.0.0.1:7637/batch_search_binary"
    assert coll._session.calls[0][1]["params"]["n_queries"] == 1


def test_client_cluster_auto_ids_uses_bulk_return_ids_without_max_id():
    class Response:
        status_code = 200

        def json(self):
            return {"status": "success", "params": {"ids": [0, 1]}}

    class Session:
        def __init__(self):
            self.calls = []

        def post(self, uri, **kwargs):
            self.calls.append((uri, kwargs))
            if uri.endswith("/max_id"):
                raise AssertionError("cluster auto-id bulk add should not call max_id")
            return Response()

    coll = object.__new__(Collection)
    coll._uri = "http://127.0.0.1:7637"
    coll._database_name = "db"
    coll._collection_name = "docs"
    coll._session = Session()
    coll._cluster_mode = True
    coll._integer_id_routing = None
    coll._binary_integer_id_safe = False
    coll._default_index = None
    coll._default_index_built = False
    coll._init_params = {}
    coll.COMMIT_FLAG = True

    returned = Collection.add(coll, ids=None, vectors=[[1.0, 2.0], [3.0, 4.0]])

    assert returned == [0, 1]
    assert coll._integer_id_routing == "internal"
    assert coll._binary_integer_id_safe is True
    assert coll._session.calls[0][0] == "http://127.0.0.1:7637/bulk_add_binary"
    assert coll._session.calls[0][1]["params"]["return_ids"] == "true"


def test_client_commit_noops_when_already_clean():
    class Session:
        def post(self, *args, **kwargs):
            raise AssertionError("clean commit should not perform an HTTP request")

    coll = object.__new__(Collection)
    coll._database_name = "db"
    coll._collection_name = "docs"
    coll._session = Session()
    coll._mesosphere_list = queue.Queue()
    coll.COMMIT_FLAG = True

    payload = Collection.commit(coll)

    assert payload["status"] == "success"
    assert payload["params"]["result"]["database_name"] == "db"


def test_client_binary_item_preparer_supports_float16_wire_dtype():
    payload, n_vectors, dim, ids, id_params = Collection._prepare_binary_items(
        [([1.0, 2.0], 7), ([3.0, 4.0], 8)],
        upsert=False,
        wire_dtype="float16",
    )

    assert n_vectors == 2
    assert dim == 2
    assert ids == [7, 8]
    assert id_params["vector_encoding"] == "float16"
    assert len(payload) == n_vectors * dim * 2
