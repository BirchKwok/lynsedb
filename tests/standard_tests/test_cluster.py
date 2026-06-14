import json

from lynse.cluster import (
    FIELDS_BINARY_MAGIC,
    REPLICA_ACTIVE,
    REPLICA_STALE,
    ClusterCoordinator,
    ClusterState,
    _derive_rpc_target,
    _encode_fields_binary,
    _encode_ids_for_wire,
    _encode_search_binary,
    _merge_pairs,
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


def test_rpc_target_is_derived_from_http_uri():
    assert _derive_rpc_target("http://127.0.0.1:7638") == ("127.0.0.1", 17638)
    assert _derive_rpc_target("http://example.com:60000") == ("example.com", 50000)


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
