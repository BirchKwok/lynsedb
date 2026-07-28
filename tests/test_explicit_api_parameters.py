import ast
import inspect
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _signature_from_ast(function_node):
    args = function_node.args
    parts = []
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for arg, default in zip(positional, defaults):
        if arg.arg == "self":
            continue
        text = arg.arg
        if default is not None:
            text += "=" + ast.unparse(default)
        parts.append(text)

    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if default is not None:
            text += "=" + ast.unparse(default)
        parts.append(text)

    if args.kwarg:
        parts.append("**" + args.kwarg.arg)

    return "(" + ", ".join(parts) + ")"


def _class_public_method_signatures(path, class_name):
    tree = ast.parse((ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name: _signature_from_ast(child)
                for child in node.body
                if isinstance(child, ast.FunctionDef) and not child.name.startswith("_")
            }
    raise AssertionError(f"{class_name} not found in {path}")


def test_python_api_kwargs_only_on_build_index():
    """Prefer explicit parameters; ``build_index`` is the exception for index-family kwargs."""
    paths = [
        "python/lynse/api/local_client.py",
        "python/lynse/api/http_api/client_api.py",
        "python/lynse/_backend.py",
        "python/lynse/result_view.py",
        "python/lynse/execution_layer/session.py",
    ]
    allowed = {
        "python/lynse/api/local_client.py:build_index",
        "python/lynse/api/http_api/client_api.py:build_index",
        "python/lynse/_backend.py:build_index",
    }
    offenders = []
    for path in paths:
        tree = ast.parse((ROOT / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.args.kwarg:
                key = f"{path}:{node.name}"
                if key not in allowed:
                    offenders.append(f"{path}:{node.lineno}:{node.name}")

    assert offenders == []
    # Ensure the intended kwargs surfaces still exist.
    for key in allowed:
        path, name = key.split(":")
        tree = ast.parse((ROOT / path).read_text())
        found = any(
            isinstance(node, ast.FunctionDef)
            and node.name == name
            and node.args.kwarg is not None
            for node in ast.walk(tree)
        )
        assert found, f"expected **kwargs on {key}"


def test_local_and_http_collection_common_signatures_match():
    local = _class_public_method_signatures(
        "python/lynse/api/local_client.py", "LocalCollection"
    )
    http = _class_public_method_signatures(
        "python/lynse/api/http_api/client_api.py", "Collection"
    )

    mismatches = {
        name: (local[name], http[name])
        for name in sorted(set(local) & set(http))
        if local[name] != http[name]
    }

    assert mismatches == {}


class _FakeResponse:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "success"}


class _FakeErrorResponse:
    def __init__(self, *, payload=None, text="fallback", json_error=None):
        self._payload = payload
        self.text = text
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_raise_error_response_preserves_json_error_detail():
    from lynse.api.http_api.client_api import ExecutionError, raise_error_response

    payload = {"error": "dimension mismatch", "expected": 8, "actual": 4}
    with pytest.raises(ExecutionError) as exc_info:
        raise_error_response(_FakeErrorResponse(payload=payload, text="generic error"))

    assert exc_info.value.args == (payload,)


def test_raise_error_response_falls_back_to_text_for_non_json_response():
    from lynse.api.http_api.client_api import ExecutionError, raise_error_response

    with pytest.raises(ExecutionError, match="upstream unavailable"):
        raise_error_response(
            _FakeErrorResponse(
                text="upstream unavailable",
                json_error=ValueError("not JSON"),
            )
        )


def test_http_add_without_ids_rejects_empty_vector_matrix_before_request():
    from lynse.api.http_api.client_api import Collection

    coll = Collection("http://server", "db", "items")
    coll._session = _FakeSession()

    with pytest.raises(ValueError, match="vectors cannot be empty"):
        coll.add(vectors=np.empty((0, 8), dtype=np.float32))

    assert coll._session.posts == []


class _FakeSession:
    def __init__(self):
        self.posts = []

    def post(self, uri, json=None, params=None, content=None, headers=None):
        self.posts.append(
            (
                uri,
                {
                    "json": json,
                    "params": params,
                    "content": content,
                    "headers": headers,
                },
            )
        )
        return _FakeResponse()


class _FakeSearchResult:
    ids = np.array([7], dtype=np.int64)
    distances = np.array([0.25], dtype=np.float32)
    distance_metric = "L2"
    index_type = "FLAT"

    def __len__(self):
        return len(self.ids)


def test_http_build_index_kwargs():
    from lynse.api.http_api.client_api import Collection

    coll = Collection("http://server", "db", "items")
    coll._session = _FakeSession()

    coll.build_index("FLAT-L2", n_clusters=128)
    _, first_kwargs = coll._session.posts[-1]
    # Non-IVF n_clusters is still forwarded in params; server/index filters it.
    assert first_kwargs["json"]["params"]["n_clusters"] == 128

    coll.build_index("IVF-L2", n_clusters=128)
    _, second_kwargs = coll._session.posts[-1]
    assert second_kwargs["json"]["n_clusters"] == 128
    assert second_kwargs["json"]["params"]["n_clusters"] == 128

    coll.build_index("SPANN-L2", n_clusters=64)
    _, spann_kwargs = coll._session.posts[-1]
    assert spann_kwargs["json"]["n_clusters"] == 64

    coll.build_index("HNSW-L2", field_name="image", m=16, ef_construction=64)
    uri, third_kwargs = coll._session.posts[-1]
    assert uri.endswith("/build_vector_field_index")
    assert third_kwargs["json"]["params"]["m"] == 16
    assert third_kwargs["json"]["params"]["ef_construction"] == 64


class _FakeRustCollection:
    is_read_only = False
    vector_dtype = "float32"
    shape = (0, 2)

    def __init__(self):
        self.calls = []

    def build_index(self, index_mode, field_name="default", **kwargs):
        self.calls.append((index_mode, field_name, dict(kwargs)))

    def max_id(self):
        return -1

    def add_items(self, vectors, ids, fields):
        self.calls.append(("add_items", vectors.copy(), list(ids), fields))

    def add_records(self, vectors, ids, fields):
        self.calls.append(("add_records", vectors.copy(), list(ids), fields))
        return list(ids)

    def upsert(self, ids, vectors, fields):
        self.calls.append(("upsert", list(ids), vectors.copy(), fields))

    def is_external_id_exists(self, id):
        return False

    def search(self, vector, **kwargs):
        self.calls.append(("search", vector.copy(), kwargs))
        return _FakeSearchResult()

    def batch_search(self, vectors, **kwargs):
        self.calls.append(("batch_search", vectors.copy(), kwargs))
        return [_FakeSearchResult() for _ in range(vectors.shape[0])]

    def retrieve_fields(self, ids):
        self.calls.append(("retrieve_fields", list(ids)))
        return []

    def external_ids(self, ids):
        return list(ids)


def test_local_build_index_kwargs():
    from lynse.api.local_client import LocalCollection

    rust = _FakeRustCollection()
    coll = LocalCollection(object(), "db", "items", rust, dim=4)

    coll.build_index("FLAT-L2", n_clusters=128)
    assert rust.calls[-1] == ("FLAT-L2", "default", {"n_clusters": 128})

    coll.build_index("IVF-L2", field_name="image", n_clusters=128)
    assert rust.calls[-1] == ("IVF-L2", "image", {"n_clusters": 128})

    coll.build_index("SPANN-L2", n_clusters=64)
    assert rust.calls[-1] == ("SPANN-L2", "default", {"n_clusters": 64})

    coll.build_index("HNSW-IP", m=8, ef_construction=64)
    assert rust.calls[-1] == ("HNSW-IP", "default", {"m": 8, "ef_construction": 64})


def test_local_wire_dtype_is_accepted_without_changing_local_float32_path():
    from lynse.api.local_client import LocalCollection

    rust = _FakeRustCollection()
    coll = LocalCollection(object(), "db", "items", rust, dim=2)

    coll.add(
        ids=None,
        vectors=np.array([[1.0, 2.0]], dtype=np.float64),
        batch_size=1,
        wire_dtype="float16",
    )
    op, vectors, ids, fields = rust.calls[-1]
    assert op == "add_items"
    assert vectors.dtype == np.float32
    assert ids == [0]
    assert fields is None

    coll.upsert(ids=42, vectors=[3.0, 4.0], wire_dtype="float16")
    op, vectors, ids, fields = rust.calls[-1]
    assert op == "add_records"
    assert ids == [42]
    assert vectors.dtype == np.float32
    assert fields is None

    coll.search(np.array([5.0, 6.0], dtype=np.float64), k=1, wire_dtype="float16")
    op, vector, kwargs = rust.calls[-1]
    assert op == "search"
    assert vector.dtype == np.float32
    assert kwargs["k"] == 1

    coll.batch_search([[7.0, 8.0], [9.0, 10.0]], k=1, wire_dtype="float16")
    op, vectors, kwargs = rust.calls[-1]
    assert op == "batch_search"
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    assert kwargs["k"] == 1


def test_search_and_to_json_signatures_are_explicit():
    from lynse.api.http_api.client_api import Collection as HttpCollection
    from lynse.api.local_client import LocalCollection
    from lynse.result_view import ResultView

    for func in (HttpCollection.search, LocalCollection.search, ResultView.to_json):
        signature = inspect.signature(func)
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    view = ResultView(
        ids=np.array([1], dtype=np.int64),
        distances=np.array([0.5], dtype=np.float32),
        result_type="search",
    )
    assert "\n" in view.to_json(indent=2)
