import ast
import inspect
from pathlib import Path

import numpy as np


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


def test_python_api_files_do_not_expose_kwargs_parameters():
    paths = [
        "python/lynse/api/local_client.py",
        "python/lynse/api/http_api/client_api.py",
        "python/lynse/_backend.py",
        "python/lynse/result_view.py",
        "python/lynse/execution_layer/session.py",
    ]
    offenders = []
    for path in paths:
        tree = ast.parse((ROOT / path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.args.kwarg:
                offenders.append(f"{path}:{node.lineno}:{node.name}")

    assert offenders == []


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


def test_http_build_index_ignores_non_ivf_n_clusters():
    from lynse.api.http_api.client_api import Collection

    coll = Collection("http://server", "db", "items")
    coll._session = _FakeSession()

    coll.build_index("FLAT-L2", n_clusters=128)
    _, first_kwargs = coll._session.posts[-1]
    assert "n_clusters" not in first_kwargs["json"]

    coll.build_index("IVF-L2", n_clusters=128)
    _, second_kwargs = coll._session.posts[-1]
    assert second_kwargs["json"]["n_clusters"] == 128

    coll.build_index("HNSW-L2", field_name="image", n_clusters=128)
    uri, third_kwargs = coll._session.posts[-1]
    assert uri.endswith("/build_vector_field_index")
    assert "n_clusters" not in third_kwargs["json"]


class _FakeRustCollection:
    is_read_only = False

    def __init__(self):
        self.calls = []

    def build_index(self, index_mode, field_name="default", n_clusters=None):
        self.calls.append((index_mode, field_name, n_clusters))


def test_local_build_index_ignores_non_ivf_n_clusters():
    from lynse.api.local_client import LocalCollection

    rust = _FakeRustCollection()
    coll = LocalCollection(object(), "db", "items", rust, dim=4)

    coll.build_index("FLAT-L2", n_clusters=128)
    assert rust.calls[-1] == ("FLAT-L2", "default", None)

    coll.build_index("IVF-L2", field_name="image", n_clusters=128)
    assert rust.calls[-1] == ("IVF-L2", "image", 128)


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
