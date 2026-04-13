"""Tests for LynseDB server CLI argument parsing."""

import json

from lynse.server import _parse_args


def test_parse_serve_with_data_dir():
    args = _parse_args(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--data-dir",
            "/tmp/lynsedb-data",
            "--api-key",
            "secret",
        ]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 9001
    assert args.data_dir == "/tmp/lynsedb-data"
    assert args.api_key == "secret"


def test_parse_run_with_root_alias():
    args = _parse_args(["run", "--root", "/tmp/legacy-root"])
    assert args.data_dir == "/tmp/legacy-root"


def test_parse_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("LYNSE_HOST", "127.0.0.2")
    monkeypatch.setenv("LYNSE_PORT", "7777")
    monkeypatch.setenv("LYNSE_DATA_DIR", "/tmp/env-data")
    monkeypatch.setenv("LYNSE_API_KEY", "env-key")

    args = _parse_args([])
    assert args.host == "127.0.0.2"
    assert args.port == 7777
    assert args.data_dir == "/tmp/env-data"
    assert args.api_key == "env-key"


def test_parse_json_config_file(tmp_path):
    config_path = tmp_path / "server.json"
    config_path.write_text(
        json.dumps(
            {
                "host": "0.0.0.0",
                "port": 9002,
                "data_dir": "/tmp/from-config",
                "api_key": "from-config-key",
                "workers": 3,
                "keep_alive_secs": 60,
                "request_timeout_secs": 120,
                "json_limit_mb": 64,
                "payload_limit_mb": 128,
                "slow_query_warn_ms": 250,
                "max_top_k": 500,
                "max_batch_vectors": 1000,
                "max_collection_vectors": 2000,
                "max_collection_vector_bytes": 4096,
                "audit_log": False,
            }
        ),
        encoding="utf-8",
    )

    args = _parse_args(["serve", "--config", str(config_path)])
    assert args.host == "0.0.0.0"
    assert args.port == 9002
    assert args.data_dir == "/tmp/from-config"
    assert args.api_key == "from-config-key"
    assert args.workers == 3
    assert args.keep_alive_secs == 60
    assert args.request_timeout_secs == 120
    assert args.json_limit_mb == 64
    assert args.payload_limit_mb == 128
    assert args.slow_query_warn_ms == 250
    assert args.max_top_k == 500
    assert args.max_batch_vectors == 1000
    assert args.max_collection_vectors == 2000
    assert args.max_collection_vector_bytes == 4096
    assert args.audit_log is False


def test_env_overrides_config(tmp_path, monkeypatch):
    config_path = tmp_path / "server.json"
    config_path.write_text(
        json.dumps({"port": 9002, "json_limit_mb": 64, "audit_log": False}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LYNSE_PORT", "9011")
    monkeypatch.setenv("LYNSE_JSON_LIMIT_MB", "96")
    monkeypatch.setenv("LYNSE_AUDIT_LOG", "true")

    args = _parse_args(["--config", str(config_path)])
    assert args.port == 9011
    assert args.json_limit_mb == 96
    assert args.audit_log is True


def test_cli_overrides_env_and_config(tmp_path, monkeypatch):
    config_path = tmp_path / "server.json"
    config_path.write_text(json.dumps({"port": 9002}), encoding="utf-8")
    monkeypatch.setenv("LYNSE_PORT", "9011")

    args = _parse_args(["--config", str(config_path), "--port", "9020"])
    assert args.port == 9020


def test_parse_server_governance_limits_from_cli():
    args = _parse_args(
        [
            "--slow-query-warn-ms",
            "0",
            "--max-top-k",
            "12",
            "--max-batch-vectors",
            "34",
            "--max-collection-vectors",
            "56",
            "--max-collection-vector-bytes",
            "789",
            "--no-audit-log",
        ]
    )
    assert args.slow_query_warn_ms == 0
    assert args.max_top_k == 12
    assert args.max_batch_vectors == 34
    assert args.max_collection_vectors == 56
    assert args.max_collection_vector_bytes == 789
    assert args.audit_log is False
