from __future__ import annotations

"""
LynseDB Server CLI.

Launch the Rust-based HTTP server for LynseDB.
This replaces the old Flask-based server.

Usage:
    python -m lynse.server                          # default: 127.0.0.1:7637
    python -m lynse.server serve --host 0.0.0.0 --port 8080 --data-dir /data/lynsedb
    lynse serve --host 0.0.0.0 --port 7637 --api-key secret
"""

import argparse
import configparser
import json
import os
import sys


def _default_port() -> int:
    return _resolve_int(
        option_name="port",
        env_names=("LYNSE_PORT", "PORT"),
        config=None,
        default=7637,
    )


def _default_data_dir() -> str:
    return os.environ.get("LYNSE_DATA_DIR") or os.environ.get("LYNSE_ROOT") or "."


def _get_config_value(config: dict, key: str):
    if not isinstance(config, dict):
        return None
    if key in config:
        return config.get(key)
    server_cfg = config.get("server")
    if isinstance(server_cfg, dict):
        return server_cfg.get(key)
    return None


def _load_config(path: str | None) -> dict:
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        raise SystemExit(f"Failed to read config file: {path}") from exc

    data = None
    lower_path = path.lower()
    if lower_path.endswith(".json"):
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON config file: {path}") from exc
    else:
        parser = configparser.ConfigParser()
        parser.optionxform = str
        try:
            parser.read_string(content)
        except configparser.Error as exc:
            raise SystemExit(f"Invalid INI config file: {path}") from exc
        data = {}
        if parser.defaults():
            data.update(parser.defaults())
        for section in parser.sections():
            data[section] = dict(parser[section])

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit("Config file must contain an object/map at top level.")
    return data


def _resolve_str(
        option_name: str,
        env_names: tuple[str, ...],
        config: dict | None,
        default: str | None,
) -> str | None:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value is not None and value != "":
            return value

    cfg_value = _get_config_value(config or {}, option_name)
    if cfg_value is not None:
        return str(cfg_value)

    return default


def _resolve_int(
        option_name: str,
        env_names: tuple[str, ...],
        config: dict | None,
        default: int | None,
) -> int | None:
    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid {env_name} value: {raw!r}") from exc
        if value <= 0:
            raise SystemExit(f"Invalid {env_name} value: {raw!r} (must be > 0)")
        return value

    cfg_value = _get_config_value(config or {}, option_name)
    if cfg_value is not None:
        try:
            value = int(cfg_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid config value for {option_name!r}: {cfg_value!r}"
            ) from exc
        if value <= 0:
            raise SystemExit(
                f"Invalid config value for {option_name!r}: {cfg_value!r} (must be > 0)"
            )
        return value

    return default


def _resolve_non_negative_int(
        option_name: str,
        env_names: tuple[str, ...],
        config: dict | None,
        default: int,
) -> int:
    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid {env_name} value: {raw!r}") from exc
        if value < 0:
            raise SystemExit(f"Invalid {env_name} value: {raw!r} (must be >= 0)")
        return value

    cfg_value = _get_config_value(config or {}, option_name)
    if cfg_value is not None:
        try:
            value = int(cfg_value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Invalid config value for {option_name!r}: {cfg_value!r}"
            ) from exc
        if value < 0:
            raise SystemExit(
                f"Invalid config value for {option_name!r}: {cfg_value!r} (must be >= 0)"
            )
        return value

    return default


def _resolve_bool(
        option_name: str,
        env_names: tuple[str, ...],
        config: dict | None,
        default: bool,
) -> bool:
    def parse_bool(raw):
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(raw)

    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            return parse_bool(raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid {env_name} value: {raw!r}") from exc

    cfg_value = _get_config_value(config or {}, option_name)
    if cfg_value is not None:
        try:
            return parse_bool(cfg_value)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid config value for {option_name!r}: {cfg_value!r}"
            ) from exc

    return default


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Support `lynse serve ...`, `lynse run ...`, and direct-flag style.
    if argv[:1] in (["serve"], ["run"]):
        argv = argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    pre_args, _ = pre_parser.parse_known_args(argv)
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="LynseDB HTTP Server (Rust)")
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="Path to JSON/INI server config file",
    )
    parser.add_argument(
        "--role",
        choices=("single", "coordinator"),
        default=_resolve_str("role", ("LYNSE_ROLE",), config, "single"),
        help="Run as a normal single-node shard or as a lightweight cluster coordinator",
    )
    parser.add_argument(
        "--host",
        default=_resolve_str("host", ("LYNSE_HOST",), config, "127.0.0.1"),
        help="Bind address (default: 127.0.0.1 or LYNSE_HOST)",
    )
    parser.add_argument(
        "--port",
        type=_positive_int,
        default=_resolve_int("port", ("LYNSE_PORT", "PORT"), config, _default_port()),
        help="Port (default: 7637, or LYNSE_PORT / PORT, or config)",
    )
    parser.add_argument(
        "--data-dir",
        "--root",
        dest="data_dir",
        default=_resolve_str(
            "data_dir",
            ("LYNSE_DATA_DIR", "LYNSE_ROOT"),
            config,
            _resolve_str("root", (), config, _default_data_dir()),
        ),
        help="Root data directory (default: current dir or LYNSE_DATA_DIR / LYNSE_ROOT, or config)",
    )
    parser.add_argument(
        "--api-key",
        default=_resolve_str("api_key", ("LYNSE_API_KEY",), config, None),
        help="Optional API key for Bearer / Basic auth (default: LYNSE_API_KEY, or config)",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=_resolve_int("workers", ("LYNSE_SERVER_WORKERS",), config, None),
        help="HTTP worker threads (default: auto)",
    )
    parser.add_argument(
        "--keep-alive-secs",
        type=_positive_int,
        default=_resolve_int("keep_alive_secs", ("LYNSE_KEEP_ALIVE_SECS",), config, 75),
        help="HTTP keep-alive timeout in seconds",
    )
    parser.add_argument(
        "--request-timeout-secs",
        type=_positive_int,
        default=_resolve_int(
            "request_timeout_secs",
            ("LYNSE_CLIENT_REQUEST_TIMEOUT_SECS",),
            config,
            300,
        ),
        help="Client request timeout in seconds",
    )
    parser.add_argument(
        "--json-limit-mb",
        type=_positive_int,
        default=_resolve_int("json_limit_mb", ("LYNSE_JSON_LIMIT_MB",), config, 256),
        help="Max JSON request size in MB",
    )
    parser.add_argument(
        "--payload-limit-mb",
        type=_positive_int,
        default=_resolve_int("payload_limit_mb", ("LYNSE_PAYLOAD_LIMIT_MB",), config, 512),
        help="Max raw payload size in MB",
    )
    parser.add_argument(
        "--slow-query-warn-ms",
        type=_non_negative_int,
        default=_resolve_non_negative_int(
            "slow_query_warn_ms",
            ("LYNSE_SLOW_QUERY_WARN_MS",),
            config,
            1000,
        ),
        help="Emit slow_query warnings for search/query requests at or above this latency in ms; 0 disables",
    )
    parser.add_argument(
        "--max-top-k",
        type=_non_negative_int,
        default=_resolve_non_negative_int("max_top_k", ("LYNSE_MAX_TOP_K",), config, 10000),
        help="Maximum k/max_results/head/tail result size accepted by the server; 0 disables",
    )
    parser.add_argument(
        "--max-batch-vectors",
        type=_non_negative_int,
        default=_resolve_non_negative_int(
            "max_batch_vectors",
            ("LYNSE_MAX_BATCH_VECTORS",),
            config,
            100000,
        ),
        help="Maximum vectors/IDs/queries accepted in one server request; 0 disables",
    )
    parser.add_argument(
        "--max-collection-vectors",
        type=_non_negative_int,
        default=_resolve_non_negative_int(
            "max_collection_vectors",
            ("LYNSE_MAX_COLLECTION_VECTORS",),
            config,
            10000000,
        ),
        help="Maximum primary vectors allowed per collection in server mode; 0 disables",
    )
    parser.add_argument(
        "--max-collection-vector-bytes",
        type=_non_negative_int,
        default=_resolve_non_negative_int(
            "max_collection_vector_bytes",
            ("LYNSE_MAX_COLLECTION_VECTOR_BYTES",),
            config,
            1099511627776,
        ),
        help="Maximum estimated dense vector bytes per collection, including named vector fields; 0 disables",
    )
    parser.add_argument(
        "--audit-log",
        action=argparse.BooleanOptionalAction,
        default=_resolve_bool("audit_log", ("LYNSE_AUDIT_LOG",), config, True),
        help="Emit structured audit events for mutating server requests",
    )
    parser.add_argument(
        "--cluster-config",
        default=_resolve_str("cluster_config", ("LYNSE_CLUSTER_CONFIG",), config, None),
        help="Coordinator mode: path to cluster JSON config",
    )
    parser.add_argument(
        "--cluster-state",
        default=_resolve_str("cluster_state", ("LYNSE_CLUSTER_STATE",), config, None),
        help=(
            "Coordinator mode: local metadata cache path. Authoritative metadata "
            "is stored on metadata owner shard(s)."
        ),
    )
    parser.add_argument(
        "--shard-api-key",
        default=_resolve_str("shard_api_key", ("LYNSE_SHARD_API_KEY",), config, None),
        help="Coordinator mode: API key used when forwarding requests to shards",
    )
    parser.add_argument(
        "--coordinator-id",
        default=_resolve_str("coordinator_id", ("LYNSE_COORDINATOR_ID",), config, None),
        help="Coordinator mode: stable ID for leader election (default: advertised URI)",
    )
    parser.add_argument(
        "--coordinator-uri",
        default=_resolve_str("coordinator_uri", ("LYNSE_COORDINATOR_URI",), config, None),
        help="Coordinator mode: URI other coordinators use to proxy to this process",
    )
    parser.add_argument(
        "--coordinator-lease-secs",
        type=float,
        default=float(_resolve_str(
            "coordinator_lease_secs",
            ("LYNSE_COORDINATOR_LEASE_SECS",),
            config,
            "5.0",
        )),
        help="Coordinator mode: leader lease duration in seconds",
    )
    parser.add_argument(
        "--metadata-owners",
        default=_resolve_str(
            "metadata_owners",
            ("LYNSE_CLUSTER_METADATA_OWNERS",),
            config,
            None,
        ),
        help=(
            "Coordinator mode: comma-separated metadata owner shard HTTP URIs. "
            "Omit to infer from shard primaries; provide 3+ for replicated metadata."
        ),
    )
    parser.add_argument(
        "--health-interval-secs",
        type=float,
        default=float(_resolve_str("health_interval_secs", ("LYNSE_HEALTH_INTERVAL_SECS",), config, "1.0")),
        help="Coordinator mode: shard health probe interval in seconds",
    )
    parser.add_argument(
        "--health-failures",
        type=_positive_int,
        default=_resolve_int("health_failures", ("LYNSE_HEALTH_FAILURES",), config, 3),
        help="Coordinator mode: consecutive failed health probes before failover",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.role == "coordinator":
        from .cluster import run_coordinator

        print(
            f"Starting LynseDB coordinator on {args.host}:{args.port} "
            f"(metadata cache: {args.cluster_state or 'cluster_state.cache.json'})"
        )
        try:
            run_coordinator(
                host=args.host,
                port=args.port,
                cluster_config=args.cluster_config,
                cluster_state=args.cluster_state,
                shard_api_key=args.shard_api_key,
                request_timeout_secs=args.request_timeout_secs,
                health_interval_secs=args.health_interval_secs,
                health_failures=args.health_failures,
                coordinator_id=args.coordinator_id,
                coordinator_uri=args.coordinator_uri,
                coordinator_lease_secs=args.coordinator_lease_secs,
                metadata_owners=[
                    item.strip()
                    for item in str(args.metadata_owners or "").split(",")
                    if item.strip()
                ],
            )
        except KeyboardInterrupt:
            print("\nCoordinator stopped.")
            sys.exit(0)
        return

    from ._backend import start_server

    auth_suffix = " [auth enabled]" if args.api_key else ""
    print(
        f"Starting LynseDB server on {args.host}:{args.port} "
        f"(root: {args.data_dir}){auth_suffix}"
    )

    os.environ["LYNSE_KEEP_ALIVE_SECS"] = str(args.keep_alive_secs)
    os.environ["LYNSE_CLIENT_REQUEST_TIMEOUT_SECS"] = str(args.request_timeout_secs)
    os.environ["LYNSE_JSON_LIMIT_MB"] = str(args.json_limit_mb)
    os.environ["LYNSE_PAYLOAD_LIMIT_MB"] = str(args.payload_limit_mb)
    os.environ["LYNSE_SLOW_QUERY_WARN_MS"] = str(args.slow_query_warn_ms)
    os.environ["LYNSE_MAX_TOP_K"] = str(args.max_top_k)
    os.environ["LYNSE_MAX_BATCH_VECTORS"] = str(args.max_batch_vectors)
    os.environ["LYNSE_MAX_COLLECTION_VECTORS"] = str(args.max_collection_vectors)
    os.environ["LYNSE_MAX_COLLECTION_VECTOR_BYTES"] = str(args.max_collection_vector_bytes)
    os.environ["LYNSE_AUDIT_LOG"] = "true" if args.audit_log else "false"
    if args.workers is not None:
        os.environ["LYNSE_SERVER_WORKERS"] = str(args.workers)

    try:
        start_server(
            host=args.host,
            port=args.port,
            root_path=args.data_dir,
            api_key=args.api_key,
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
