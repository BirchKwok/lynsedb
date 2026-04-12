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
        try:
            from ruamel.yaml import YAML
        except Exception as exc:
            raise SystemExit(
                "YAML config requires ruamel.yaml; install dependencies or use JSON."
            ) from exc
        yaml = YAML(typ="safe")
        try:
            data = yaml.load(content)
        except Exception as exc:
            raise SystemExit(f"Invalid YAML config file: {path}") from exc

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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
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
        help="Path to JSON/YAML server config file",
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
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

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
