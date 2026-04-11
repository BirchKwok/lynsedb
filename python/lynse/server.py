"""
LynseDB Server CLI.

Launch the Rust-based HTTP server for LynseDB.
This replaces the old Flask-based server.

Usage:
    python -m lynse.server                          # default: 127.0.0.1:7637
    python -m lynse.server run --host 0.0.0.0 --port 8080 --root /data/lynsedb
    lynse run --host 0.0.0.0 --port 7637 --api-key secret
"""

import argparse
import os
import sys


def _default_port() -> int:
    raw = os.environ.get("LYNSE_PORT") or os.environ.get("PORT") or "7637"
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid port value: {raw!r}") from exc


def _parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Support both `lynse run ...` and the older direct flag style.
    if argv[:1] == ["run"]:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="LynseDB HTTP Server (Rust)")
    parser.add_argument(
        "--host",
        default=os.environ.get("LYNSE_HOST", "127.0.0.1"),
        help="Bind address (default: 127.0.0.1 or LYNSE_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_default_port(),
        help="Port (default: 7637, or LYNSE_PORT / PORT)",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("LYNSE_ROOT", "."),
        help="Root data directory (default: current dir or LYNSE_ROOT)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LYNSE_API_KEY"),
        help="Optional API key for Bearer / Basic auth (default: LYNSE_API_KEY)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    from ._backend import start_server

    auth_suffix = " [auth enabled]" if args.api_key else ""
    print(f"Starting LynseDB server on {args.host}:{args.port} (root: {args.root}){auth_suffix}")
    try:
        start_server(host=args.host, port=args.port, root_path=args.root, api_key=args.api_key)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
