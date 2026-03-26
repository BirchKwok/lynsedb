"""
LynseDB Server CLI.

Launch the Rust-based HTTP server for LynseDB.
This replaces the old Flask-based server.

Usage:
    python -m lynse.server                          # default: 127.0.0.1:7637
    python -m lynse.server --host 0.0.0.0 --port 8080 --root /data/lynsedb
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="LynseDB HTTP Server (Rust)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7637, help="Port (default: 7637)")
    parser.add_argument("--root", default=".", help="Root data directory (default: current dir)")
    args = parser.parse_args()

    from ._backend import start_server

    print(f"Starting LynseDB server on {args.host}:{args.port} (root: {args.root})")
    try:
        start_server(host=args.host, port=args.port, root_path=args.root)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
