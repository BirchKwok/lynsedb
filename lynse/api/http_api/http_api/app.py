from flask import Flask
from waitress import serve
import socket

from lynse.api.http_api.http_api.database_ops import database_ops
from lynse.api.http_api.http_api.collection_ops import collection_ops


app = Flask(__name__)
app.register_blueprint(database_ops)
app.register_blueprint(collection_ops)


def get_local_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e:
        return "127.0.0.1"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Start the LynseDB HTTP API server.')
    parser.add_argument('--host', default='127.0.0.1', help='The host to bind to.')
    parser.add_argument('--port', default=7637, type=int, help='The port to bind to.')
    parser.add_argument('--threads', default=10, type=int, help='Number of threads per worker.')
    parser.add_argument('run', help='Run the server.')
    args = parser.parse_args()

    if args.run:
        if args.host == '0.0.0.0':
            local_ip = get_local_ip()
            print(f"Server running at:")
            print(f"  - Localhost: http://localhost:{args.port}")
            print(f"  - Local IP: http://{local_ip}:{args.port}", end="\n\n")
        else:
            print(f"Server running at http://{args.host}:{args.port}", end="\n\n")

        serve(app, host=args.host, port=args.port, threads=args.threads)


def launch_in_jupyter(host: str = '127.0.0.1', port: int = 7637, threads: int = 10):
    """
    Launch the HTTP API server in Python environment.

    Parameters:
        host (str): The host to bind to. Default is '127.0.0.1'
        port (int): The port to bind to. Default is 7637.
        threads (int): Number of threads per worker. Default is 10.

    Returns:
        None
    """
    import threading
    if host == '0.0.0.0':
        local_ip = get_local_ip()
        print(f"Server running at:")
        print(f"  - Localhost: http://localhost:{port}")
        print(f"  - Local IP: http://{local_ip}:{port}", end="\n\n")
    else:
        print(f"Server running at http://{host}:{port}", end="\n\n")

    # set the thread as daemon thread
    server_thread = threading.Thread(target=serve, args=(app,), kwargs={'host': host, 'port': port, 'threads': threads})
    server_thread.daemon = True
    server_thread.start()


if __name__ == '__main__':
    main()
