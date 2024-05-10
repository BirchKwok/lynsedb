import json
import os
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import msgpack
import yaml

from flask import Flask, request, jsonify, Response
from waitress import serve

from min_vec.api.native_api.high_level import MinVectorDBLocalClient
from min_vec.core_components.filter import Filter
from min_vec.core_components.limited_dict import LimitedDict

app = Flask(__name__)


data_dict = LimitedDict(max_size=1000)


# 构建配置文件的路径
config_path = Path(os.path.expanduser('~/.MinVectorDB/config.yaml'))
default_root_path = Path(os.path.expanduser('~/.MinVectorDB/data/min_vec_db')).as_posix()


@app.before_request
def handle_msgpack_input():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = msgpack.unpackb(request.get_data(), raw=False)
        request.decoded_data = data


def generate_config(root_path: str = default_root_path, config_path: Path = config_path):
    """Generate a default configuration file.
        .. versionadded:: 0.3.2
    """
    if root_path is None:
        root_path = default_root_path

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {'root_path': root_path}

        with open(config_path, 'w') as file:
            yaml.dump(config, file)
    else:
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        if 'root_path' not in config or config['root_path'] != root_path:
            config['root_path'] = root_path
            with open(config_path, 'w') as file:
                yaml.dump(config, file)

    return config_path, root_path, config


root_path, config = 'PlaceholderName', {'root_path': 'PlaceholderName'}


@app.route('/', methods=['GET'])
def index():
    return Response(json.dumps({'status': 'success', 'message': 'MinVectorDB HTTP API'}), mimetype='application/json')


@app.route('/required_collection', methods=['POST'])
def required_collection():
    """Create a collection in the database.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'n_clusters' not in data:
        data['n_clusters'] = 16
    if 'chunk_size' not in data:
        data['chunk_size'] = 100000
    if 'distance' not in data:
        data['distance'] = 'cosine'
    if 'index_mode' not in data:
        data['index_mode'] = 'IVF-FLAT'
    if 'dtypes' not in data:
        data['dtypes'] = 'float32'
    if 'use_cache' not in data:
        data['use_cache'] = True
    if 'scaler_bits' not in data:
        data['scaler_bits'] = 8
    if 'n_threads' not in data:
        data['n_threads'] = 10
    if 'warm_up' not in data:
        data['warm_up'] = True
    if 'drop_if_exists' not in data:
        data['drop_if_exists'] = False
    if 'description' not in data:
        data['description'] = None

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        my_vec_db.require_collection(
            collection=data['collection_name'],
            dim=data['dim'],
            n_clusters=data['n_clusters'],
            chunk_size=data['chunk_size'],
            distance=data['distance'],
            index_mode=data['index_mode'],
            dtypes=data['dtypes'],
            use_cache=data['use_cache'],
            scaler_bits=data['scaler_bits'],
            n_threads=data['n_threads'],
            warm_up=data['warm_up'],
            drop_if_exists=data['drop_if_exists'],
            description=data['description']
        )

        if data['collection_name'] not in data_dict:
            data_dict[data['collection_name']] = queue.Queue()

        return Response(json.dumps({
            'status': 'success',
            'params': {
                'collection_name': data['collection_name'],
                'dim': data['dim'], 'n_clusters': data['n_clusters'], 'chunk_size': data['chunk_size'],
                'distance': data['distance'], 'index_mode': data['index_mode'], 'dtypes': data['dtypes'],
                'use_cache': data['use_cache'], 'scaler_bits': data['scaler_bits'],
                'n_threads': data['n_threads'],
                'warm_up': data['warm_up'], 'drop_if_exists': data['drop_if_exists']
            }
        }, sort_keys=False), mimetype='application/json')

    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/drop_collection', methods=['POST'])
def drop_collection():
    """Drop a collection from the database.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        my_vec_db.drop_collection(data['collection_name'])

        return Response(json.dumps(
            {
                'status': 'success', 'params': {
                'collection_name': data['collection_name']
            }
            },
            sort_keys=False),
            mimetype='application/json')

    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/drop_database', methods=['GET'])
def drop_database():
    """Drop the database.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    try:
        shutil.rmtree(config['root_path'])

        return Response(json.dumps({
            'status': 'success'
        }, sort_keys=False),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/database_exists', methods=['GET'])
def database_exists():
    """Check if the database exists.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    try:
        exists = Path(config['root_path']).exists()
        return Response(json.dumps({
            'status': 'success', 'params': {
                'exists': True if exists else False
            }
        },
            sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/show_collections', methods=['GET'])
def show_collections():
    """Show all collections in the database.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collections = my_vec_db.show_collections()
        return Response(json.dumps({'status': 'success', 'params': {'collections': collections}},
                                   sort_keys=False), mimetype='application/json')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_item', methods=['POST'])
def add_item():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = request.decoded_data
    else:
        data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    executor = ThreadPoolExecutor(max_workers=1)

    future = executor.submit(process_add_item, data)
    data_dict[data['collection_name']].put(future)

    return Response(json.dumps(
            {
                'status': 'success', 'params': {
                    'collection_name': data['collection_name'], 'item': {
                        'id': data['item']['id']
                    }
                }
            }, sort_keys=False),
            mimetype='application/json')


def process_add_item(data):
    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        id = collection.add_item(vector=data['item']['vector'], id=data['item']['id'],
                                 field=data['item'].get('field', None))
        return {'status': 'success', 'id': id}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.route('/bulk_add_items', methods=['POST'])
def bulk_add_items():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = request.decoded_data
    else:
        data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    executor = ThreadPoolExecutor(max_workers=10)
    future = executor.submit(process_bulk_add_items, data)
    data_dict[data['collection_name']].put(future)

    return Response(json.dumps({
            'status': 'success', 'params': {
                'collection_name': data['collection_name'], 'ids': [i['id'] for i in data['items']]
            }
        }, sort_keys=False),
            mimetype='application/json')


def process_bulk_add_items(data):
    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        items = []
        for item in data['items']:
            items.append((item['vector'], item['id'], item.get('field', None)))
        ids = collection.bulk_add_items(items)
        return {'status': 'success', 'ids': ids}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.route('/query', methods=['POST'])
def query():
    """Query the database for the vectors most similar to the given vector.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'k' not in data:
        data['k'] = 10

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])

        if data['query_filter'] is None:
            query_filter = None
        else:
            query_filter = Filter().load_dict(data['query_filter'])

        ids, scores = collection.query(vector=data['vector'], k=data['k'],
                                       query_filter=query_filter,
                                       distance=data.get('distance', 'cosine'),
                                       return_similarity=data.get('return_similarity', True))

        if ids is not None:
            ids = ids.tolist()
            scores = scores.tolist()

        return Response(json.dumps(
            {
                'status': 'success', 'params': {
                    'collection_name': data['collection_name'], 'items': {
                        'k': data['k'], 'ids': ids, 'scores': scores,
                        'distance': collection.most_recent_query_report['Query Distance'],
                        'query time': collection.most_recent_query_report['Query Time']
                    }
            }
            }, sort_keys=False),
            mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/commit', methods=['POST'])
def commit():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = request.decoded_data
    else:
        data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    tasks_queue: queue.Queue = data_dict[data['collection_name']]

    if not tasks_queue.empty():
        tasks = []
        while not tasks_queue.empty():
            tasks.append(tasks_queue.get())

        for future in as_completed(tasks):
            result = future.result()
            if result.get('status') == 'error':
                return jsonify(result), 500

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        if 'items' in data:
            items = []
            for item in data['items']:
                items.append((item['vector'], item.get('id', None), item.get('field', None)))

            collection.bulk_add_items(items)

        collection.commit()

        return Response(json.dumps({
            'status': 'success', 'params': {
                'collection_name': data['collection_name']
            }
        }, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/collection_shape', methods=['POST'])
def collection_shape():
    """Get the shape of a collection.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'shape': collection.shape
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/set_environment', methods=['POST'])
def set_environment():
    """Set the environment variables.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        for key, value in data.items():
            os.environ[key] = value
        return Response(json.dumps({'status': 'success', 'params': data}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_environment', methods=['GET'])
def get_environment():
    """Get the environment variables.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    env_list = ['MVDB_LOG_LEVEL', 'MVDB_LOG_PATH', 'MVDB_TRUNCATE_LOG', 'MVDB_LOG_WITH_TIME',
                'MVDB_KMEANS_EPOCHS', 'MVDB_QUERY_CACHE_SIZE', 'MVDB_DATALOADER_BUFFER_SIZE']

    params = {key: eval("global_config.key") for key in env_list}
    try:
        return Response(json.dumps({'status': 'success', 'params': params}, sort_keys=False),
                        mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_collection_query_report', methods=['POST'])
def get_collection_query_report():
    """Get the query report of a collection.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'query_report': collection.query_report_
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_collection_status_report', methods=['POST'])
def get_collection_status_report():
    """Get the status report of a collection.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])

        db_report = {'DATABASE STATUS REPORT': {
            'Database shape': (
                0, collection._matrix_serializer.dim) if collection._matrix_serializer.IS_DELETED else collection.shape,
            'Database last_commit_time': collection._matrix_serializer.last_commit_time,
            'Database commit status': collection._matrix_serializer.COMMIT_FLAG,
            'Database index_mode': collection._matrix_serializer.index_mode,
            'Database distance': collection._distance,
            'Database use_cache': collection._use_cache,
            'Database status': 'DELETED' if collection._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'status_report': db_report
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/is_collection_exists', methods=['POST'])
def is_collection_exists():
    """Check if a collection exists.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'exists': data['collection_name'] in my_vec_db.show_collections()
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_collection_config', methods=['POST'])
def get_collection_config():
    """Get the configuration of a collection.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        config_json_path = Path(config['root_path']) / 'collections.json'
        with open(config_json_path, 'r') as file:
            collections = json.load(file)
            collection_config = collections[data['collection_name']]

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'config': collection_config
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_commit_msg', methods=['POST'])
def update_commit_msg():
    """Save the commit message of a collection.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        if (Path(config['root_path']) / 'commit_msg.json').exists():
            with open(config['root_path'] + '/commit_msg.json', 'r') as file:
                commit_msg = json.load(file)
                commit_msg[data['collection_name']] = data
        else:
            commit_msg = {data['collection_name']: data}

        with open(config['root_path'] + '/commit_msg.json', 'w') as file:
            json.dump(commit_msg, file)

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_commit_msg', methods=['POST'])
def get_commit_msg():
    """Get the commit message of a collection.
        .. versionadded:: 0.3.2

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        if (Path(config['root_path']) / 'commit_msg.json').exists():
            with open(config['root_path'] + '/commit_msg.json', 'r') as file:
                commit_msg = json.load(file)
                commit_msg = commit_msg.get(data['collection_name'], None)
        else:
            commit_msg = 'No commit message found for this collection'

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'commit_msg': commit_msg
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_collection_description', methods=['POST'])
def update_collection_description():
    """Update the description of a collection.
        .. versionadded:: 0.3.4

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        my_vec_db.update_collection_description(data['collection_name'], data['description'])

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'description': data['description']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/update_description', methods=['POST'])
def update_description():
    """Update the description of the database.
        .. versionadded:: 0.3.4

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        collection.update_description(data['description'])

        return Response(json.dumps({'status': 'success', 'params': {
            'description': data['description']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show_collections_details', methods=['GET'])
def show_collections_details():
    """Show all collections in the database with details.
        .. versionadded:: 0.3.4

    Returns:
        dict: The status of the operation.
    """
    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collections_details = my_vec_db.show_collections_details()
        return Response(json.dumps({'status': 'success', 'params': {'collections': collections_details.to_dict()}},
                                   sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    global config_path, default_root_path, config, root_path
    import argparse
    parser = argparse.ArgumentParser(description='Start the MinVectorDB HTTP API server.')
    parser.add_argument('--host', default='127.0.0.1', help='The host to bind to.')
    parser.add_argument('--port', default=7637, type=int, help='The port to bind to.')
    parser.add_argument('--threads', default=4, type=int, help='Number of threads per worker.')
    parser.add_argument('--config', default=config_path, help='The path to the configuration file.')
    parser.add_argument('--root_path', default=default_root_path, help='The path to the database root directory.')
    parser.add_argument('run', help='Run the server.')
    args = parser.parse_args()

    config_path = args.config
    root_path = args.root_path
    config_path, root_path, config = generate_config(root_path=root_path, config_path=config_path)

    if args.run:
        # Flask.run(app, host=args.host, port=args.port, debug=True)
        serve(app, host=args.host, port=args.port, threads=args.threads)


if __name__ == '__main__':
    main()
