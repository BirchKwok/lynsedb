import json
import uuid
from threading import Thread

import msgpack

from flask import request, jsonify, Response, Blueprint

from ....configs.config import config
from ....core_components.fields_cache import IndexSchema
from ....core_components.limited_dict import LimitedDict
from ....core_components.safe_dict import SafeDict

collection_ops = Blueprint('collection_ops', __name__)

data_dict = LimitedDict(max_size=1000)
tasks = SafeDict()

root_path = config.LYNSE_DEFAULT_ROOT_PATH


@collection_ops.before_request
def handle_msgpack_input():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = msgpack.unpackb(request.get_data(), raw=False)
        request.decoded_data = data


@collection_ops.route('/', methods=['GET'])
def index():
    return Response(json.dumps({'status': 'success', 'message': 'LynseDB HTTP API'}), mimetype='application/json')


@collection_ops.route('/required_collection', methods=['POST'])
def required_collection():
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'chunk_size' not in data:
        data['chunk_size'] = 100000
    if 'dtypes' not in data:
        data['dtypes'] = 'float32'
    if 'use_cache' not in data:
        data['use_cache'] = True
    if 'n_threads' not in data:
        data['n_threads'] = 10
    if 'warm_up' not in data:
        data['warm_up'] = True
    if 'drop_if_exists' not in data:
        data['drop_if_exists'] = False
    if 'description' not in data:
        data['description'] = None
    if 'cache_chunks' not in data:
        data['cache_chunks'] = 20

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        my_vec_db.require_collection(
            collection=data['collection_name'],
            dim=data['dim'],
            chunk_size=data['chunk_size'],
            dtypes=data['dtypes'],
            use_cache=data['use_cache'],
            n_threads=data['n_threads'],
            warm_up=data['warm_up'],
            drop_if_exists=data['drop_if_exists'],
            description=data['description'],
            cache_chunks=data['cache_chunks']
        )
        return jsonify({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']
        }}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/drop_collection', methods=['POST'])
def drop_collection():
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        my_vec_db.drop_collection(data['collection_name'])
        return jsonify({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']}
        }), 200
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/show_collections', methods=['POST'])
def show_collections():
    """Show all collections in the database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'database_name' not in data:
        return jsonify({'error': 'Missing required parameter: database_name'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collections = my_vec_db.show_collections()

        return Response(json.dumps({
            'status': 'success', 'params': {
                'database_name': data['database_name'],
                'collections': collections
            }
        }, sort_keys=False), mimetype='application/json')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/add_item', methods=['POST'])
def add_item():
    from ....api.native_api.high_level import LocalClient

    if request.headers.get('Content-Type') == 'application/msgpack':
        data = request.decoded_data
    else:
        data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        id = collection.add_item(vector=data['item']['vector'], id=data['item']['id'],
                                 field=data['item'].get('field', {}))

        return Response(json.dumps(
            {
                'status': 'success', 'params': {
                'database_name': data['database_name'],
                'collection_name': data['collection_name'], 'item': {
                    'id': data['item']['id']
                }
            }
            }, sort_keys=False),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/bulk_add_items', methods=['POST'])
def bulk_add_items():
    from ....api.native_api.high_level import LocalClient

    if request.headers.get('Content-Type') == 'application/msgpack':
        data = request.decoded_data
    else:
        data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        items = []
        for item in data['items']:
            items.append((item['vector'], item['id'], item.get('field', None)))
        ids = collection.bulk_add_items(items)

        return Response(json.dumps({
            'status': 'success', 'params': {
                'database_name': data['database_name'],
                'collection_name': data['collection_name'], 'ids': [i['id'] for i in data['items']]
            }
        }, sort_keys=False),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/search', methods=['POST'])
def search():
    """Search the database for the vectors most similar to the given vector.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient
    from ....core_components.fields_cache.filter import Filter

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'k' not in data:
        data['k'] = 10

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        if data['search_filter'] is None:
            search_filter = None
        else:
            search_filter = Filter().load_dict(data['search_filter'])

        indexer = getattr(collection, '_indexer', None)
        if indexer is not None:
            index_mode_judge = 'Binary' not in indexer.index_mode
        else:
            index_mode_judge = False

        ids, scores, field = collection.search(
            vector=data['vector'], k=data['k'],
            search_filter=search_filter,
            return_fields=data.get('return_fields', False),
            rescore=data.get('rescore', False),
            rescore_multiplier=(data.get('rescore_multiplier', 2)
                                if index_mode_judge else data.get('rescore_multiplier', 10))
        )

        if ids is not None:
            ids = ids.tolist()
            scores = scores.tolist()

        return Response(json.dumps(
            {
                'status': 'success', 'params': {
                'database_name': data['database_name'],
                'collection_name': data['collection_name'], 'items': {
                    'k': data['k'], 'ids': ids, 'scores': scores,
                    'fields': field,
                }
            }
            }, sort_keys=False),
            mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def commit_task(data):
    from ....api.native_api.high_level import LocalClient

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        if 'items' in data:
            items = []
            for item in data['items']:
                items.append((item['vector'], item['id'], item.get('field', {})))

            collection.bulk_add_items(items)

        collection.commit()

        return {'status': 'Success', 'result': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']
        }}
    except KeyError as e:
        return {'status': 'Error', 'message': f' missing required parameter {e}'}
    except Exception as e:
        return {'status': 'Error', 'message': str(e)}


@collection_ops.route('/commit', methods=['POST'])
def commit():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = request.decoded_data
    else:
        data = request.json

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'Processing'}

    def execute_task():
        result = commit_task(data)
        tasks[task_id] = result

    thread = Thread(target=execute_task)
    thread.start()

    return jsonify({'task_id': task_id}), 202


@collection_ops.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(task)


@collection_ops.route('/collection_shape', methods=['POST'])
def collection_shape():
    """Get the shape of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'shape': collection.shape
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/get_collection_status_report', methods=['POST'])
def get_collection_status_report():
    """Get the status report of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        db_report = {'DATABASE STATUS REPORT': {
            'Database shape': (
                0, collection._matrix_serializer.dim) if collection._matrix_serializer.IS_DELETED else collection.shape,
            'Database last_commit_time': collection._matrix_serializer.last_commit_time,
            'Database commit status': collection._matrix_serializer.COMMIT_FLAG,
            'Database use_cache': collection._use_cache,
            'Database status': 'DELETED' if collection._matrix_serializer.IS_DELETED else 'ACTIVE'
        }}

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'status_report': db_report
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/is_collection_exists', methods=['POST'])
def is_collection_exists():
    """Check if a collection exists.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'exists': data['collection_name'] in my_vec_db.show_collections()
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/get_collection_config', methods=['POST'])
def get_collection_config():
    """Get the configuration of a collection.

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        config_json_path = root_path / data['database_name'] / 'collections.json'
        with open(config_json_path, 'r') as file:
            collections = json.load(file)
            collection_config = collections[data['collection_name']]

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'config': collection_config
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/update_commit_msg', methods=['POST'])
def update_commit_msg():
    """Save the commit message of a collection.

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        if (root_path / data['database_name'] / 'commit_msg.json').exists():
            with open(root_path / data['database_name'] / 'commit_msg.json', 'r') as file:
                commit_msg = json.load(file)
                commit_msg[data['collection_name']] = data
        else:
            commit_msg = {data['collection_name']: data}

        with open(root_path / data['database_name'] / 'commit_msg.json', 'w') as file:
            json.dump(commit_msg, file)

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/get_commit_msg', methods=['POST'])
def get_commit_msg():
    """Get the commit message of a collection.

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        if (root_path / data['database_name'] / 'commit_msg.json').exists():
            with open(root_path / data['database_name'] / 'commit_msg.json', 'r') as file:
                commit_msg = json.load(file)
                commit_msg = commit_msg.get(data['collection_name'], None)
        else:
            commit_msg = 'No commit message found for this collection'

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'commit_msg': commit_msg
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/update_collection_description', methods=['POST'])
def update_collection_description():
    """Update the description of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        my_vec_db.update_collection_description(data['collection_name'], data['description'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'description': data['description']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/update_description', methods=['POST'])
def update_description():
    """Update the description of the database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        collection.update_description(data['description'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'description': data['description']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/show_collections_details', methods=['POST'])
def show_collections_details():
    """Show all collections in the database with details.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collections_details = my_vec_db.show_collections_details()
        return Response(json.dumps({
            'status': 'success', 'params': {
                'database_name': data['database_name'],
                'collections': collections_details.to_dict()
            }
        },
            sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/build_index', methods=['POST'])
def build_index():
    """Build the index of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        collection.build_index(index_mode=data.get('index_mode', 'IVF-FLAT'))

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'index_mode': data.get('index_mode', 'IVF-FLAT')
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/remove_index', methods=['POST'])
def remove_index():
    """Remove the index of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        collection.remove_index()

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/head', methods=['POST'])
def head():
    """Get the first n items of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'n' not in data:
        data['n'] = 5

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        head = collection.head(n=data['n'])
        head = (head[0].tolist(), head[1].tolist(), head[2])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'head': head
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/tail', methods=['POST'])
def tail():
    """Get the last n items of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'n' not in data:
        data['n'] = 5

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        tail = collection.tail(n=data['n'])
        tail = (tail[0].tolist(), tail[1].tolist(), tail[2])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'tail': tail
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/read_by_only_id', methods=['POST'])
def read_by_only_id():
    """Read the item by only id.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        vector, id, field = collection.read_by_only_id(data['id'])
        vector = vector.tolist()
        id = id.tolist()

        item = (vector, id, field)
        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'], 'item': item
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/get_collection_path', methods=['POST'])
def get_collection_path():
    """Get the path of a database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'collection_path': collection._database_path}}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/query', methods=['POST'])
def query():
    """Query the database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        result = collection.query(data['query_filter'], data['filter_ids'], data['return_ids_only'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'result': result}}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@collection_ops.route('/query_vectors', methods=['POST'])
def query_vectors():
    """Query the database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        result = collection.query_vectors(data['query_filter'], data['filter_ids'])
        result = (result[0].tolist(), result[1].tolist(), result[2])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'result': result}}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/build_field_index', methods=['POST'])
def build_field_index():
    """Build the index of a field.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        schema = IndexSchema().load_from_dict(data['schema'])

        collection.build_field_index(schema=schema,
                                     rebuild_if_exists=data['rebuild_if_exists'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/list_fields', methods=['POST'])
def list_fields():
    """List all fields of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])
        fields = collection.list_fields()

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'fields': fields
        }}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/list_field_index', methods=['POST'])
def list_field_index():
    """List all field indexes of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        field_indices = collection.list_field_index()

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'field_indices': field_indices}}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/remove_field_index', methods=['POST'])
def remove_field_index():
    """Remove the index of a field.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        collection.remove_field_index(data['field_name'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'field_name': data['field_name']
        }}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/remove_all_field_indices', methods=['POST'])
def remove_all_field_indices():
    """Remove all field indices of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        collection.remove_all_field_indices()

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name']
        }}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/index_mode', methods=['POST'])
def index_mode():
    """Get the index mode of a collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        index_mode = collection.index_mode

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'index_mode': index_mode
        }}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/is_id_exists', methods=['POST'])
def is_id_exists():
    """Check if an ID exists in the database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        is_id_exists = collection.is_id_exists(data['id'])

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'id': data['id'],
            'is_id_exists': is_id_exists
        }}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@collection_ops.route('/max_id', methods=['POST'])
def max_id():
    """Get the maximum ID in the collection.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.high_level import LocalClient

    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = LocalClient(root_path=root_path / data['database_name'])
        collection = my_vec_db.get_collection(data['collection_name'])

        max_id = int(collection.max_id)

        return Response(json.dumps({'status': 'success', 'params': {
            'database_name': data['database_name'],
            'collection_name': data['collection_name'],
            'max_id': max_id
        }}, sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
