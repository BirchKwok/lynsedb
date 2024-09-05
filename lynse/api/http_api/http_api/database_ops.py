import json
import os
import shutil

import msgpack

from flask import request, jsonify, Response, Blueprint

from ....configs.config import config
from ....core_components.safe_dict import SafeDict


database_ops = Blueprint('database_ops', __name__)


tasks = SafeDict()

root_path = config.LYNSE_DEFAULT_ROOT_PATH


@database_ops.before_request
def handle_msgpack_input():
    if request.headers.get('Content-Type') == 'application/msgpack':
        data = msgpack.unpackb(request.get_data(), raw=False)
        request.decoded_data = data


@database_ops.route('/', methods=['GET'])
def index():
    return Response(json.dumps({'status': 'success', 'message': 'LynseDB HTTP API'}), mimetype='application/json')


@database_ops.route('/drop_database', methods=['POST'])
def drop_database():
    data = request.json
    if not data or 'database_name' not in data:
        return jsonify({'error': 'No data provided or missing database_name'}), 400

    try:
        shutil.rmtree(root_path / data['database_name'])
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@database_ops.route('/database_exists', methods=['POST'])
def database_exists():
    """Check if the database exists.

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    try:
        exists = (root_path / data['database_name']).exists()
        return Response(json.dumps({
            'status': 'success', 'params': {
                'exists': True if exists else False
            }
        },
            sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@database_ops.route('/set_environment', methods=['POST'])
def set_environment():
    """Set the environment variables.

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


@database_ops.route('/get_environment', methods=['GET'])
def get_environment():
    """Get the environment variables.

    Returns:
        dict: The status of the operation.
    """
    env_list = ['LYNSE_LOG_LEVEL', 'LYNSE_LOG_PATH', 'LYNSE_TRUNCATE_LOG', 'LYNSE_LOG_WITH_TIME',
                'LYNSE_KMEANS_EPOCHS', 'LYNSE_SEARCH_CACHE_SIZE']

    params = {key: eval("global_config.key") for key in env_list}
    try:
        return Response(json.dumps({'status': 'success', 'params': params}, sort_keys=False),
                        mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@database_ops.route('/list_databases', methods=['GET'])
def list_databases():
    """List all databases.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.database_manager import DatabaseManager

    try:
        data_manager = DatabaseManager(config.LYNSE_DEFAULT_ROOT_PATH)
        databases = data_manager.list_database()
        return Response(json.dumps({'status': 'success', 'params': {'databases': databases}}, sort_keys=False),
                        mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@database_ops.route('/delete_database', methods=['POST'])
def delete_database():
    """Delete a database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.database_manager import DatabaseManager

    data = request.json
    try:
        data_manager = DatabaseManager(config.LYNSE_DEFAULT_ROOT_PATH)
        data_manager.delete(data['database_name'])
        return Response(json.dumps({'status': 'success', 'params': {'database_name': data['database_name']}},
                                   sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@database_ops.route('/create_database', methods=['POST'])
def create_database():
    """Create a database.

    Returns:
        dict: The status of the operation.
    """
    from ....api.native_api.database_manager import DatabaseManager
    from ....api.native_api.high_level import LocalClient

    data = request.json
    try:
        data_manager = DatabaseManager(config.LYNSE_DEFAULT_ROOT_PATH)
        data_manager.register(data['database_name'])

        if data['drop_if_exists']:
            LocalClient(root_path=root_path / data['database_name']).drop_database()

        LocalClient(root_path=root_path / data['database_name'])
        return Response(json.dumps({'status': 'success', 'params': {'database_name': data['database_name']}},
                                   sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
