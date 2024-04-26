"""使用flask实现的http接口, 用于向外部提供服务"""
import json
import operator
import os
import shutil
from pathlib import Path

import numpy as np
import portalocker
import yaml

from flask import Flask, request, jsonify, Response
from waitress import serve

from min_vec.api.high_level import MinVectorDBLocalClient
from min_vec.structures.filter import Filter, FieldCondition, MatchField, IDCondition, MatchID


os.environ['MVDB_LOG_LEVEL'] = 'ERROR'

app = Flask(__name__)

version = '0.3.2'

# 构建配置文件的路径
config_path = Path(os.path.expanduser('~/.MinVectorDB/config.yaml'))
default_root_path = Path(os.path.expanduser('~/.MinVectorDB/data/min_vec_db')).as_posix()


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
            portalocker.lock(file, portalocker.LOCK_EX)
            yaml.dump(config, file)
    else:
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            if 'root_path' not in config or config['root_path'] != root_path:
                config['root_path'] = root_path
                with open(config_path, 'w') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/required_collection \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "dim": 4,
              "n_clusters": 10,
              "chunk_size": 1024,
              "distance": "cosine",
              "index_mode": "IVF-FLAT",
              "dtypes": "float32",
              "use_cache": true,
              "scaler_bits": 8,
              "n_threads": 4,
              "warm_up": true,
              "drop_if_exists": false
             }' -w "\n"
        2) use python requests
        import requests
        
        url = 'http://localhost:7637/required_collection'
        data = {
            "collection_name": "example_collection",
            "dim": 4,
            "n_clusters": 10,
            "chunk_size": 1024,
            "distance": "L2",
            "index_mode": "IVF-FLAT",
            "dtypes": "float32",
            "use_cache": True,
            "scaler_bits": 8,
            "n_threads": 4,
            "warm_up": True,
            "drop_if_exists": False
        }
        response = requests.post(url, json=data)
        print(response.json())
        
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

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.require_collection(
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
            drop_if_exists=data['drop_if_exists']
        )

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/drop_collection \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests
        
        url = 'http://localhost:7637/drop_collection'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())
        
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

    Example:
        1) use curl
        curl -X GET http://localhost:7637/drop_database -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/drop_database'
        response = requests.get(url)
        print(response.json())

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

    Example:
        1) use curl
        curl -X GET http://localhost:7637/database_exists -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/database_exists'
        response = requests.get(url)
        print(response.json())

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

    Example:
        1) use curl
        curl -X GET http://localhost:7637/show_collections -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/show_collections'
        response = requests.get(url)
        print(response.json())

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
    """Add an item to a collection.
        .. versionadded:: 0.3.2

    Example:
        1) use curl
        curl -X POST http://localhost:7637/add_item \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "item": {
                  "vector": [0.1, 0.2, 0.3, 0.4],
                  "id": 1,
                  "field": {
                      "name": "example",
                      "age": 18
                    }
                }
            }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/add_item'
        data = {
            "collection_name": "example_collection",
            "item": {
                "vector": [0.1, 0.2, 0.3, 0.4],
                "id": 1,
                "field": {
                    "name": "example",
                    "age": 18
                }
            }
        }
        response = requests.post(url, json=data)
        print(response.json())

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        id = collection.add_item(vector=np.array(data['item']['vector']), id=data['item'].get('id', None),
                                 field=data['item'].get('field', None))

        return Response(json.dumps(
            {
                'status': 'success', 'params': {
                    'collection_name': data['collection_name'], 'item': {
                        'vector': data['item']['vector'], 'id': id,
                        'field': data['item'].get('field', None)
                    }
                }
            }, sort_keys=False),
            mimetype='application/json')

    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    """Query the database for the vectors most similar to the given vector.
        .. versionadded:: 0.3.2

    Example:
        1) use curl
        curl -X POST http://localhost:7637/query \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "vector": [0.1, 0.2, 0.3, 0.4],
              "k": 10,
              "distance": "cosine",
              "query_filter": {
                  "must": [
                      {
                          "field": "name",
                          "operator": "eq",
                          "value": "example"
                      },
                      {
                          "ids": [1, 2, 3]
                      }
                  ],
                  "any": [
                      {
                          "field": "age",
                          "operator": "gt",
                          "value": 18
                        }
                    ]
                },
                "return_similarity": true
            }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/query'
        data = {
            "collection_name": "example_collection",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "k": 10,
            "distance": "cosine",
            "query_filter": {
                "must": [
                    {
                        "field": "name",
                        "operator": "eq",
                        "value": "example"
                    },
                    {
                        "ids": [1, 2, 3]
                    }
                ],
                "any": [
                    {
                        "field": "age",
                        "operator": "gt",
                        "value": 18
                    }
                ]
            },
            "return_similarity": true
        }
        response = requests.post(url, json=data)
        print(response.json())

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
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key=condition['field'],
                        matcher=MatchField(
                            value=condition['value'],
                            comparator=getattr(operator, condition['operator'])
                        )
                    ) if 'field' in condition and 'operator' in condition
                    else IDCondition(matcher=MatchID(ids=condition['ids']))
                    for condition in data['query_filter']['must']
                ] if data['query_filter']['must'] is not None else None,
                any=[
                    FieldCondition(
                        key=condition['field'],
                        matcher=MatchField(
                            value=condition['value'],
                            comparator=getattr(operator, condition['operator'])
                        )
                    ) if 'field' in condition and 'operator' in condition
                    else IDCondition(matcher=MatchID(ids=condition['ids']))
                    for condition in data['query_filter']['any']
                ] if data['query_filter']['any'] is not None else None
            )

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


@app.route('/bulk_add_items', methods=['POST'])
def bulk_add_items():
    """Bulk add items to a collection.
        .. versionadded:: 0.3.2

    Example:
        1) use curl
        curl -X POST http://localhost:7637/bulk_add_items \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "items": [
                  {
                      "vector": [0.1, 0.2, 0.3, 0.4],
                      "id": 1,
                      "field": {
                          "name": "example",
                          "age": 18
                        }
                    },
                    {
                        "vector": [0.5, 0.6, 0.7, 0.8],
                        "id": 2,
                        "field": {
                            "name": "example",
                            "age": 20
                        }
                    }
                ]
            }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/bulk_add_items'

        data = {
            "collection_name": "example_collection",
            "items": [
                {
                    "vector": [0.1, 0.2, 0.3, 0.4],
                    "id": 1,
                    "field": {
                        "name": "example",
                        "age": 18
                    }
                },
                {
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "id": 2,
                    "field": {
                        "name": "example",
                        "age": 20
                    }
                }
            ]
        }
        response = requests.post(url, json=data)
        print(response.json())

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        items = []
        for item in data['items']:
            items.append((item['vector'], item.get('id', None), item.get('field', None)))

        ids = collection.bulk_add_items(items)

        return Response(json.dumps({
            'status': 'success', 'params': {
                'collection_name': data['collection_name'], 'items': [
                    {'vector': item[0], 'id': item[1], 'field': item[2]} for item in items
                ], 'ids': ids
            }
        }, sort_keys=False),
            mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/commit', methods=['POST'])
def commit():
    """Commit the database.
        .. versionadded:: 0.3.2

    Example:
        1) use curl
        curl -X POST http://localhost:7637/commit \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/commit'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

    Returns:
        dict: The status of the operation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        my_vec_db = MinVectorDBLocalClient(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/collection_shape \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/collection_shape'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/set_environment \
         -H "Content-Type: application/json" \
         -d '{
              "MVDB_LOG_LEVEL": "ERROR",
              "MVDB_LOG_PATH": "/path/to/log",
              "MVDB_TRUNCATE_LOG": "true",
              "MVDB_LOG_WITH_TIME": "true",
              "MVDB_KMEANS_EPOCHS": "100",
              "MVDB_QUERY_CACHE_SIZE": "100",
              "MVDB_DATALOADER_BUFFER_SIZE": "1000"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/set_environment'
        data = {
            "MVDB_LOG_LEVEL": "ERROR",
            "MVDB_LOG_PATH": "/path/to/log",
            "MVDB_TRUNCATE_LOG": "true",
            "MVDB_LOG_WITH_TIME": "true",
            "MVDB_KMEANS_EPOCHS": "100",
            "MVDB_QUERY_CACHE_SIZE": "100",
            "MVDB_DATALOADER_BUFFER_SIZE": "1000"
        }

        response = requests.post(url, json=data)
        print(response.json())

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

    Example:
        1) use curl
        curl -X GET http://localhost:7637/get_environment -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/get_environment'
        response = requests.get(url)
        print(response.json())

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/get_collection_query_report \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/get_collection_query_report'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/get_collection_status_report \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/get_collection_status_report'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/is_collection_exists \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/is_collection_exists'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/get_collection_config \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/get_collection_config'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/save_commit_msg \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "last_commit_time": "last_commit_time",
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/save_commit_msg'
        data = {
            "collection_name": "example_collection",
            "last_commit_time": "last_commit_time",
        }
        response = requests.post(url, json=data)
        print(response.json())

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
            portalocker.lock(file, portalocker.LOCK_EX)
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

    Example:
        1) use curl
        curl -X POST http://localhost:7637/get_commit_msg \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection"
             }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/get_commit_msg'
        data = {
            "collection_name": "example_collection"
        }
        response = requests.post(url, json=data)
        print(response.json())

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
            commit_msg = None

        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'commit_msg': commit_msg
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
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
        serve(app, host=args.host, port=args.port, threads=args.threads)


if __name__ == '__main__':
    main()
