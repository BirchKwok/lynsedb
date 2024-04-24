"""使用flask实现的http接口, 用于向外部提供服务"""
import json
import operator
import os
from pathlib import Path

import numpy as np
import yaml

from flask import Flask, request, jsonify, Response

from min_vec.api.high_level import MinVectorDB
from min_vec.structures.filter import Filter, FieldCondition, MatchField, IDCondition, MatchID

os.environ['MVDB_LOG_LEVEL'] = 'ERROR'

app = Flask(__name__)

# 构建配置文件的路径
config_path = Path(os.path.expanduser('~/.MinVectorDB/config.yaml'))
default_root_path = Path(os.path.expanduser('~/.MinVectorDB/data/min_vec_db')).as_posix()


def generate_config(root_path: str = default_root_path, config_path: Path = config_path):
    """Generate a default configuration file.
        .. versionadded:: 0.3.1
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


@app.route('/required_collection', methods=['POST'])
def required_collection():
    """Create a collection in the database.
        .. versionadded:: 0.3.1

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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
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
        .. versionadded:: 0.3.1

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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
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
        .. versionadded:: 0.3.1

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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        my_vec_db.drop_database()

        return Response(json.dumps({
            'status': 'success'
        }, sort_keys=False),
            mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/database_exists', methods=['GET'])
def database_exists():
    """Check if the database exists.
        .. versionadded:: 0.3.1

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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        return Response(json.dumps({
            'status': 'success', 'params': {
                    'exists': True if my_vec_db.STATUS == 'INITIALIZED' else False
                }
        },
       sort_keys=False), mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/show_collections', methods=['GET'])
def show_collections():
    """Show all collections in the database.
        .. versionadded:: 0.3.1

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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        collections = my_vec_db.show_collections()
        return Response(json.dumps({'status': 'success', 'params': {'collections': collections}},
                                   sort_keys=False), mimetype='application/json')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_item', methods=['POST'])
def add_item():
    """Add an item to a collection.
        .. versionadded:: 0.3.1

    Example:
        1) use curl
        curl -X POST http://localhost:7637/add_item \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "item": {
                  "vector": [0.1, 0.2, 0.3, 0.4],
                  "id": 1,
                  "fields": {
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
                "fields": {
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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        with collection.insert_session():
            collection.add_item(vector=np.array(data['item']['vector']), id=data['item'].get('id', None),
                                field=data['item'].get('fields', None))

        return Response(json.dumps(
            {
                'status': 'success', 'params': {'collection_name': data['collection_name'], 'item': {
                        'vector': data['item']['vector'], 'id': data['item'].get('id', None),
                        'fields': data['item'].get('fields', None)
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
        .. versionadded:: 0.3.1

    Example:
        1) use curl
        curl -X POST http://localhost:7637/query \
         -H "Content-Type: application/json" \
         -d '{
              "collection_name": "example_collection",
              "vector": [0.1, 0.2, 0.3, 0.4],
              "k": 10,
              'distance': 'cosine',
              "query_filter": {
                  "must": [
                      {
                          "field": "name",
                          "operator": "eq",
                          "value": "example"
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
                return_similarity: true
            }' -w "\n"

        2) use python requests
        import requests

        url = 'http://localhost:7637/query'
        data = {
            "collection_name": "example_collection",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "k": 10,
            'distance': 'cosine',
            "query_filter": {
                "must": [
                    {
                        "field": "name",
                        "operator": "eq",
                        "value": "example"
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
            return_similarity: true
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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])

        query_filter = Filter(
            must=[
                FieldCondition(
                    key=condition['field'],
                    matcher=MatchField(
                        value=condition['value'],
                        comparator=getattr(operator, condition['operator'])
                    )
                ) for condition in data['query_filter']['must']
            ] if 'must' in data['query_filter'] else None,
            any=[
                FieldCondition(
                    key=condition['field'],
                    matcher=MatchField(
                        value=condition['value'],
                        comparator=getattr(operator, condition['operator'])
                    )
                ) for condition in data['query_filter']['any']
            ] if 'any' in data['query_filter'] else None
        )
        ids, scores = collection.query(vector=data['vector'], k=data['k'],
                                       query_filter=query_filter,
                                       distance=data.get('distance', 'cosine'),
                                       return_similarity=data.get('return_similarity', False))

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
        .. versionadded:: 0.3.1

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
                      "fields": {
                          "name": "example",
                          "age": 18
                        }
                    },
                    {
                        "vector": [0.5, 0.6, 0.7, 0.8],
                        "id": 2,
                        "fields": {
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
                    "fields": {
                        "name": "example",
                        "age": 18
                    }
                },
                {
                    "vector": [0.5, 0.6, 0.7, 0.8],
                    "id": 2,
                    "fields": {
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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        items = []
        for item in data['items']:
            items.append((item['vector'], item.get('id', None), item.get('fields', None)))

        with collection.insert_session():
            collection.bulk_add_items(items)

        return Response(json.dumps({
            'status': 'success', 'params': {
                'collection_name': data['collection_name'], 'items': [
                    {'vector': item[0], 'id': item[1], 'fields': item[2]} for item in items
                ]
            }
        }, sort_keys=False),
            mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/collection_shape', methods=['POST'])
def collection_shape():
    """Get the shape of a collection.
        .. versionadded:: 0.3.1

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
        my_vec_db = MinVectorDB(root_path=config['root_path'])
        collection = my_vec_db.get_collection(data['collection_name'])
        return Response(json.dumps({'status': 'success', 'params': {
            'collection_name': data['collection_name'], 'shape': collection.shape
        }}, sort_keys=False), mimetype='application/json')
    except KeyError as e:
        return jsonify({'error': f'Missing required parameter {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    global config_path, default_root_path, config, root_path
    import argparse

    parser = argparse.ArgumentParser(description='Start the MinVec HTTP API server.')
    parser.add_argument('--host', default='127.0.0.1', help='The host to bind to.')
    parser.add_argument('--port', default=7637, type=int, help='The port to bind to.')
    parser.add_argument('--config', default=config_path, help='The path to the configuration file.')
    parser.add_argument('--root_path', default=default_root_path, help='The path to the database root directory.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('run', help='Run the HTTP API server.')

    version = '0.3.1'

    parser.add_argument('--version', action='version', version=version)

    args = parser.parse_args()

    config_path = args.config
    root_path = args.root_path
    config_path, root_path, config = generate_config(root_path=root_path, config_path=config_path)

    if args.run:
        app.run(host=args.host, port=args.port, debug=args.debug)
