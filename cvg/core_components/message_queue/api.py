import flask
from flask import jsonify

from cvg.core_components.message_queue import MessageQueue

queue_app = flask.Flask(__name__)


@queue_app.route('/init', methods=['GET'])
def init():
    return jsonify({"status": "success", "message": "Convergence Message queue is initialized."})


@queue_app.route('/put', methods=['POST'])
def put():
    data = flask.request.json
    queue = MessageQueue(data['file_path'])
    queue.put(data['item'], data['partition'])
    return 'OK'


@queue_app.route('/get', methods=['POST'])
def get():
    data = flask.request.json
    queue = MessageQueue(data['file_path'])
    if queue.empty(data['partition']):
        return flask.jsonify({})

    item = queue.get(data['partition'])
    return flask.jsonify(item)


@queue_app.route('/get_batch', methods=['POST'])
def get_batch():
    data = flask.request.json
    queue = MessageQueue(data['file_path'])
    items = queue.get_batch(data['partition'], data.get('batch_size'))
    return flask.jsonify(items)


@queue_app.route('/stop', methods=['POST'])
def stop():
    data = flask.request.json
    queue = MessageQueue(data['file_path'])
    queue.stop()
    return 'OK'

