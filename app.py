import os
import base64
from flask import Flask, request
from flask_cors import CORS
from fileManagment import getHeadersFile

app = Flask(__name__)
cors = CORS(app)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    return {
        'message': 'DataScience API'
    }, 200

@app.route('/upload', methods=["POST"])
def upload():
    args = request.args
    data = request.data

    args_name = args['name']
    args_type = args_name.split('.')[-1]
    args_chunkIndex = int(args['chunkIndex'])
    args_totalChunks = int(args['totalChunks'])

    filename = 'data.' + args_type

    firstChunk = args_chunkIndex == 0
    lastChunk = args_chunkIndex == args_totalChunks - 1

    data_bytes = base64.b64decode(data)
    data_str = data_bytes.decode()

    mode = 'w'
    if firstChunk: mode = 'w'
    else: mode = 'a'

    data_file = open(filename, mode)
    n = data_file.write(data_str)
    data_file.close()

    return {
        'message': 'Upload file successfully'
    }, 200

@app.route('/headers', methods=["GET"])
def headers():
    args = request.args

    file_name = 'data'
    args_ext = args['ext']

    header_list = getHeadersFile(file_name, args_ext)
    return {
        'header': header_list
    }, 200


if __name__ == '__main__':
    app.run(debug=True)