import os
import base64
from flask import Flask, request
from flask_cors import CORS

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
    print(type(data), ' ->\n ', data_bytes)
    data_str = data_bytes.decode()
    print(type(data_str), ' ->\n ', data_str)

    mode = 'w'
    if firstChunk: mode = 'w'
    else: mode = 'a'

    data_file = open(filename, mode)
    n = data_file.write(data_str)
    data_file.close()
    
    print(type(n), ' ->\n ', n)

    return {
        'message': 'Upload file successfully'
    }, 200

if __name__ == '__main__':
    app.run(debug=True)