import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from fileManagment import getHeadersFile
from analize import linearRegression, polinomialRegression, gaussianNB, decisionTree, neuronalNetwork

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

@app.route('/analize', methods=['POST'])
def analize():
    # try:
    data = request.json
    print(data)
    ext = data['ext']
    config_algorithm = data['config']['algorithm']
    config_option = data['config']['option']
    param_x = data['params']['independiente']
    param_y = data['params']['dependiente']
    param_pred = data['params']['time']

    if config_algorithm == 'Regresión lineal': 
        print('Regresión lineal...')
        res = linearRegression(param_x, param_y, param_pred, config_option, ext)
        return jsonify({
            'func': res[0],
            'pred': res[1],
            'imageB64': str(res[2])
        })
    elif config_algorithm == 'Regresión polinomial': 
        print('Regresión polinomial...')
        res = polinomialRegression(param_x, param_y, param_pred, config_option, ext)
        return jsonify({
            'func': res[0],
            'pred': res[1],
            'imageB64': str(res[2])
        })
    elif config_algorithm == 'Clasificador gaussiano': 
        res = gaussianNB(param_y, param_pred, ext)
        return jsonify({
            'func': '',
            'pred': str(res),
            'imageB64': ''
        })
    elif config_algorithm == 'Clasificador de árboles de decisión':
        res = decisionTree(param_y, param_pred, ext)
        return jsonify({
            'func': '',
            'pred': str(res[0]),
            'imageB64': str(res[1])
        })
    elif config_algorithm == 'Redes neuronales':
        param_layer = data['params']['layers']
        param_iter = data['params']['iteracion']
        res = neuronalNetwork(param_y, param_layer, param_iter, param_pred, ext)
        return jsonify({
            'func': '',
            'pred': str(res),
            'imageB64': ''
        })
    else:
        return jsonify({
            'func': '',
            'pred': 0.0,
            'imageB64': ''
        })

if __name__ == '__main__':
    app.run(debug=True)