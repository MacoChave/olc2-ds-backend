import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64

from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from fileManagment import getDataFrameFile

def linearRegression(x_field, y_field, pred, options, file_ext):
    df = getDataFrameFile('data', file_ext)
    func_tendencia = ''
    pred_tendencia = 0.0
    str_image = ''

    x = np.asarray(df[x_field]).reshape(-1, 1)
    y_true = df[y_field]

    regr = LinearRegression()
    regr.fit(x, y_true)
    y_pred = regr.predict(x)

    # VALORES
    coef = regr.coef_[0]
    intercept = regr.intercept_
    r2 = r2_score(y_true, y_pred)
    error = mean_squared_error(y_true, y_pred)
    # print('COEF -> ', coef)
    # print('INTERCEPT -> ', intercept)
    # print('R2 -> ', r2)
    # print('ERROR -> ', error)

    # FUNCIÓN DE TENDENCIA
    trend_match = [s for s in options if "Función de tendencia" in s]
    if (len(trend_match) != 0):
        func_tendencia = f'y(x) = {coef} x + {intercept}'
        # print('FUNCTION -> ', func_tendencia)

    # PREDICCION
    pred_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(pred_match) != 0):
        pred_tendencia = regr.predict([[int(pred)]])[0]
        # print('PREDICT -> ', pred_tendencia)

    # GRAFICA
    plot_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(plot_match) != 0):
        plt.scatter(x, y_true, color = '#1594AD')
    if (len(trend_match) != 0):
        plt.plot(x, y_pred, color = '#AD5203')
    if (len(trend_match) != 0 or len(plot_match) != 0):
        plt.savefig('grafico.jpg')
        with open("grafico.jpg", "rb") as img_file:
            str_image = base64.b64encode(img_file.read())
    
    return [
        func_tendencia,
        pred_tendencia,
        str_image
    ]


def polinomialRegression(x_field, y_field, pred, options, file_ext):
    df = getDataFrameFile('data', file_ext)
    print(df.head())
    pass

def gaussianNB(file_ext):
    df = getDataFrameFile('data', file_ext)
    print(df.head())
    pass

def decisionTree(file_ext):
    df = getDataFrameFile('data', file_ext)
    print(df.head())
    pass

def neuronalNetwork(file_ext):
    df = getDataFrameFile('data', file_ext)
    print(df.head())
    pass