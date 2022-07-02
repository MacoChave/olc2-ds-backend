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
    coef = round(coef, 2)
    intercept = regr.intercept_
    intercept = round(intercept, 2)
    r2 = r2_score(y_true, y_pred)
    error = mean_squared_error(y_true, y_pred)
    print('COEF -> ', coef)
    print('INTERCEPT -> ', intercept)
    print('R2 -> ', r2)
    print('ERROR -> ', error)

    # FUNCIÓN DE TENDENCIA
    trend_match = [s for s in options if "Función de tendencia" in s]
    if (len(trend_match) != 0):
        func_tendencia = f'y(x) = {coef} x + {intercept}'
        print('FUNCTION -> ', func_tendencia)

    # PREDICCION
    pred_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(pred_match) != 0):
        pred_tendencia = regr.predict([[int(pred)]])[0]
        pred_tendencia = round(pred_tendencia, 2)
        print('PREDICT -> ', pred_tendencia)

    # GRAFICA
    plot_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(plot_match) != 0):
        plt.scatter(x, y_true, color = '#1594AD')
    if (len(trend_match) != 0):
        plt.plot(x, y_pred, color = '#AD5203')
    if (len(trend_match) != 0 or len(plot_match) != 0):
        try:
            print('Guardar imagen..')
            plt.savefig('grafico.jpg')
            print('Imagen a base 64...')
            with open("grafico.jpg", "rb") as img_file:
                str_image = base64.b64encode(img_file.read())
        except: 
            print('Failed to save figure')

    print('Analisis finalizado...')
    return [
        func_tendencia,
        pred_tendencia,
        str_image
    ]

def polinomialRegression(x_field, y_field, pred, options, file_ext):
    df = getDataFrameFile('data', file_ext)
    func_tendencia = ''
    pred_tendencia = 0.0
    str_image = ''

    x = np.asarray(df[x_field]).reshape(-1, 1)
    y_true = df[y_field]

    # DEGREE 2
    print(f'degree = 2')
    pf = PolynomialFeatures(degree = 2)
    x_ = pf.fit_transform(x)
    regr = LinearRegression()
    regr.fit(x_, y_true)
    y_pred = regr.predict(x_)

    # VALORES
    coef = regr.coef_
    intercept = regr.intercept_
    r2 = r2_score(y_true, y_pred)
    error = mean_squared_error(y_true, y_pred)

    last_r2 = r2

    # ITERACION DEGREE 3, 4, 5
    for i in [3, 4, 5]:
        print(f'degree = {i}')
        loop_pf = PolynomialFeatures(degree = i)
        loop_x_ = loop_pf.fit_transform(x)
        loop_regr = LinearRegression()
        loop_regr.fit(loop_x_, y_true)
        loop_y_pred = loop_regr.predict(loop_x_)

        # VALORES
        loop_coef = loop_regr.coef_
        loop_intercept = loop_regr.intercept_
        loop_r2 = r2_score(y_true, loop_y_pred)
        if (round(last_r2, 2) < round(loop_r2, 2)):
            print(f'{round(last_r2, 2)} < {round(loop_r2, 2)}')
            pf = loop_pf
            x_ = loop_x_
            regr = loop_regr
            r2 = loop_r2
            last_r2 = loop_r2
            coef = loop_coef
            intercept = loop_intercept
            y_pred = loop_y_pred
            break


    print('COEF -> ', coef)
    print('INTERCEPT -> ', intercept)
    print('R2 -> ', r2)

    # FUNCIÓN DE TENDENCIA
    trend_match = [s for s in options if "Función de tendencia" in s]
    if (len(trend_match) != 0):
        func_tendencia = f'f(y) = {intercept}'
        for i in range(1, len(coef)):
            func_tendencia = f'{func_tendencia} + ({coef[i * -1]}) x^{i}'
        print('FUNCTION -> ', func_tendencia)

    # PREDICCION
    pred_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(pred_match) != 0):
        x_new_min = int(pred)
        x_new_max = int(pred)
        x_new = np.linspace(x_new_min, x_new_max, num = 1)
        x_new = x_new[:, np.newaxis]
        x_ = pf.fit_transform(x_new)
        print('predict: ', regr.predict(x_)[-1])
        pred_tendencia = regr.predict(x_)[-1]
        pred_tendencia = round(pred_tendencia, 2)
        print('PREDICT -> ', pred_tendencia)

    # GRAFICA
    plot_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(plot_match) != 0):
        plt.scatter(x, y_true, color = '#1594AD')
    if (len(trend_match) != 0):
        plt.plot(x, y_pred, color = '#AD5203')
    if (len(trend_match) != 0 or len(plot_match) != 0):
        try:
            print('Guardar imagen..')
            plt.savefig('grafico.jpg')
            print('Imagen a base 64...')
            with open("grafico.jpg", "rb") as img_file:
                str_image = base64.b64encode(img_file.read())
        except: 
            print('Failed to save figure')

    print('Analisis finalizado...')

    return [
        func_tendencia,
        pred_tendencia,
        str_image
    ]

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