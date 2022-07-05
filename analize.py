import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64

from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
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
        func_tendencia = f'y = {coef} x + {intercept}'
        print('FUNCTION -> ', func_tendencia)

    # PREDICCION
    pred_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(pred_match) != 0):
        pred_tendencia = regr.predict([[int(pred)]])[0]
        pred_tendencia = round(pred_tendencia, 2)
        print('PREDICT -> ', pred_tendencia)

    # GRAFICA
    ax = plt.subplot()
    plot_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(plot_match) != 0):
        ax.scatter(x, y_true, color = '#1594AD')
    if (len(trend_match) != 0):
        ax.plot(x, y_pred, color = '#AD5203')
    if (len(trend_match) != 0 or len(plot_match) != 0):
        try:
            print('Guardar imagen..')
            plt.title(f'Regresion lineal. Predicción a {pred}.')
            plt.xlabel(x_field)
            plt.ylabel(y_field)
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
        func_tendencia = f'y = {intercept}'
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
    ax = plt.subplot()
    plot_match = [s for s in options if "Predicción de tendencia" in s]
    if (len(plot_match) != 0):
        ax.scatter(x, y_true, color = '#1594AD')
    if (len(trend_match) != 0):
        ax.plot(x, y_pred, color = '#AD5203')
    if (len(trend_match) != 0 or len(plot_match) != 0):
        try:
            print('Guardar imagen..')
            plt.title(f'Regresion polinomial de grado {len(coef) - 1}. Predicción a {pred}.')
            plt.xlabel(x_field)
            plt.ylabel(y_field)
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

def getFeatures(df, y_field):
    df = df.drop([y_field], axis = 1)
    field_match = [s for s in df.head() if "NO" in s]
    if len(field_match) == 1: df = df.drop(['NO'], axis = 1)
    headers = df.head()
    columns = headers.columns
    print(columns)

    fields = []
    le = LabelEncoder()
    for col in columns:
        col_list = df[col].tolist()
        col_encoded = le.fit_transform(col_list)
        fields.append(col_encoded)
    
    features = list(zip(*fields))
    return [features, le]

def gaussianNB(y_field, pred, file_ext):
    df = getDataFrameFile('data', file_ext)
    
    y_true = df[y_field]

    [features, le] = getFeatures(df, y_field)
    # label = le.fit_transform(y_true)

    model = GaussianNB()
    model.fit(features, y_true)

    val = []
    for p in pred:
        val.append(int(p))

    predict = model.predict([val])
    predict = predict[0]
    
    return predict

def decisionTree(y_field, pred, file_ext):
    df = getDataFrameFile('data', file_ext)
    str_image = ''

    y_true = df[y_field]

    [features, le] = getFeatures(df, y_field)
    # label = le.fit_transform(y_true)

    clf = DecisionTreeClassifier()
    clf.fit(features, y_true)
    plot_tree(clf, filled = True)
    plt.savefig('grafico.jpg')
    print('Imagen a base 64...')
    with open("grafico.jpg", "rb") as img_file:
        str_image = base64.b64encode(img_file.read())
    
    predict = clf.predict([pred])
    predict = predict[0]
    
    return [predict, str_image]

def neuronalNetwork(y_field, layers_size, iteraciones, pred, file_ext):
    df = getDataFrameFile('data', file_ext)
    
    y_true = df[y_field]

    [features, le] = getFeatures(df, y_field)
    # label = le.fit_transform(y_true)

    x_train, x_test, y_train, y_test = train_test_split(features, y_true, test_size = 0.2)

    scaler = StandardScaler().fit(x_train)
    x_test = scaler.transform(x_test)

    layers = []
    for l in layers_size:
        layers.append(int(l))
    hidden_layer_sizes = tuple(layers)
    model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, max_iter = int(iteraciones), solver = 'lbfgs', verbose = 10, tol = 0.000001, random_state = 0)
    model.fit(x_train, y_train)

    val = []
    for p in pred:
        val.append(int(p))
    predict = model.predict([val])
    predict = predict[0]

    return predict