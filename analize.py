import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from fileManagment import getDataFrameFile

def linearRegression(x_field, y_field, pred, options, file_ext):    
    df = getDataFrameFile('data', file_ext)

    x = np.asarray(df[x_field]).reshape(-1, 1)
    y_true = df[y_field]

    regr = LinearRegression()
    regr.fit(x, y_true)
    y_pred = regr.predict(x)

    # VALORES
    coef = regr.coef_
    intercept = regr.intercept_
    r2 = r2_score(y_true, y_pred)
    error = mean_squared_error(y_true, y_pred)
    print('coef_', coef)
    print('intercept', intercept)
    print('r2', r2)
    print('error', error)

    # PREDICCION
    predict = regr.predict([[int(pred)]])
    print('predict', predict)

    # GRAFICA
    plt.scatter(x, y_true, color = 'black')
    plt.plot(x, y_pred, color = 'blue')
    plt.savefig('grafico.jpg')
    pass

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