import pandas as pd

def getHeadersFile(file_name, file_ext):
    if file_ext == 'csv':
        data = pd.read_csv(file_name + '.' + file_ext, sep = ',')
        if data.shape[1] > 1: return list(data.columns)
        data = pd.read_csv(file_name + '.' + file_ext, sep=';')
        return list(data.columns)
    elif file_ext == 'json':
        data = pd.read_json(file_name + '.' + file_ext)
        return list(data.columns)
    elif file_ext == 'xls':
        data = pd.read_excel(file_name + '.' + file_ext)
        return list(data.columns)

def getDataFrameFile(file_name, file_ext):
    if file_ext == 'csv':
        data = pd.read_csv(file_name + '.' + file_ext, sep = ',')
        if data.shape[1] > 1: return data
        data = pd.read_csv(file_name + '.' + file_ext, sep=';')
        return data
    elif file_ext == 'json':
        data = pd.read_json(file_name + '.' + file_ext)
        return data
    elif file_ext == 'xls':
        data = pd.read_excel(file_name + '.' + file_ext)
        return data