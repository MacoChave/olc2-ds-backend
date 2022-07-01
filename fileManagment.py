import pandas as pd

def getHeadersFile(file_name, file_ext):
    if file_ext == 'csv':
        data = pd.read_csv(file_name + '.' + file_ext)
        return list(data.columns.values)
    elif file_ext == 'json':
        data = pd.read_json(file_name + '.' + file_ext)
        return list(data.columns.values)
    else:
        data = pd.read_excel(file_name + '.' + file_ext)
        return list(data.columns.values)