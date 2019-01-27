import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
import sys
import os

#Funcao auxiliar para estimar uso de memoria
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

#Funcao para exibir mensagem e terminar o programa
def print_and_exit(msg, memory_usage):
    print(msg, memory_usage)
    sys.exit()

#Funcao para carregar e preparar os arquivos
def load_and_prepare_files():
    print("********************CARREGANDO ARQUIVOS********************")
    colspecs = [(2,10),(12,24),(56,69),(108,121)]
    names = ["DataPregao", "Ticker", "PrecoAbertura", "PrecoUltimoNegocio"]
    frames = []

    for root, dirs, files in os.walk("./data"):
        for filename in files:
            print("Carregando o arquivo: ", filename)
            temp_dataset = pd.read_fwf('data/' + filename, header=None, colspecs=colspecs, names=names)
            temp_dataset.drop([0], inplace=True)
            temp_dataset = temp_dataset[:-1]
            frames.append(temp_dataset)

    dataset = pd.concat(frames)
    print("Todos os arquivos carregados e tratados\n")
    return dataset

def otimiza_memoria(data_frame):
    print("********************OTIMIZANDO MEMORIA********************")
    print("Total de memoria: ", mem_usage(data_frame))
    data_frame['DataPregao'] = data_frame['DataPregao'].astype('category')
    data_frame['Ticker'] = data_frame['Ticker'].astype('category')
    data_frame_float = data_frame.select_dtypes(include=['float'])
    converted_float = data_frame_float.apply(pd.to_numeric,downcast='float')
    data_frame[converted_float.columns] = converted_float

    print("Total de memoria: ", mem_usage(data_frame), "\n")
    return data_frame

data_frame = load_and_prepare_files()
data_frame = otimiza_memoria(data_frame)

print("********************EXECUTANDO MODELO********************")
# Here we normalize date and ticker
data_frame['DataPregao'] = data_frame['DataPregao'].cat.codes
data_frame['Ticker'] = data_frame['Ticker'].cat.codes

#Ordena o dataset para evitar problemas na estimativa
data_frame.sort_values(by=['DataPregao', 'Ticker'], inplace=True)

# Slicing the data_frame to define X and Y
X = data_frame.values[:, 0:3]
Y = data_frame.values[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 100)

rgr_knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
rgr_knn.fit(X_train, y_train)
y_pred_knn = rgr_knn.predict(X_test)
print("KRR accuracy is ", explained_variance_score(y_test,y_pred_knn)*100)

X_test_knn_2 = [[482, 46229, 2308], [481, 46229, 2329], [480, 46229, 2374]]
y_pred_knn_2 = rgr_knn.predict(X_test_knn_2)
print(y_pred_knn_2)

X_test_knn_3 = [[481, 46229, 2329]]
y_pred_knn_3 = rgr_knn.predict(X_test_knn_3)
print(y_pred_knn_3)