import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
import datetime
import sys

#Funcao auxiliar para estimar uso de memoria
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def print_and_exit(msg, memory_usage):
    print(msg, memory_usage)
    sys.exit()

colspecs = [(2,10),(12,24),(56,69),(108,121)]

names = ["DataPregao", "Ticker", "PrecoAbertura", "PrecoUltimoNegocio"]

dataset_2010 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2011 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2012 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2013 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2014 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2015 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2015 = pd.read_fwf('data/COTAHIST_A2015.TXT', header=None, colspecs=colspecs, names=names)
dataset_2016 = pd.read_fwf('data/COTAHIST_A2016.TXT', header=None, colspecs=colspecs, names=names)
dataset_2017 = pd.read_fwf('data/COTAHIST_A2018.TXT', header=None, colspecs=colspecs, names=names)
dataset_2018 = pd.read_fwf('data/COTAHIST_A2018.TXT', header=None, colspecs=colspecs, names=names)

#Remover o cabecalho e trailer
dataset_2010.drop([0], inplace=True)
dataset_2010 = dataset_2010[:-1]

dataset_2011.drop([0], inplace=True)
dataset_2011 = dataset_2011[:-1]

dataset_2012.drop([0], inplace=True)
dataset_2012 = dataset_2012[:-1]

dataset_2013.drop([0], inplace=True)
dataset_2013 = dataset_2013[:-1]

dataset_2014.drop([0], inplace=True)
dataset_2014 = dataset_2014[:-1]

dataset_2015.drop([0], inplace=True)
dataset_2015 = dataset_2015[:-1]

dataset_2016.drop([0], inplace=True)
dataset_2016 = dataset_2016[:-1]

dataset_2017.drop([0], inplace=True)
dataset_2017 = dataset_2017[:-1]

dataset_2018.drop([0], inplace=True)
dataset_2018 = dataset_2018[:-1]

frames = [dataset_2010, dataset_2011, dataset_2012, dataset_2013, dataset_2014, dataset_2015, dataset_2016, dataset_2017, dataset_2018]
#frames = [dataset_2015, dataset_2016, dataset_2017, dataset_2018]
data_frame = pd.concat(frames)

#Otimizando o uso de memoria
data_frame['DataPregao'] = data_frame['DataPregao'].astype('category')
data_frame['Ticker'] = data_frame['Ticker'].astype('category')

data_frame_float = data_frame.select_dtypes(include=['float'])
converted_float = data_frame_float.apply(pd.to_numeric,downcast='float')

data_frame[converted_float.columns] = converted_float

# Here we normalize date and ticker
data_frame['DataPregao'] = data_frame['DataPregao'].cat.codes
data_frame['Ticker'] = data_frame['Ticker'].cat.codes

# Slicing the data_frame to define X and Y
X = data_frame.values[:, 0:3]
Y = data_frame.values[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 100)

rgr_knn = KNeighborsRegressor(n_neighbors=6, weights='distance')
rgr_knn.fit(X_train, y_train)
y_pred_knn = rgr_knn.predict(X_test)
print("KRR accuracy is ", explained_variance_score(y_test,y_pred_knn)*100)

X_test_knn_2 = [[482, 46229, 2308]]
y_pred_knn_2 = rgr_knn.predict(X_test_knn_2)
print(y_pred_knn_2)

X_test_knn_3 = [[481, 46229, 2329]]
y_pred_knn_3 = rgr_knn.predict(X_test_knn_3)
print(y_pred_knn_3)