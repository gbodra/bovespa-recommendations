import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import explained_variance_score
import datetime
import sys

colspecs = [
    (2,10),(12,24),(56,69),(69,82),(82,95),(95,108),
    (108,121),(121,134),(134,147),(147,152),(152,170),(170,188)
]

names = [
    "DataPregao", "Ticker", "PrecoAbertura", "PrecoMax", "PrecoMin",
    "PrecoMed", "PrecoUltimoNegocio", "PrecoMelhorOfertaC", "PrecoMelhorOfertaV",
    "TotalNegocios", "TotalTitulosNegociados", "VolumeTitulosNegociados"
]
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
data_frame = pd.concat(frames)

#Transformar a data em date time
data_frame['DataPregao'] = pd.to_datetime(data_frame['DataPregao'])

# Here we normalize date and ticker
data_frame['DataPregao'] = data_frame['DataPregao'].astype('category')
data_frame['DataPregao'] = data_frame['DataPregao'].cat.codes
data_frame['Ticker'] = data_frame['Ticker'].astype('category')
data_frame['Ticker'] = data_frame['Ticker'].cat.codes

data_frame = data_frame[['DataPregao', 'Ticker', 'PrecoAbertura', 'PrecoUltimoNegocio']]

# Slicing the data_frame to define X and Y
X = data_frame.values[:, 0:3]
Y = data_frame.values[:,3]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 100)

#print(X_test)
#sys.exit()
rgr = DecisionTreeRegressor(random_state=100)
rgr.fit(X_train, y_train)

y_pred = rgr.predict(X_test)

print "Decision regression tree accuracy is ", explained_variance_score(y_test,y_pred)*100

#2018-12-14 -> 482 / PETR4 -> 46229
X_test_2 = [[482, 46229, 2308]]
y_pred_2 = rgr.predict(X_test_2)
print(y_pred_2)

X_test_3 = [[481, 46229, 2329]]
y_pred_3 = rgr.predict(X_test_3)
print(y_pred_3)

#-------------NN------------
rgr_nn = MLPRegressor()
rgr_nn.fit(X_train, y_train)

y_pred_nn = rgr_nn.predict(X_test)

print "Neural network regression tree accuracy is ", explained_variance_score(y_test,y_pred_nn)*100

#2018-12-14 -> 482 / PETR4 -> 46229
X_test_nn_2 = [[482, 46229, 2308]]
y_pred_nn_2 = rgr_nn.predict(X_test_nn_2)
print(y_pred_nn_2)

X_test_nn_3 = [[481, 46229, 2329]]
y_pred_nn_3 = rgr_nn.predict(X_test_nn_3)
print(y_pred_nn_3)