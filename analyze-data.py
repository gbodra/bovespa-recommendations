from pymongo import MongoClient
import pandas as pd
import numpy as np
import pprint
import custom_functions
from datetime import datetime
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import sys

client = MongoClient('localhost', 27017)

db = client['bovespa']

cotacoes_PETR4 = db.cotacao_diaria.find({"Ticker": "PETR4"})

df =  pd.DataFrame(list(cotacoes_PETR4))
df['FechamentoDiaAnterior'] = df.PrecoUltimoNegocio.shift(1)

df['EMA_200'] = df['PrecoUltimoNegocio'].ewm(span=200, adjust=False).mean()
df['EMA_34'] = df['PrecoUltimoNegocio'].ewm(span=34, adjust=False).mean()

#Calculo do RSI ou IFR
# df['Gain'] = np.where(df['PrecoAbertura'] < df['PrecoUltimoNegocio'], df['PrecoUltimoNegocio'] - df['PrecoAbertura'], 0)
# df['Loss'] = np.where(df['PrecoAbertura'] > df['PrecoUltimoNegocio'], df['PrecoUltimoNegocio'] - df['PrecoAbertura'], 0)
# df['Avg_Gain'] = df['Gain'].rolling(window=14).mean()
# df['Avg_Gain'] = df['Gain'].rolling(window=14).mean()


df.fillna(value=0, inplace=True)
df.drop(columns=['Ticker', '_id'], inplace=True)

df = df[
        [
            'DataPregao', 'PrecoAbertura', 'PrecoMax', 'PrecoMin'
            # , 'FechamentoDiaAnterior'
            # , 'EMA_200'
            # , 'EMA_34'
            , 'PrecoUltimoNegocio'
        ]
    ]

# pprint.pprint(df.tail(10))

#Ordena o dataset para evitar problemas na estimativa
# df.sort_values(by=['DataPregao'], inplace=True)

# Slicing the data_frame to define X and Y
X = df.values[:, 0:4]
Y = df.values[:,4]

# pprint.pprint(X)
# pprint.pprint(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

rgr = DecisionTreeRegressor(presort=True)
rgr.fit(X_train, y_train)

y_pred = rgr.predict(X_test)
print("DRT accuracy is ", explained_variance_score(y_test,y_pred)*100)



# rgr_knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
# rgr_knn.fit(X_train, y_train)
# y_pred_knn = rgr_knn.predict(X_test)
# print("KRR accuracy is ", explained_variance_score(y_test,y_pred_knn)*100)

# X_test_2 = [
#     [20181214, 23.08, 23.33, 22.96, 21.408139]
#     , [20181213, 23.29, 23.45, 22.96, 21.391638]
#     , [20181212, 23.74, 23.90, 23.20, 21.371856]]

# X_test_2 = Normalizer().fit_transform(X_test_2)
# y_pred_2 = rgr.predict(X_test_2)
# print(y_pred_2)