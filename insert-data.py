from pymongo import MongoClient
import numpy as np
import pandas as pd
import os
import json
import sys

#Funcao para carregar e preparar os arquivos
def load_and_prepare_files():
    print("********************CARREGANDO ARQUIVOS********************")
    colspecs = [(2,10),(12,24),(56,69),(69,82),(82,95),(108,121)]
    names = ["DataPregao", "Ticker", "PrecoAbertura", "PrecoMax", "PrecoMin", "PrecoUltimoNegocio"]
    frames = []

    for root, dirs, files in os.walk("./data"):
        for filename in files:
            print("Carregando o arquivo: ", filename)
            temp_dataset = pd.read_fwf('data/' + filename, header=None, colspecs=colspecs, names=names)
            temp_dataset.drop([0], inplace=True)
            temp_dataset = temp_dataset[:-1]
            frames.append(temp_dataset)

    dataset = pd.concat(frames)
    dataset['PrecoAbertura'] = dataset['PrecoAbertura']/100
    dataset['PrecoMax'] = dataset['PrecoMax']/100
    dataset['PrecoMin'] = dataset['PrecoMin']/100
    dataset['PrecoUltimoNegocio'] = dataset['PrecoUltimoNegocio']/100
    print("Todos os arquivos carregados e tratados\n")
    return dataset

df = load_and_prepare_files()
data_json = json.loads(df.to_json(orient='records'))

client = MongoClient('localhost', 27017)

db = client['bovespa']

print("Inserindo dados...")
db.cotacao_diaria.insert(data_json)
print("Dados carregados com sucesso!")