from __future__ import absolute_import, division, print_function
import sys
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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

#frames = [dataset_2010, dataset_2011, dataset_2012, dataset_2013, dataset_2014, dataset_2015, dataset_2016, dataset_2017, dataset_2018]
frames = [dataset_2018]
data_frame = pd.concat(frames)

#Transformar a data em date time
data_frame['DataPregao'] = pd.to_datetime(data_frame['DataPregao'])

# Here we normalize date and ticker
data_frame['DataPregao'] = data_frame['DataPregao'].astype('category')
data_frame['DataPregao'] = data_frame['DataPregao'].cat.codes
data_frame['Ticker'] = data_frame['Ticker'].astype('category')
data_frame['Ticker'] = data_frame['Ticker'].cat.codes

#Split into test and train
train_dataset = data_frame.sample(frac=0.8,random_state=0)
test_dataset = data_frame.drop(train_dataset.index)

#print(sns.pairplot(train_dataset[["DataPregao", "Ticker", "PrecoAbertura", "PrecoUltimoNegocio"]], diag_kind="kde"))

#Train stats
train_stats = train_dataset.describe()
train_stats.pop("PrecoUltimoNegocio")
train_stats = train_stats.transpose()

#Split label (target value) from features
train_labels = train_dataset.pop('PrecoUltimoNegocio')
test_labels = test_dataset.pop('PrecoUltimoNegocio')

#Normalize data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#Build model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

#Inspect model
#print("model.summary(): \n", model.summary())

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
#print("example_result: \n", example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print("hist.tail(): \n", hist.tail())

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: R$ {:5.2f}".format(mae))