import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
import datetime
import sys
import matplotlib.pyplot as plt

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

#
# dataset = pd.read_csv('data/WIN_N_M5.csv', header=None, names=['datetime', 'open', 'high', 'low', 'close', 'contracts', 'volume'])
dataset = pd.read_csv('data/WIN_N_Daily.csv', header=None, names=['datetime', 'open', 'high', 'low', 'close', 'contracts', 'volume'])

dataset.drop(columns=['contracts', 'volume'] ,inplace=True)

dataset['datetime'] = pd.to_datetime(dataset['datetime'])

dataset['ts'] = dataset.datetime.values.astype(np.int64) // 10 ** 9

dataset.drop(columns=['datetime'], inplace=True)
dataset = dataset[['ts','open', 'high', 'low', 'close']]
# dataset = dataset[['ts','open', 'close']]

dataset_float = dataset.select_dtypes(include=['float'])
converted_float = dataset_float.apply(pd.to_numeric,downcast='float')

dataset[converted_float.columns] = converted_float

# dataset['direction'] = dataset.close > dataset.close.shift()

print(dataset.head())
# sys.exit()
# Slicing the data_frame to define X and Y
X = dataset.values[:, 0:4]
Y = dataset.values[:,4]
# X = dataset.values[:, 0:2]
# Y = dataset.values[:,2]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

rgr = DecisionTreeRegressor()

analysis = []
for x in range(10):
    rgr.fit(X_train, y_train)
    y_pred = rgr.predict(X_test)
    X_test_3 = [[1516060800, 94800.0, 94800.0, 94050.0]]
    y_pred_3 = rgr.predict(X_test_3)
    analysis.append(y_pred_3)

print('Average: ', (sum(analysis)/len(analysis)))
analysis.sort()
print('List sorted: ', analysis)
print("DRT accuracy is ", explained_variance_score(y_test,y_pred)*100)

# X_test_3_ts = pd.to_datetime(pd.Series(['2018.01.16 18:05']))

# X_test_3_ts = X_test_3_ts.values.astype(np.int64) // 10 ** 9

#Daily
X_test_3_ts = pd.to_datetime(pd.Series(['2018.01.18']))

X_test_3_ts = X_test_3_ts.values.astype(np.int64) // 10 ** 9

# print(X_test_3_ts)
# sys.exit()



# Plot the results
plt.figure()
plt.scatter(X[:,0], Y, edgecolor="black", c="darkorange", label="data")
plt.scatter(X_test[:,0], y_pred, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
# plt.show()