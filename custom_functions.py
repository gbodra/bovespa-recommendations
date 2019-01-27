import pandas as pd
import numpy as np

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def otimiza_memoria(data_frame):
    print("********************OTIMIZANDO MEMORIA********************")
    print("Total de memoria: ", mem_usage(data_frame))
    data_frame['DataPregao'] = data_frame['DataPregao'].astype('category')
    data_frame_float = data_frame.select_dtypes(include=['float'])
    converted_float = data_frame_float.apply(pd.to_numeric,downcast='float')
    data_frame[converted_float.columns] = converted_float

    print("Total de memoria: ", mem_usage(data_frame), "\n")
    return data_frame