import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from rnn_v2 import data_preprocessing
import pandas as pd

#데이터 처리를 위해 필요한 인스턴스
N_TIME_STEPS = 50
N_FEATURES = 6
step = 25
RANDOM_SEED = 42
segments = []
labels = []
file_name=['dataset_v2/HJH_2018_10_03_3_log.txt']

get_df_data = data_preprocessing.get_data(file_name)

segments, labels = data_preprocessing.data_shape(get_df_data)

reshaped_segments = np.array(segments).reshape(-1, N_TIME_STEPS, N_FEATURES)

labels = np.array(pd.get_dummies(labels),dtype=np.int8)

'''
print(reshaped_segments)
print(reshaped_segments.shape)
print(labels)
print(labels.shape)
'''

X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)




