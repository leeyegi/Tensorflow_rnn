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
import pickle


#데이터 처리를 위해 필요한 인스턴스
N_TIME_STEPS = 50
N_FEATURES = 6
step = 25
RANDOM_SEED = 42
segments = []
labels = []
file_name=['dataset_v2/HJH_2018_10_03_3_log.txt', 'dataset_v2/HJH_2018_10_04_3_log.txt',
           'dataset_v2/HJH_2018_10_05_2_log.txt','dataset_v2/HJH_2018_10_06_3_log.txt',
           'dataset_v2/HJH_2018_10_12_3_log.txt','dataset_v2/HJH_2018_10_13_1_log.txt',
           'dataset_v2/HJH_2018_10_15_3_log.txt','dataset_v2/HJH_2018_10_16_1_log.txt',
           'dataset_v2/HJH_2018_10_17_3_log.txt','dataset_v2/HJH_2018_10_22_3_log.txt',
           'dataset_v2/HJH_2018_10_24_3_log.txt']

#데이터 파일을 한번에 받은 후 데이터 pandas로 dataframe형태로 generation
#class_num도 붙여줌
get_df_data = data_preprocessing.get_data(file_name)

