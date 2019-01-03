#센서 데이터 전처리
#하드웨어를 통해 받은 데이터를 android상에서 받아와 txt파일을 생성 후 전처리를 수행하는 파일
#데이터의 shape (label, (ax,ay,az,gx,gy,gz)*50개 )


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random


#불러올 파일 이름
#태그파일이 아니라 로그파일만 불러와도 됨

file_name=['../dataset_v3/2_HJH_2018_11_19_1_log.txt']


#file_name=['../dataset_v2/HJH_2018_10_03_3_log.txt']


#data frame의 index들
columns_data = ['num', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'class_num']
#columns_tag=['num', 'start', 'end']


#데이터를 받아와서 index에 맞추고 lable도 매기는 메소드
def get_data(file_name):

    # 데이터 불러오기
    #tmp = pd.read_csv(file_name[i] + "log.txt", sep=" ", header=None, names=columns_data)
    df_from_each_file = (pd.read_csv(f,sep=" ", header=None, names=columns_data) for f in file_name)
    df_data = pd.concat(df_from_each_file, ignore_index=True)

    #df_tag = pd.read_csv(file_name[i]+"tag.txt" ,sep=" ", header=None, names=columns_tag)
    print(df_data)
    #print(df_tag)

    class_num=0         #정답 레이블-> 순서대로 달리므로 현재는 tag필요없음
    len,_=df_data.shape #dataframe의 shpae
    print("len"+str(len))

    #dataframe의 행수만큼 반복
    for j in range(0, len):
        if (df_data.loc[j,'num']==0):           # num의 값이 0이면 label +1
            class_num+=1
            if(class_num==17):
                class_num=1
                print("convert"+str(class_num))
            print("check"+str(class_num))
        df_data.loc[j,'class_num'] = class_num      #dataframe의 label에 정답레이블 달아줌
        df_data.loc[j] = df_data.loc[j].apply(pd.to_numeric, errors='coerce', )
    df_data=df_data.fillna(0)


    print(df_data)
    #print(df_tag)

    return df_data



#50% overlap하면서 label별로 50개씩 데이터 잘라 plot을 생성
def plot_activity(label, df):
    len,_ = df.shape    #데이터의 shape
    data = []
    print("len"+str(len))

    #overlap하는 부분
    for i in range(0,len-50,25):
        if df.loc[i,'class_num']==label and df.loc[i+50,'class_num']==label:    #50개의 6새센서 데이터가 한 세트
            data = df.iloc[i:i+50,1:8]                                      #ax,ay,az, gx,gy,gz값들어감

    #print(data)
    #print(np.array(data).shape)

    axis = df.plot(subplots=True, figsize=(16, 12),title=label)
    #print(axis.__class__)
    #print(axis)

    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
        #print(ax.__class__)
        #print(ax)

N_TIME_STEPS = 60
N_FEATURES = 6
segments = []
labels = []
#데이터를 50%씩 overlap하여 50개의 데이터저장(array)
def data_shape(df):
    a = 1

    len,_ = df.shape    #데이터의 shape
    label_index=1
    print("len"+str(len))

    #데이터를 받아와 50개씩 잘라 저장 overlap 50%
    for i in range(0, len - 60, 30):
        if df.loc[i, 'class_num'] == label_index and df.loc[i + 60, 'class_num'] == label_index:  # 50개의 6새센서 데이터가 한 세트
            print("save")
            ax = df['ax'].values[i: i + N_TIME_STEPS]
            ay = df['ay'].values[i: i + N_TIME_STEPS]
            az = df['az'].values[i: i + N_TIME_STEPS]
            gx = df['gx'].values[i: i + N_TIME_STEPS]
            gy = df['gy'].values[i: i + N_TIME_STEPS]
            gz = df['gz'].values[i: i + N_TIME_STEPS]

            label = stats.mode(df['class_num'][i: i + N_TIME_STEPS])[0][0]
            segments.append([ax, ay, az, gx, gy, gz])
            labels.append(label)

        # 50개씩 자를때 첫번째와 50번째에 class_num값이 다르면 class_num 바꿈
        if df.loc[i, 'class_num'] == label_index and \
                df.loc[i + 60, 'class_num'] != label_index and \
                df.loc[i + 60, 'class_num'] == df.loc[i + 30, 'class_num']:  # 50개의 6새센서 데이터가 한 세트
            print("change"+str(a))
            a+=1

            label_index = df.loc[i + 30, 'class_num']
            # print(label_index)

            #i=i+25
            #print(label_index)
    reshaped_segments = np.array(segments, dtype=np.float32).reshape(-1, N_FEATURES, N_TIME_STEPS)
    reshaped_segments=np.transpose(reshaped_segments, (0,2,1))
    reshaped_labels = np.array(pd.get_dummies(labels),dtype=np.float32)

    #print(segments)
    print(np.array(segments).shape)
    print(reshaped_segments.shape)
    #print(labels)
    #return segments, labels
    return reshaped_segments, reshaped_labels


if __name__ == "__main__":
    get_df_data=get_data(file_name)
    #print(get_df_data)'''
    '''
    for i in range(1,17):
        get_split_data=plot_activity(i, get_df_data)
    '''

    reshaped_segments,reshaped_labels=data_shape(get_df_data)

    #print(reshaped_segments)
    #print(reshaped_segments.shape)
    #print(reshaped_labels)
    #print(reshaped_labels.shape)

    #get_df_data.to_csv("../get_df_data.txt")

    '''
    reshaped_segments_txt=pd.DataFrame(reshaped_segments)
    labels_txt=pd.DataFrame(labels)
    reshaped_segments_txt.to_csv("../reshaped_segments_txt.txt")
    labels_txt.to_csv("../labels_txt.txt")
    
    pickle.dump(reshaped_segments, open("../reshaped_segments.txt", "wb",encoding='utf-8'))
    pickle.dump(labels, open("../labels.txt", "wb", encoding='utf-8'))
    np.savetxt("../labels.txt", labels, delimiter=' ')
    '''