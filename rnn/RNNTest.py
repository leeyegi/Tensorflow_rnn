import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from rnn import txtget as txt
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import seaborn as sns
import pickle



#칫솔질 데이터를 받기위한 배열선언
#xfile은 학습할 데이터 값이 저장되는 리스트
#yfile은 class가 저장되는 리스트
xfile=[]
yfile=[]

#3개의 파일을 가져와서 txt모듈에 있는 get함수를 호출해 리턴되는 값을 result에 저장
#다양한 파일의 데이터 셋을 가져오는 경우 getmany함수를 사용 - 매개변수는 리스트 형식
#02.t.txt - 태깅이 되어있는 파일
#02.a.txt - 가속도값이 저장되어있는 파일
#02.g.txt - 자이로값이 저장되어있는 파일
'''result = txt.get("02.g.txt", "02.a.txt", "02.t.txt")'''
result = txt.getmany([["dataset/old/1g2018041952.txt", "dataset/old/1a2018041952.txt", "dataset/old/1t2018041952.txt"],
                      ["dataset/old/2g201804091947.txt","dataset/old/2a201804091947.txt","dataset/old/2t201804091947.txt"],
                      ["dataset/old/3g20180409131241.txt","dataset/old/3a20180409131241.txt","dataset/old/3t20180409131241.txt"],
                      ["dataset/old/4g20180409455.txt", "dataset/old/4a20180409455.txt", "dataset/old/4t20180409455.txt"]
                      ])

#result - class와 학습할 데이터 값이 같이 들어있는 리스트
#       - > x(학습할 데이터) 와 y(클래스) 를 분리
for i in result:
    yfile.append(i[0])
    xfile.append(i[1:11])

#받은 리스트를 넘파이 배열로 저장
#dataset_x의 형상은 (데이터셋 크기,10,6)
#dataset_y의 형상은 (데이터셋 크기, )
dataset_x=np.array(xfile)
dataset_y=np.array(yfile)
print(dataset_x)
print(dataset_y)

#class를 one hot encoding으로 만들어 y의 형상을(데이터셋 크기, 16)으로 바꿈
'''print("y배열 one hot으로 바꿈")'''
dataset_y_onehot = np.zeros((dataset_y.size, dataset_y.max()+1))
dataset_y_onehot[np.arange(1060), dataset_y] = 1

print(dataset_y_onehot.shape)
print(dataset_y_onehot)



#학습할 셋과 테스트할 셋을 나눔
#전체 데이터셋 * 0.8 -> 훈련데이터
#전체 데이터셋 * 0.2 -> 테스트 데이터
train_test_per = int(len(dataset_x)*0.8)

train_data_x=dataset_x[:train_test_per]
train_data_y=dataset_y_onehot[:train_test_per]
test_data_x=dataset_x[train_test_per:]
test_data_y=dataset_y_onehot[train_test_per:]

#전체 학습 & 테스트 할 데이터의 형상
#n_hidden과 total_batch를 위해 데이터 셋의 형상이 필요
'''print(dataset_x.shape)
print(dataset_y_onehot.shape)'''
(data_size_x, data_size_y, data_size_z)=dataset_x.shape
'''print(data_size_x)'''



#########
# 옵션 설정
#########
learning_rate = 0.0001              #현재 신경망이 단순하기 떄문에 학습이 금방끝나므로 학습률을 낮게 조정(0.001)
total_epoch = 20                    #에폭 -> 훈련 데이터를 모두 소진했을때 에폭수가 1 올라감(20)
batch_size = 1000                    #한번에 몇개의 데이터를 학습할지 배치사이즈 설정(1000)

# RNN 은 순서가 있는 자료를 다루므로,
# 한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지를 설정
# 이를 위해 가로 픽셀수를 n_input 으로, 세로 픽셀수를 입력 단계인 n_step 으로 설정
n_input = 10
n_step = 6
n_hidden = data_size_x
n_class = 17

#########
# 신경망 모델 구성
# X -> (?, 10, 6)
# Y -> (?, 17)
# W -> (데이터셋 크기, 17)
# b -> (17, )
#########
X = tf.placeholder(tf.float32, [None, n_input, n_step])
Y = tf.placeholder(tf.float32, [None,n_class ])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# RNN 에 학습에 사용할 셀을 생성합니다
# 다음 함수들을 사용하면 다른 구조의 셀로 간단하게 변경할 수 있습니다
# BasicRNNCell,BasicLSTMCell,GRUCell
cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# RNN 신경망을 생성합니다
# 원래는 다음과 같은 과정을 거쳐야 하지만
# states = tf.zeros(batch_size)
# for i in range(n_step):
#     outputs, states = cell(X[[:, i]], states)
# ...
# 다음처럼 tf.nn.dynamic_rnn 함수를 사용하면
# CNN 의 tf.nn.conv2d 함수처럼 간단하게 RNN 신경망을 만들어줍니다.
# 겁나 매직!!
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 결과를 Y의 다음 형식과 바꿔야 하기 때문에
# Y : [batch_size, n_class]
# outputs 의 형태를 이에 맞춰 변경해야합니다.
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
#        -> [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

#비용함수와 최적화 함수를 설정
#비용함수 -> softmax_cross_entropy_with_logits - > softmax함수 수행후 비용함수로 교차엔트로피오차함수를 함께 사용
#최적화 -> adam = adagrad(개별 매개변수에 적응적 학습률을 조정) + 모멘텀(학습률에 중력값을 주어 local minimum에 빠지지 않게 해줌)
cost_pre=tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y)
cost = tf.reduce_mean(cost_pre)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#########
# 신경망 모델 학습
######
sess = tf.Session()
init = tf.global_variables_initializer()

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#비용함수와 정확도 그래프를 그리기위해 학습과정에서 비용함수값과 정확도 값을 리스트로 저장
#train_cost_list - 훈련데이터가 학습되면서 에폭당 비용함수 값을 저장하는 리스트
#train_acc_list - 훈련제이터가 학습되면서 에폭당 비용함수 값을 저장하는 리스트
train_cost_list=[]
train_acc_list=[]

def run_train(session, train_x, train_y):
    print ("\nStart training")
    total_cost = 0
    sess.run(init)

    for epoch in range(total_epoch):
        total_batch = int(data_size_x / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = train_data_x, train_data_y
            # 훈련데이터가 학습하는 코드

            train_opti, train_acc, train_cost = sess.run([optimizer, accuracy, cost], feed_dict={X: batch_xs, Y: batch_ys})
            total_cost += train_cost


        print('Epoch:', '%04d' % (epoch + 1), '정확도:', train_acc, 'Avg. cost =', '{:.3f}'.format(train_cost))
        # 그리프를 그리기 위해 에폭이 끝나면 비용함수와 정확도 값 리스트에 추가
        train_acc = float('{:.2f}'.format(train_acc))
        train_acc_list.append(train_acc)
        train_cost = float('{:.2f}'.format(total_cost / total_batch))
        print(train_cost)
        train_cost_list.append(train_cost)



def cross_validate(session, split_size=5):
  results = []
  kf = KFold(n_splits=split_size)
  for train_idx, val_idx in kf.split(train_data_x, train_data_y):
    train_x = train_data_x[train_idx]
    train_y = train_data_y[train_idx]
    val_x = train_data_x[val_idx]
    val_y = train_data_y[val_idx]
    run_train(session, train_x, train_y)
    results.append(session.run(accuracy, feed_dict={X: val_x, Y: val_y}))
  return results



#########
# 결과 확인
######
result = cross_validate(sess)
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
acc_final, loss_final = sess.run([accuracy, cost], feed_dict={X: test_data_x, Y: test_data_y})

print('정확도:', acc_final)
print ("Cross-validation result: %s" % result)
print ("Test accuracy:"+str(acc_final))


'''
print(train_cost_list)
print(train_acc_list)
'''

"""
#비용함수 그래프 그리기
x = np.arange(total_epoch)
plt.plot(x, train_cost_list, label='cost')
plt.xlabel("epochs")
plt.show()

#정확도 그래프 그리기
x = np.arange(total_epoch)
plt.plot(x, train_acc_list, label='acc')
plt.xlabel("epochs")
plt.show()
"""


#테스트 데이터를 학습 후 예측값 prediction 저장
prediction=sess.run(is_correct, feed_dict={X:test_data_x, Y:test_data_y})
'''print(test_data_y.shape)
print(prediction.shape)'''


#matrix confusion을 위해 onehot -> label로 바꿈
onehot_to_label=sess.run(tf.argmax(test_data_y, axis=1))
'''print(onehot_to_label)'''


#cnt - 테스트 데이터의 클래스별 갯수 확인 리스트
#cnt_T - 테스트 데이터의 예측값을 저장 - 올바르게 예측했을때
#cnt_F - 테스트 데이터의 예측값을 저장 - 예측을 실패했을때
cnt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cnt_T=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
cnt_F=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(len(onehot_to_label)):
    tmp = onehot_to_label[i]
    cnt[tmp]+=1
    if prediction[i]==True:
        cnt_T[tmp]+=1
    else :
        cnt_F[tmp] += 1

print(cnt)
print("true")
print(cnt_T)
print("false")
print(cnt_F)


#metrics.confusion_matrix을 하기위해 테스트 데이터 예측값 필요
prediction = tf.argmax(model,1)
best=sess.run([prediction], feed_dict={X:test_data_x})
'''print(best[0])'''

#metrics.confusion_matrix그리기 - normalization 수행
LABELS=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

confusion_matrix = metrics.confusion_matrix(onehot_to_label, best[0])
'''print(confusion_matrix)'''
cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(16, 16))
sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f")
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

"""
#metrics.confusion_matrix그리기 - normalization 미수행
LABELS=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

confusion_matrix = metrics.confusion_matrix(onehot_to_label, best[0])
print(confusion_matrix)

plt.figure(figsize=(16, 16))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
"""