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
file_name=[#'../dataset_v2/HJH_2018_10_03_3_log.txt', '../dataset_v2/HJH_2018_10_04_3_log.txt',
           #'../dataset_v2/HJH_2018_10_05_2_log.txt','../dataset_v2/HJH_2018_10_06_3_log.txt',
           #'../dataset_v2/HJH_2018_10_12_3_log.txt','../dataset_v2/HJH_2018_10_13_1_log.txt',
           #'../dataset_v2/HJH_2018_10_15_3_log.txt','../dataset_v2/HJH_2018_10_16_1_log.txt',
           #'../dataset_v2/HJH_2018_10_17_3_log.txt','../dataset_v2/HJH_2018_10_22_3_log.txt',
           '../dataset_v2/HJH_2018_10_24_3_log.txt']

#데이터 파일을 한번에 받은 후 데이터 pandas로 dataframe형태로 generation
#class_num도 붙여줌
get_df_data = data_preprocessing.get_data(file_name)

#dataframe을 받아서 훈련에 필요한 데이터 shape을 맞춰줌
segments, labels = data_preprocessing.data_shape(get_df_data)

reshaped_segments = np.array(segments).reshape(-1, N_TIME_STEPS, N_FEATURES)

labels = np.array(pd.get_dummies(labels),dtype=np.int8)

'''
print(reshaped_segments)
print(reshaped_segments.shape)
print(labels)
print(labels.shape)
'''
#데이터를 랜덤으로 섞어 훈련데이터와 테스트 데이터 setting함
x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)

print("data shape")
print(x_train.shape)
print(x_test.shape)

#lstm모델을 만들기 위한 인스턴스
N_CLASSES = 16
N_HIDDEN_UNITS = 64

#lstm모델 만들기
def create_LSTM_model(inputs):
    '''
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }'''

    W = tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    b = tf.Variable(tf.random_normal([N_CLASSES]))

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + b['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # Stack 3 LSTM layers
    #lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(3)]
    #lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    cell1 = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS)
    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=0.8)
    cell3 = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS)
    cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, output_keep_prob=0.7)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3])

    #outputs, states = tf.contrib.rnn.static_rnn(multi_cell, hidden, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + b['output']


tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="inputs")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])


pred_Y = create_LSTM_model(X)

pred_softmax = tf.nn.softmax(pred_Y, name="y_")

#L2_LOSS = 0.01

#l2 = L2_LOSS *     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
cost_pre=tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_softmax, labels=Y)
loss = tf.reduce_mean(cost_pre)


LEARNING_RATE = 0.01

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


N_EPOCHS = 1000
BATCH_SIZE = 512

saver = tf.train.Saver()

history = dict(train_loss=[],
               train_acc=[],
               test_loss=[],
               test_acc=[])

sess = tf.Session()
#sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("rnn_v2_16/",graph_def=sess.graph_def)

train_count = len(x_train)

#학습 시작
for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: x_train[start:end],
                                       Y: y_train[start:end]})

    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
        X: x_train, Y: y_train})

    _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
        X: x_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)

    if i != 1 and i % 10 != 0:
        continue

    print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

#학습 종료 후 파라미터 저장 및 체크포인트 저장
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: x_test, Y: y_test})

print()
print(f'final results: accuracy: {acc_final} loss: {loss_final}')


pickle.dump(predictions, open("predictions.p", "wb"))
pickle.dump(history, open("history.p", "wb"))
tf.train.write_graph(sess.graph_def, '.', '../rnn_v2_model_check.pbtxt')
saver.save(sess, save_path = "../rnn_v2_model_check.ckpt")
sess.close()


#그래프 그리기
plt.figure(figsize=(12, 8))

plt.plot(np.array(history['train_loss']), "r--", label="Train loss")
plt.plot(np.array(history['train_acc']), "g--", label="Train accuracy")

plt.plot(np.array(history['test_loss']), "r-", label="Test loss")
plt.plot(np.array(history['test_acc']), "g-", label="Test accuracy")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()
print(predictions.__class__)
print(predictions.shape)

LABELS = ['1', '2', '3', '4', '5', '6','7', '8', '9', '10', '11', '12','13', '14', '15', '16']
max_test = np.argmax(y_test, axis=1)
max_predictions = np.argmax(predictions, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
