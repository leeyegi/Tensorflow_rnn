import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from rnn_v2 import data_preprocessing
import pickle


#데이터 처리를 위해 필요한 인스턴스
N_TIME_STEPS = 50
N_FEATURES = 6
step = 25
RANDOM_SEED = 42
segments = []
labels = []
file_name=[#'dataset_v2/HJH_2018_10_03_3_log.txt', 'dataset_v2/HJH_2018_10_04_3_log.txt',
           #'dataset_v2/HJH_2018_10_05_2_log.txt','dataset_v2/HJH_2018_10_06_3_log.txt',
           #'dataset_v2/HJH_2018_10_12_3_log.txt','dataset_v2/HJH_2018_10_13_1_log.txt',
           #'dataset_v2/HJH_2018_10_15_3_log.txt','dataset_v2/HJH_2018_10_16_1_log.txt',
           #'dataset_v2/HJH_2018_10_17_3_log.txt','dataset_v2/HJH_2018_10_22_3_log.txt',
           '../rnn_v2/dataset_v2/HJH_2018_10_24_3_log.txt']


get_df_data = data_preprocessing.get_data(file_name)

reshaped_segments, reshaped_labels=data_preprocessing.data_shape(get_df_data)


print(reshaped_segments)
print(reshaped_segments.shape)
print(reshaped_labels)
print(reshaped_labels.shape)

#데이터를 랜덤으로 섞어 훈련데이터와 테스트 데이터 setting함
x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, reshaped_labels, test_size=0.2, random_state=RANDOM_SEED)

print("data shape")
print(x_train.shape)
print(x_test.shape)


#lstm모델을 만들기 위한 인스턴스
N_CLASSES = 16
N_HIDDEN_UNITS = 64

#lstm모델 만들기

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # Stack 2 LSTM layers
    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']


X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name='input')
Y = tf.placeholder(tf.float32, [None, N_CLASSES])


pred_Y = create_LSTM_model(X)

pred_softmax = tf.nn.softmax(pred_Y, name='pred')


L2_LOSS = 0.0015

l2 = L2_LOSS *     sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred_Y, labels = Y), name='pred2') + l2



LEARNING_RATE = 0.005

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))



N_EPOCHS = 10
BATCH_SIZE = 200


history = dict(train_loss=[],
               train_acc=[],
               test_loss=[],
               test_acc=[])

train_count = len(x_train)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.train.write_graph(sess.graph_def, '.', 'rnn.pbtxt')

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
        saver.save(sess, './rnn.ckpt')

    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: x_test, Y: y_test})


import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib



MODEL_NAME = 'rnn'

# Freeze the graph

input_graph_path = MODEL_NAME+'.pbtxt'
checkpoint_path = './'+MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "pred"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input"], # an array of the input node(s)
        ["pred"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())

# tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)
