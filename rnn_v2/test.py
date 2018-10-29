import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from rnn_v2 import data_preprocessing
import pickle

n_classes = 16
num_units = 64
time_steps = 50
n_input = 6
learning_rate=0.01

file_name=[#'dataset_v2/HJH_2018_10_03_3_log.txt', 'dataset_v2/HJH_2018_10_04_3_log.txt',
           #'dataset_v2/HJH_2018_10_05_2_log.txt','dataset_v2/HJH_2018_10_06_3_log.txt',
           #'dataset_v2/HJH_2018_10_12_3_log.txt','dataset_v2/HJH_2018_10_13_1_log.txt',
           #'dataset_v2/HJH_2018_10_15_3_log.txt','dataset_v2/HJH_2018_10_16_1_log.txt',
           #'dataset_v2/HJH_2018_10_17_3_log.txt','dataset_v2/HJH_2018_10_22_3_log.txt',
           'dataset_v2/HJH_2018_10_24_3_log.txt']


get_df_data = data_preprocessing.get_data(file_name)

reshaped_segments, reshaped_labels=data_preprocessing.data_shape(get_df_data)


#데이터를 랜덤으로 섞어 훈련데이터와 테스트 데이터 setting함
x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, reshaped_labels, test_size=0.2, random_state=42)


#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]),name="weights")
out_bias=tf.Variable(tf.random_normal([n_classes]),name="bias")

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1,name="input_tensor")

#defining the network
#lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
#lstm_layer=rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units) for _ in range(3)])
#lstm_layer=rnn.LSTMBlockCell(num_units,forget_bias=1)
lstm_layer=tf.nn.rnn_cell.BasicLSTMCell(num_units)
#lstm_layer=tf.nn.rnn_cell.GRUCell(num_units)
#lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)
outputs,_= tf.contrib.rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.add(tf.matmul(outputs[-1],out_weights), out_bias, name="output")

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
output_dir="./test"
batch_size=100
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # added
    tf.train.write_graph(sess.graph_def, '.', output_dir + '/model.pbtxt')
    total_batch = int(x_train.shape[0] / batch_size)
    iter=1
    while iter<100:
        for i in range(total_batch):
            #한번에 batchc_size만큼 학습
            batch_x = x_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]

            batch_x=batch_x.reshape((batch_size,time_steps,n_input))

            sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")



        filename = saver.save(sess, output_dir + '/model.ckpt')

        iter=iter+1
