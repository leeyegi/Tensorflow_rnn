import tensorflow as tf
import numpy as np
from rnn_v2 import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

file_name=['../dataset_v2/HJH_2018_10_03_3_log.txt', '../dataset_v2/HJH_2018_10_04_3_log.txt',
           '../dataset_v2/HJH_2018_10_05_2_log.txt','../dataset_v2/HJH_2018_10_06_3_log.txt',
           '../dataset_v2/HJH_2018_10_12_3_log.txt','../dataset_v2/HJH_2018_10_13_1_log.txt',
           '../dataset_v2/HJH_2018_10_15_3_log.txt','../dataset_v2/HJH_2018_10_16_1_log.txt',
           '../dataset_v2/HJH_2018_10_17_3_log.txt','../dataset_v2/HJH_2018_10_22_3_log.txt',
           '../dataset_v2/HJH_2018_10_24_3_log.txt']


get_df_data = data_preprocessing.get_data(file_name)
reshaped_segments, reshaped_labels=data_preprocessing.data_shape(get_df_data)
reshaped_segments_2=reshaped_segments.reshape(reshaped_segments.shape[0], 300)
print(reshaped_segments_2.shape)
print(reshaped_segments_2)
x_train, x_test, y_train, y_test = train_test_split(reshaped_segments_2, reshaped_labels, test_size=0.2)
print(x_train.shape)
print(y_train.shape)


def nn_layer(name, input_data, output_size):
    W = tf.get_variable(name=name + "_W",
                        shape=[input_data.get_shape().as_list()[1], output_size],
                        initializer=tf.contrib.layers.xavier_initializer())
    B = tf.get_variable(name=name + "_B",
                        shape=[output_size],
                        initializer=tf.contrib.layers.xavier_initializer())
    return tf.matmul(input_data, W) + B


dropout_rate = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 300], name="x")
Y = tf.placeholder(tf.float32, [None, 16], name="y")

with tf.name_scope("Layer2"):
    _L2 = tf.nn.relu(nn_layer("L2", X, 256))
    L2 = tf.nn.dropout(_L2, dropout_rate)

with tf.name_scope("Layer3"):
    _L3 = tf.nn.relu(nn_layer("L3", L2, 256))
    L3 = tf.nn.dropout(_L3, dropout_rate)

with tf.name_scope("Layer4"):
    _L4 = tf.nn.relu(nn_layer("L4", L3, 128))
    L4 = tf.nn.dropout(_L4, dropout_rate)

with tf.name_scope("Layer5"):
    _L5 = tf.nn.relu(nn_layer("L5", L4, 128))
    L5 = tf.nn.dropout(_L5, dropout_rate)

with tf.name_scope("Layer6"):
    _H = nn_layer("L6", L5, 16)
    H = tf.nn.dropout(_H, dropout_rate)

with tf.name_scope("Train") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=H, labels=Y))
    cost_summ = tf.summary.scalar("Cost", cost)
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    correct = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summ = tf.summary.scalar("Accuracy", accuracy, ["Epoch"])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    merged = tf.summary.merge_all()
    epoch_merged = tf.summary.merge_all("Epoch")
    writer = tf.summary.FileWriter("./logs", sess.graph)

    batch_size = 512
    train_count = len(x_train)

    for epoch in range(1000):
        for start, end in zip(range(0, train_count, batch_size),
                          range(batch_size, train_count + 1, batch_size)):
            sess.run(train, feed_dict={X: x_train[start:end],
                                           Y: y_train[start:end]})
            summ, _ = sess.run([merged, train], feed_dict={X: x_train,
                                                           Y: y_train,
                                                           dropout_rate: 0.75})
            #writer.add_summary(summ, epoch * batch_size + step)

        summ, _accuracy = sess.run([epoch_merged, accuracy],
                                   feed_dict={X: x_test,
                                              Y: y_test,
                                              dropout_rate: 1.0})
        writer.add_summary(summ, (epoch + 1) * batch_size)

        print("Epoch:", (epoch + 1))
        print("Test Accuracy:", _accuracy)



