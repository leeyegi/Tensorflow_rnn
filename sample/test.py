import tensorflow as tf
import numpy as np


data =  [[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
         [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]]

label = [[0], [1], [0], [0], [0]]

steps = 1
classes = 5
batch_size = 2
loss = 0.0


x = tf.placeholder(tf.float32, shape=[batch_size, 20])
y = tf.placeholder(tf.float32, shape=[classes,1])
x_input = tf.split(0, steps, x)

W = tf.Variable(tf.random_normal([classes, 2]))
b = tf.Variable(tf.random_normal([classes,1]))

lstm = tf.nn.rnn_cell.BasicLSTMCell(steps)
output, state = tf.nn.rnn(lstm, x_input, dtype=tf.float32 )

pred = tf.matmul(W, output[-1] ) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_pred = tf.equal( tf.argmax(pred,1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    feed_dict = {x: data, y:label}
    sess.run(init)

    for i in range(10000):
        sess.run(optimizer, feed_dict=feed_dict)
        if i % 100 == 0 :
            print(sess.run( pred, feed_dict))
            print(sess.run(accuracy, feed_dict))

    print(sess.run( pred, feed_dict={x: data}))