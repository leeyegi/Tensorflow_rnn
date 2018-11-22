import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

def nn_layer(name, input_data, output_size):
    W = tf.get_variable(name=name + "_W",
                        shape=[input_data.get_shape().as_list()[1], output_size],
                        initializer=tf.contrib.layers.xavier_initializer())
    B = tf.get_variable(name=name + "_B",
                        shape=[output_size],
                        initializer=tf.contrib.layers.xavier_initializer())
    return tf.matmul(input_data, W) + B

ensemble_nn_num = 5

dropout_rate = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 28*28], name="x")
Y = tf.placeholder(tf.float32, [None, 10], name="y")

with tf.name_scope("Layer2"):
    _L2 = tf.nn.relu(nn_layer("L2", X, 14*14))
    L2 = tf.nn.dropout(_L2, dropout_rate)
    
with tf.name_scope("Layer3"):
    _L3 = tf.nn.relu(nn_layer("L3", L2, 14*14))
    L3 = tf.nn.dropout(_L3, dropout_rate)
    
with tf.name_scope("Layer4"):
    _L4 = tf.nn.relu(nn_layer("L4", L3, 14*14))
    L4 = tf.nn.dropout(_L4, dropout_rate)
    
with tf.name_scope("Layer5"):
    _L5 = tf.nn.relu(nn_layer("L5", L4, 14*14))
    L5 = tf.nn.dropout(_L5, dropout_rate)
    
with tf.name_scope("Layer6"):
    _H = nn_layer("L6", L5, 10)
    H = tf.nn.dropout(_H, dropout_rate)

with tf.name_scope("Train") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(H, Y))
    train = tf.train.AdamOptimizer(0.001).minimize(cost)
    
    correct = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver(max_to_keep=5)
save_dir = './checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sess = tf.Session()
    
for i in xrange(ensemble_nn_num):
    sess.run(tf.global_variables_initializer())
    
    batch_size = mnist.train.num_examples/100
    
    for epoch in xrange(15):
        for step in xrange(batch_size):
            image, label = mnist.train.next_batch(100)
            sess.run(train, feed_dict={X: image,
                                       Y: label,
                                       dropout_rate: 0.75})
        
        _accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images,
                                                  Y: mnist.test.labels,
                                                  dropout_rate: 1.0})
            
        print "NN:", i, "Epoch:", (epoch + 1)
        print "Test Accuracy:", _accuracy
    saver.save(sess, save_path=save_dir + "nn" + str(i))

pred_labels = []
for i in xrange(ensemble_nn_num):
    saver.restore(sess, save_path=save_dir + "nn" + str(i))
    pred = sess.run(H, feed_dict={X: mnist.test.images, dropout_rate: 1.0})
    pred_labels.append(pred)
    
# Get average of the predictions of NNs
ensemble_pred_labels = np.mean(pred_labels, axis=0)

ensemble_correct = np.equal(np.argmax(ensemble_pred_labels, 1), np.argmax(mnist.test.labels, 1))
ensemble_accuracy = np.mean(ensemble_correct.astype(np.float32))
print "Ensemble Accuracy:", ensemble_accuracy

