import tensorflow as tf
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
    
    batch_size = mnist.train.num_examples/100
    
    for epoch in xrange(15):
        for step in xrange(batch_size):
            image, label = mnist.train.next_batch(100)
            summ, _ = sess.run([merged, train], feed_dict={X: image,
                                                           Y: label,
                                                           dropout_rate: 0.75})
            writer.add_summary(summ, epoch * batch_size + step)
        
        summ, _accuracy = sess.run([epoch_merged, accuracy],
                                          feed_dict={X: mnist.test.images,
                                                     Y: mnist.test.labels,
                                                     dropout_rate: 1.0}) 
        writer.add_summary(summ, (epoch + 1) * batch_size)
        
        print "Epoch:", (epoch + 1)
        print "Test Accuracy:", _accuracy

