import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import pickle
import matplotlib.pyplot as plt
import numpy as np

#hyperparameter
batch_size = 100
h_size = 28
w_size = 28
c_size = 1
hidden_size = 100
total_epoch=20
learning_rate = 0.0025


#mnist data불러옴
mnist = input_data.read_data_sets("data/", one_hot=True, reshape=False)
trainimgs, trainlabels, testimgs, testlabels \
 = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

#train data 수, test data 수, dim, class수
ntrain, ntest, dim, nclasses \
 = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]


#reset graph
#tf.reset_default_graph()

#placeholder
X = tf.placeholder(tf.float32, shape=[batch_size, h_size, w_size, c_size], name="in_") # [100, 28, 28, 1]
Y = tf.placeholder(tf.float32, shape=[batch_size, 10])

#image -> vector sequence
#28*28이미지 -> 28*1의 28개의 sequence로 변형
x_split = tf.split(X, h_size, axis=1) # [100, 28, 28, 1] -> list of [100, 1, 28, 1]

#rnn 모델에 필요한 인스턴스
U = tf.Variable(tf.random_normal([w_size, hidden_size], stddev=0.01))
W = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # always square
V = tf.Variable(tf.random_normal([hidden_size, 10], stddev=0.01))

#hidden state
s = {}
s_init = tf.random_normal(shape=[batch_size, hidden_size], stddev=0.01)
s[-1] = s_init


#model specification
for t, x_split in enumerate(x_split):
    x = tf.reshape(x_split, [batch_size, w_size]) # [100, 1, 28, 1] -> [100, 28]
    s[t] = tf.nn.tanh(tf.matmul(x, U) + tf.matmul(s[t-1], W))

pred_softmax=tf.nn.softmax(tf.matmul(s[h_size - 1], V), name="out_")
cost = -tf.reduce_mean(tf.log(tf.reduce_sum(pred_softmax*Y, axis=1)))

#training
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#결과값 pred
correct_pred = tf.argmax(pred_softmax, axis=1)
correct_actual =tf.argmax(Y, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(correct_pred, correct_actual), tf.float32))

#train도중 loss, acc저장
history = dict(train_loss=[],
               train_acc=[],
               test_loss=[],
               test_acc=[])

#session과 saver
saver = tf.train.Saver()
summary_op = tf.summary.merge_all()
#sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

#validate
test_inputs = testimgs[:batch_size]
test_outputs = testlabels[:batch_size]

'''
def accuracy(network, t):
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))
'''
total_batch=int(ntrain/hidden_size)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1,total_epoch):

        for i in range(total_batch):
            batch_x = trainimgs[i * batch_size:(i + 1) * batch_size]
            batch_y = trainlabels[i * batch_size:(i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            predic_train, acc_train, loss_train = sess.run([pred_softmax, acc, cost], feed_dict={
                X: batch_x, Y: batch_y})

        predic_test, acc_test, loss_test = sess.run([pred_softmax, acc, cost], feed_dict={
            X: test_inputs, Y: test_outputs})

        print(f'epoch: {epoch} batch {i} test accuracy: {acc_test} loss: {loss_test}')

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)

    tf.train.write_graph(sess.graph_def, '.', '../rnnraw.pbtxt')

    #predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, cost],
    #                                              feed_dict={X: testimgs, Y: testlabels})
    saver.save(sess, save_path="../rnnraw.ckpt")


print()
#print(f'final results: accuracy: {acc_final} loss: {loss_final}')
#pickle.dump(predictions, open("predictions.p", "wb"))
pickle.dump(history, open("history.p", "wb"))


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

