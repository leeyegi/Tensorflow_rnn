import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import pickle
import matplotlib.pyplot as plt
import numpy as np
from rnn_v2 import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import random


file_name=['../dataset_v2/HJH_2018_10_03_3_log.txt', '../dataset_v2/HJH_2018_10_04_3_log.txt',
           '../dataset_v2/HJH_2018_10_05_2_log.txt','../dataset_v2/HJH_2018_10_06_3_log.txt',
           '../dataset_v2/HJH_2018_10_12_3_log.txt','../dataset_v2/HJH_2018_10_13_1_log.txt',
           '../dataset_v2/HJH_2018_10_15_3_log.txt','../dataset_v2/HJH_2018_10_16_1_log.txt',
           '../dataset_v2/HJH_2018_10_17_3_log.txt','../dataset_v2/HJH_2018_10_22_3_log.txt',
           '../dataset_v2/HJH_2018_10_24_3_log.txt']


get_df_data = data_preprocessing.get_data(file_name)

reshaped_segments, reshaped_labels=data_preprocessing.data_shape(get_df_data)

c = list(zip(reshaped_segments, reshaped_labels))

random.shuffle(c)

reshaped_segments, reshaped_labels = zip(*c)

reshaped_segments=np.array(reshaped_segments)
reshaped_labels=np.array(reshaped_labels)

x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, reshaped_labels, test_size=0.2, random_state=42)



#hyperparameter
batch_size = 1300
h_size = 50
w_size = 6
c_size = 1
hidden_size = 512
total_epoch=1000
learning_rate = 0.0001

#train data 수, test data 수, dim, class수
ntrain, ntest, dim, nclasses \
 = x_train.shape[0], x_test.shape[0], x_train.shape[1], y_train.shape[1]

print(x_train.shape)
print(x_test.shape)
print(dim)
print(nclasses)

#reset graph
tf.reset_default_graph()

#placeholder
X = tf.placeholder(tf.float32, shape=[None, h_size, w_size], name="in_") # [100, 28, 28, 1]
Y = tf.placeholder(tf.float32, shape=[None, 16])

#X = tf.transpose(X, [0, 2, 1])
#X = tf.reshape(X, [-1, w_size])
#image -> vector sequence
#50*6을 50*1의 6개의 sequence로 변환
x_split = tf.split(X, h_size, axis=1)

#rnn 모델에 필요한 인스턴스
U = tf.Variable(tf.random_normal([w_size, hidden_size], stddev=0.01))
W = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # always square
V = tf.Variable(tf.random_normal([hidden_size, 16], stddev=0.01))

#hidden state
s = {}
s_init = tf.random_normal(shape=[batch_size, hidden_size], stddev=0.01)
s[-1] = s_init


#model specification
for t, x_split in enumerate(x_split):
    x = tf.reshape(x_split, [batch_size, w_size]) # [100, 1, 28, 1] -> [100, 28]
    s[t] = tf.nn.tanh(tf.matmul(x, U) + tf.matmul(s[t-1], W))

pred_softmax=tf.nn.softmax(tf.matmul(s[h_size - 1], V), name="out_")
print(pred_softmax.shape)
#cost = -tf.reduce_mean(tf.log(tf.reduce_sum(pred_softmax*Y, axis=1)))

cost_pre=tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_softmax, labels=Y)
loss = tf.reduce_mean(cost_pre)

#training
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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
test_inputs = x_test[:batch_size]
test_outputs = y_test[:batch_size]

'''
def accuracy(network, t):
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))
'''
total_batch=int(ntrain/batch_size)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(total_epoch):

        for i in range(total_batch):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            predic_train, acc_train, loss_train = sess.run([pred_softmax, acc, loss], feed_dict={
                X: batch_x, Y: batch_y})

        predic_test, acc_test, loss_test = sess.run([pred_softmax, acc, loss], feed_dict={
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
    prediction, acc, loss = sess.run([pred_softmax, acc, loss], feed_dict={
        X: test_inputs, Y: test_outputs})


print()
#print(f'final results: accuracy: {acc_final} loss: {loss_final}')
pickle.dump(prediction, open("predictions.p", "wb"))
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

print(prediction.__class__)
print(prediction.shape)


LABELS = ['1', '2', '3', '4', '5', '6','7', '8', '9', '10', '11', '12','13', '14', '15', '16']
max_test = np.argmax(test_outputs, axis=1)
max_predictions = np.argmax(prediction, axis=1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()