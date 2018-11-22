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


file_name=[#'../dataset_v2/HJH_2018_10_03_3_log.txt', '../dataset_v2/HJH_2018_10_04_3_log.txt',
           #'../dataset_v2/HJH_2018_10_05_2_log.txt','../dataset_v2/HJH_2018_10_06_3_log.txt',
           #'../dataset_v2/HJH_2018_10_12_3_log.txt','../dataset_v2/HJH_2018_10_13_1_log.txt',
           #'../dataset_v2/HJH_2018_10_15_3_log.txt','../dataset_v2/HJH_2018_10_16_1_log.txt',      #데이터 조음
           #'../dataset_v2/HJH_2018_10_17_3_log.txt','../dataset_v2/HJH_2018_10_22_3_log.txt',
           '../dataset_v2/HJH_2018_10_24_3_log.txt'                                                #데이터 조음
            ]


get_df_data = data_preprocessing.get_data(file_name)

reshaped_segments, reshaped_labels=data_preprocessing.data_shape(get_df_data)

c = list(zip(reshaped_segments, reshaped_labels))

random.shuffle(c)

reshaped_segments, reshaped_labels = zip(*c)

reshaped_segments=np.array(reshaped_segments)
reshaped_labels=np.array(reshaped_labels)

x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, reshaped_labels, test_size=0.2, random_state=42)
#train data 수, test data 수, dim, class수
ntrain, ntest, dim, nclasses \
 = x_train.shape[0], x_test.shape[0], x_train.shape[1], y_train.shape[1]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(ntrain)
print(ntest)
print(dim)
print(nclasses)



#hyperparameter
batch_size = int(ntrain/4)
print("batch")
print(batch_size)
h_size = 50
w_size = 6
c_size = 1
hidden_size = 512
total_epoch=30
learning_rate = 0.0001


#reset graph
tf.reset_default_graph()
X = tf.placeholder(tf.float64, shape=(None, h_size, w_size))
Y = tf.placeholder(tf.float64, shape=(None, 16))
init_state = tf.placeholder(tf.float64, shape=(None, hidden_size))

# Simple RNN
Wxh = tf.get_variable('Wxh', shape=(w_size, hidden_size),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
Whh = tf.get_variable('Whh', shape=(hidden_size, hidden_size),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
Wh = tf.concat([Wxh, Whh], axis=0)
Why = tf.get_variable('Why', shape=(hidden_size, 16),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
bh = tf.get_variable('bh', shape=(1, hidden_size),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
by = tf.get_variable('by', shape=(w_size, 1),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

# Output layer
linear_w = tf.get_variable('linear_w', shape=(16, w_size),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
linear_b = tf.get_variable('linear_b', shape=(w_size),
                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

y_preds = list()
losses = list()
hiddens = np.zeros(h_size+1, dtype=np.object)
hiddens[-1] = init_state

unstacked_inputs = tf.unstack(X, axis=1)
unstacked_outputs = tf.unstack(Y, axis=1)

for t, (input_t, y_true) in enumerate(zip(unstacked_inputs, unstacked_outputs)):
    concat_x = tf.concat([input_t, hiddens[t-1]], axis=1)
    hidden = tf.tanh(tf.matmul(concat_x, Wh) + bh)



pred_softmax=tf.nn.softmax(tf.matmul(hidden[h_size - 1], Why), name="out_")
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
test_inputs = x_test
test_outputs = y_test
print("test")
print(test_inputs.shape)
print(test_outputs)

'''
def accuracy(network, t):
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))
'''
total_batch=4

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(total_epoch):

        for i in range(4):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]
            print("x batch")
            print(batch_x.shape[0])

            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, init_state:np.zeros((batch_x.shape[0], hidden_size))})
            predic_train, acc_train, loss_train = sess.run([pred_softmax, acc, loss], feed_dict={
                X: batch_x, Y: batch_y, init_state:np.zeros((batch_x.shape[0], hidden_size))})

        predic_test, acc_test, loss_test = sess.run([pred_softmax, acc, loss], feed_dict={
            X: test_inputs, Y: test_outputs, init_state:np.zeros((test_inputs.shape[0], hidden_size))})

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
        X: test_inputs, Y: test_outputs, init_state:np.zeros((test_inputs.shape[0], hidden_size))})


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
max_predictions.tolist()
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)
cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

print("cm")
print(max_test.shape)
print(max_test)

print(max_predictions.__len__)
print(max_predictions)

print(confusion_matrix)
print(confusion_matrix.shape)

print(cm)
print(cm.shape)

plt.figure(figsize=(16, 16))
sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f")
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()