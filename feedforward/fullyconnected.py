import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, 300], name='in_')
Y = tf.placeholder(tf.float32, [None, 16])
#keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([300, 512]))
L1 = tf.nn.relu(tf.matmul(X, W1))
# 텐서플로우에 내장된 함수를 이용하여 dropout 을 적용합니다.
# 함수에 적용할 레이어와 확률만 넣어주면 됩니다. 겁나 매직!!
#L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([512, 256]))
L2 = tf.nn.relu(tf.matmul(L1, W2))
#L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 128]))
L3 = tf.nn.relu(tf.matmul(L2, W3))
L3 = tf.nn.dropout(L3, 0.8)

W4 = tf.Variable(tf.random_normal([128, 64]))
L4 = tf.nn.relu(tf.matmul(L3, W4))
L4 = tf.nn.dropout(L4, 0.7)

W5 = tf.Variable(tf.random_normal([61, 16]))
model = tf.matmul(L4, W5)

pred_softmax = tf.nn.softmax(model, name='y_')

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.0025).minimize(cost)

#########
# 신경망 모델 학습
######
saver = tf.train.Saver()
init = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(init)

history = dict(train_loss=[],
               train_acc=[],
               test_loss=[],
               test_acc=[])

batch_size = 500
#total_batch = int(x_train/1000)
train_count=len(x_train)

for epoch in range(1,1000):
    total_cost = 0

    for start, end in zip(range(0, train_count, batch_size),
                          range(batch_size, train_count + 1, batch_size)):
        sess.run(optimizer, feed_dict={X: x_train[start:end],
                                       Y: y_train[start:end]})

    predic_train, acc_train, loss_train = sess.run([pred_softmax, accuracy, cost], feed_dict={
        X: x_train, Y: y_train})

    predic_train, acc_test, loss_test = sess.run([pred_softmax, accuracy, cost], feed_dict={
        X: x_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    history['test_loss'].append(loss_test)
    history['test_acc'].append(acc_test)


    print(f'epoch: {epoch} test accuracy: {acc_test} loss: {loss_test}')

tf.train.write_graph(sess.graph_def, '.', './checkpoint/fc.pbtxt')
predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, cost], feed_dict={X: x_test, Y: y_test})
saver.save(sess, save_path = "./checkpoint/fc.ckpt")

print('최적화 완료!')

#########
# 결과 확인
######
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: x_test,
                                   Y: y_test,}))

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

#heatmap그리기
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