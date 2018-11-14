import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from rnn_v2 import data_preprocessing
import pickle
import tensorflow as tf

#tflite를 만들기위한 nn
#input -> in_
#predic -> y_
#데이터 처리를 위해 필요한 인스턴스
N_TIME_STEPS = 50
N_FEATURES = 6
step = 25
RANDOM_SEED = 42
segments = []
labels = []
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


#데이터를 랜덤으로 섞어 훈련데이터와 테스트 데이터 setting함
x_train, x_test, y_train, y_test = train_test_split(reshaped_segments_2, reshaped_labels, test_size=0.2, random_state=RANDOM_SEED)

# hyperparameters
learning_rate = 0.0025
training_epochs = 10000
batch_size = 500
display_step = 1

# 네트워크 구성하기위한 hyperparameter
n_hidden_1 = 512
n_hidden_2 = 512
n_feature = 50
n_timp_step=6
n_classes = 16

#reset graph
#tf.reset_default_graph()

# tf Graph input
X = tf.placeholder("float", [None, n_timp_step*n_feature], name='in_')
Y = tf.placeholder("float", [None, n_classes])


# weight와 bias 지정
weights = {
    'h1': tf.Variable(tf.random_normal([n_timp_step*n_feature, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#모델 생성
def multilayer_perceptron(x):
    #X = tf.transpose(x, [1, 0, 2])
    #X = tf.reshape(X, [-1, N_FEATURES])
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    print(layer_1.shape)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

logits = multilayer_perceptron(X)

pred_softmax = tf.nn.softmax(logits, name='y_')


#loss와 optimizer를 지정
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

# Initializing the variables

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
summary_op = tf.summary.merge_all()

history = dict(train_loss=[],
               train_acc=[],
               test_loss=[],
               test_acc=[])

train_count = len(x_train)

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(1, training_epochs + 1):
        for start, end in zip(range(0, train_count, batch_size),
                              range(batch_size, train_count + 1, batch_size)):
            sess.run(optimizer, feed_dict={X: x_train[start:end],
                                           Y: y_train[start:end]})


        predic_train, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss_op], feed_dict={
            X: x_train, Y: y_train})

        predic_train, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss_op], feed_dict={
            X: x_test, Y: y_test})

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)

        if i != 1 and i % 10 != 0:
            continue

        print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')
    tf.train.write_graph(sess.graph_def, '.', './checkpoint/rnn.pbtxt')

    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss_op], feed_dict={X: x_test, Y: y_test})
    saver.save(sess, save_path = "./checkpoint/rnn.ckpt")

print()
print(f'final results: accuracy: {acc_final} loss: {loss_final}')
pickle.dump(predictions, open("predictions.p", "wb"))
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
