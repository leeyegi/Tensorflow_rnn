#데이터 preprocessing 후 rnn을 이용해 칫솔질 데이터 학습시키는 파일
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

#데이터 파일
file_name=['../dataset_v3/2_HJH_2018_11_19_1_log.txt', '../dataset_v3/2_HJH_2018_11_21_3_log.txt',
           '../dataset_v3/2_HJH_2018_11_26_3_log.txt','../dataset_v3/2_HJH_2018_11_27_1_log.txt',
           '../dataset_v3/2_HJH_2018_11_27_2_log.txt','../dataset_v3/2_HJH_2018_11_28_2_log.txt',
           '../dataset_v3/2_HJH_2018_11_28_3_log.txt','../dataset_v3/2_HJH_2018_11_29_1_log.txt',
           '../dataset_v3/HJH_20181201.txt',
           '../dataset_v2/HJH_2018_10_16_1_log.txt','../dataset_v2/HJH_2018_10_24_3_log.txt'

            ]

#데이터를 불러와 data preprocessing모듈에서 dataframe형성해 label이 달린 데이터를 반환
get_df_data = data_preprocessing.get_data(file_name)

#반환된 데이터를 datapreprocessing에서 data와 label로 분리 numpy 형식 
reshaped_segments, reshaped_labels=data_preprocessing.data_shape(get_df_data)

#해당 데이터들 shuffle 수행
c = list(zip(reshaped_segments, reshaped_labels))
random.shuffle(c)
reshaped_segments, reshaped_labels = zip(*c)

reshaped_segments=np.array(reshaped_segments)
reshaped_labels=np.array(reshaped_labels)

#데이터들을 trainset과 testset으로 나눔
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
h_size = 40
w_size = 6
c_size = 1
hidden_size = 512
total_epoch=3000
learning_rate = 0.0001


#reset graph
tf.reset_default_graph()

#===================================
#rnn 모델 
#===================================
#placeholder
X = tf.placeholder(tf.float32, shape=[None, h_size, w_size], name="in_") # [100, 28, 28, 1]
Y = tf.placeholder(tf.float32, shape=[None, 16])
init_state = tf.placeholder(tf.float32, shape=[None, hidden_size], name="hidden_")      #학습하는 데이터 크기를 담는 placeholder
print("placeholder")
print(X.shape)
print(Y.shape)
print(init_state.shape)


#X = tf.transpose(X, [0, 2, 1])
#X = tf.reshape(X, [-1, w_size])
#image -> vector sequence
#50*6을 50*1의 6개의 sequence로 변환
x_split = tf.split(X, h_size, axis=1)

#rnn 모델에 필요한 인스턴스
U1 = tf.Variable(tf.random_normal([w_size, hidden_size], stddev=0.01))
W1 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # always square

U2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01))
W2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # always square

U3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01))
W3 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01)) # always square
V = tf.Variable(tf.random_normal([hidden_size, 16], stddev=0.01))

print("state shape")
print(U1.shape)
print(W1.shape)
print(U2.shape)
print(W2.shape)
print(U3.shape)
print(W3.shape)
print(V.shape)


batch_size_hidden = tf.shape(init_state)[0]

#hidden state
s1 = {}
s1_init = tf.random_normal(shape=[batch_size_hidden,hidden_size], stddev=0.01)
s1[-1] = s1_init

s2 = {}
s2_init = tf.random_normal(shape=[batch_size_hidden,hidden_size], stddev=0.01)
s2[-1] = s2_init

s3 = {}
s3_init = tf.random_normal(shape=[batch_size_hidden,hidden_size], stddev=0.01)
s3[-1] = s3_init

print("s init")
print(s1_init.shape)
print(s2_init.shape)
print(s3_init.shape)

print("s")
print(s1.__sizeof__)
print(s2.__sizeof__)
print(s3.__sizeof__)

#model specification
for t, x_split in enumerate(x_split):
    x = tf.reshape(x_split, [batch_size_hidden, w_size]) # [100, 1, 28, 1] -> [100, 28]
    s1[t] = tf.nn.relu(tf.matmul(x, U1) + tf.matmul(s1[t-1], W1))
    s2[t] = tf.nn.tanh(tf.matmul(s1[t], U2) + tf.matmul(s1[t-1], W2))
    s3[t] = tf.nn.tanh(tf.matmul(s2[t], U3) + tf.matmul(s2[t-1], W3))


#모델 prediction
pred_softmax=tf.nn.softmax(tf.matmul(s3[h_size - 1], V), name="out_")
print(pred_softmax.shape)
#cost = -tf.reduce_mean(tf.log(tf.reduce_sum(pred_softmax*Y, axis=1)))

#prediction한 모델의 loss값 구함
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

'''
def accuracy(network, t):
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))
'''
total_batch=4
#===================================
#rnn 모델 학습 
#===================================
with tf.Session() as sess:
    sess.run(init)
    
    #에폭수만큼 학습진행
    for epoch in range(total_epoch):
        
        #batchsize만큼 학습 진행
        for i in range(4):
            batch_x = x_train[i * batch_size:(i + 1) * batch_size]
            batch_y = y_train[i * batch_size:(i + 1) * batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, init_state : np.zeros((batch_x.shape[0], hidden_size))})

            predic_train, acc_train, loss_train = sess.run([pred_softmax, acc, loss], feed_dict={
                X: batch_x, Y: batch_y, init_state : np.zeros((batch_x.shape[0], hidden_size))})
        
        #validation
        predic_test, acc_test, loss_test = sess.run([pred_softmax, acc, loss], feed_dict={
            X: test_inputs, Y: test_outputs, init_state : np.zeros((test_inputs.shape[0], hidden_size))})

        print(f'epoch: {epoch} batch {i} test accuracy: {acc_test} loss: {loss_test}')

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)

    tf.train.write_graph(sess.graph_def, '.', '../rnnraw.pbtxt')

    #predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, cost],
    #                                              feed_dict={X: testimgs, Y: testlabels})
    saver.save(sess, save_path="../rnnraw.ckpt")
    
    #test
    prediction, acc, loss = sess.run([pred_softmax, acc, loss], feed_dict={
        X: test_inputs, Y: test_outputs,init_state : np.zeros((test_inputs.shape[0], hidden_size))})


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


#confusion matrix 그리기
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