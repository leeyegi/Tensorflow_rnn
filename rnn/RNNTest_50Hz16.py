import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from rnn import overlap_splite6_update

##############################
#6개의 서브 세트가지고와서 데이터 후처리
##############################
#3개의 파일을 가져와서 txt모듈에 있는 get함수를 호출해 리턴되는 값을 result에 저장
#다양한 파일의 데이터 셋을 가져오는 경우 getmany함수를 사용 - 매개변수는 리스트 형식
dataset= overlap_splite6_update.getmany([['dataset/2018_5_4_13_32_6g.txt', 'dataset/2018_5_4_13_32_6a.txt', 'dataset/2018_5_4_13_32_6tg_16.txt'],
                                         ['dataset/2018_5_4_13_46_47g.txt', 'dataset/2018_5_4_13_46_47a.txt', 'dataset/2018_5_4_13_46_47tg_16.txt'],
                                         ['dataset/2018_5_4_22_31_51g.txt', 'dataset/2018_5_4_22_31_51a.txt', 'dataset/2018_5_4_22_31_51tg_16.txt'],
                                         ['dataset/2018_5_8_20_28_25g.txt', 'dataset/2018_5_8_20_28_25a.txt','dataset/2018_5_8_20_28_25tg_16.txt'],
                                         ['dataset/2018_5_8_20_42_14g.txt', 'dataset/2018_5_8_20_42_14a.txt','dataset/2018_5_8_20_42_14tg_16.txt'],
                                         ['dataset/2018_5_8_20_58_21g.txt', 'dataset/2018_5_8_20_58_21a.txt','dataset/2018_5_8_20_58_21tg_16.txt'],

                                         ['dataset/2018_5_14_20_41_36g.txt', 'dataset/2018_5_14_20_41_36a.txt', 'dataset/2018_5_14_20_41_36tg_16.txt'],
                                         ['dataset/2018_5_14_21_6_23g.txt', 'dataset/2018_5_14_21_6_23a.txt', 'dataset/2018_5_14_21_6_23tg_16.txt']

                                         ])

#result - class와 학습할 데이터 값이 같이 들어있는 리스트
#       - > x(학습할 데이터) 와 y(클래스) 를 분리
def seperate_x_y_data(dataset_idx):
    # 칫솔질 데이터를 받기위한 배열선언
    # xfile은 학습할 데이터 값이 저장되는 리스트
    # yfile은 class가 저장되는 리스트
    xfile = []
    yfile = []
    for i in dataset_idx:
        yfile.append(i[0])
        xfile.append(i[1:])
    return xfile, yfile

dataset1_xfile, dataset1_yfile=seperate_x_y_data(dataset[0])
dataset2_xfile, dataset2_yfile=seperate_x_y_data(dataset[1])
dataset3_xfile, dataset3_yfile=seperate_x_y_data(dataset[2])
dataset4_xfile, dataset4_yfile=seperate_x_y_data(dataset[3])
dataset5_xfile, dataset5_yfile=seperate_x_y_data(dataset[4])
dataset6_xfile, dataset6_yfile=seperate_x_y_data(dataset[5])

print(len(dataset1_xfile))
print(len(dataset1_yfile))


#받은 리스트를 넘파이 배열로 저장
#dataset_x의 형상은 (데이터셋 크기,10,6)
#dataset_y의 형상은 (데이터셋 크기, )
def list_to_array(dataset_xfile, dataset_yfile):
    dataset_x=np.array(dataset_xfile, dtype=np.float32)
    dataset_y = np.array(dataset_yfile, dtype=np.int8)
    return dataset_x, dataset_y

dataset1_x, dataset1_y=list_to_array(dataset1_xfile, dataset1_yfile)
dataset2_x, dataset2_y=list_to_array(dataset2_xfile, dataset2_yfile)
dataset3_x, dataset3_y=list_to_array(dataset3_xfile, dataset3_yfile)
dataset4_x, dataset4_y=list_to_array(dataset4_xfile, dataset4_yfile)
dataset5_x, dataset5_y=list_to_array(dataset5_xfile, dataset5_yfile)
dataset6_x, dataset6_y=list_to_array(dataset6_xfile, dataset6_yfile)
'''
print(dataset6_x.shape)
print(dataset6_y.shape)
print(dataset6_y)
'''

#class를 one hot encoding으로 만들어 y의 형상을(데이터셋 크기, 16)으로 바꿈
'''print("y배열 one hot으로 바꿈")'''
def label_to_onehot(dataset_y):
    dataset_y_onehot = np.zeros((dataset_y.size, 17))
    dataset_y_onehot[np.arange(dataset_y.size), dataset_y] = 1
    '''
    print(dataset_y.shape)
    print(dataset_y_onehot.shape)
    '''
    return dataset_y_onehot

dataset1_y_onehot=label_to_onehot(dataset1_y)
dataset2_y_onehot=label_to_onehot(dataset2_y)
dataset3_y_onehot=label_to_onehot(dataset3_y)
dataset4_y_onehot=label_to_onehot(dataset4_y)
dataset5_y_onehot=label_to_onehot(dataset5_y)
dataset6_y_onehot=label_to_onehot(dataset6_y)
'''
print(dataset1_y_onehot.shape)
print(dataset6_y_onehot.shape)
'''

#전체 학습 & 테스트 할 데이터의 형상
#n_hidden과 total_batch를 위해 데이터 셋의 형상이 필요
'''print(dataset_x.shape)
print(dataset_y_onehot.shape)'''
(data_size_x, data_size_y, data_size_z)=dataset1_x.shape
'''print(data_size_x)'''


#########
# 옵션 설정(hyperharameter&architecture)
#########
learning_rate = 0.01              #현재 신경망이 단순하기 떄문에 학습이 금방끝나므로 학습률을 낮게 조정
total_epoch = 50                   #에폭 -> 훈련 데이터를 모두 소진했을때 에폭수가 1 올라감
batch_size = data_size_x*2                  #한번에 몇개의 데이터를 학습할지 배치사이즈 설정
display_step = 1

n_input = 50
n_step = 6
n_hidden = int(data_size_x/2)
n_class = 17


#########
# 신경망 모델 구성
# X -> (?, 50, 6)
# Y -> (?, 17)
# W -> (데이터셋 크기, 17)
# b -> (17, )
#########

def inference(x):
    # RNN 에 학습에 사용할 셀을 생성
    # BasicRNNCell,BasicLSTMCell,GRUCell
    cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
    cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, name="hidden2")
    cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=0.8)
    cell3 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, name="hidden3")
    cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, output_keep_prob=0.7)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3])

    # RNN 신경망을 생성합니다
    outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

    # 결과를 Y의 다음 형식과 바꿔야 하기 때문에
    # Y : [batch_size, n_class]
    # outputs 의 형태를 이에 맞춰 변경해야합니다.
    # outputs : [batch_size, n_step, n_hidden]
    #        -> [n_step, batch_size, n_hidden]
    #        -> [batch_size, n_hidden]
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = outputs[-1]

    W = tf.Variable(tf.random_normal([n_hidden, n_class]), name="W")
    b = tf.Variable(tf.random_normal([n_class]), name="b")

    model = tf.matmul(outputs, W) + b
    return model


def loss(model, y):
    #신경망의 손실값을 구함
    cost_pre = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y, name="cost_pre")
    cost = tf.reduce_mean(cost_pre, name="cost")
    return cost

def training(cost, global_step, model):
    #모델을 훈련시키는 메소드
    #손실값을 저장(텐서보드를 통해 loss값 확인하기 위함)
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step, name="optimizer")
    return optimizer

def evaluate(model, y):
    #만든 모댈을 평가하느 메소드
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1), name="is_correct")
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name="accuracy")
    tf.summary.scalar("acc", accuracy)
    return is_correct,accuracy



with tf.variable_scope("rnn_model_50hz_16"):
    X = tf.placeholder(tf.float32, [None, n_input, n_step], name="X")
    Y = tf.placeholder(tf.int32, [None, n_class], name="Y")

    model = inference(X)

    cost = loss(model, Y)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step, model)

    corr_op, eval_op = evaluate(model, Y)

    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.Session()

    summary_writer = tf.summary.FileWriter("rnn_model_50hz_16/",graph_def=sess.graph_def)

    init_op = tf.global_variables_initializer()

    sess.run(init_op)

    # 비용함수와 정확도 그래프를 그리기위해 학습과정에서 비용함수값과 정확도 값을 리스트로 저장
    # train_cost_list - 훈련데이터가 학습되면서 에폭당 비용함수 값을 저장하는 리스트
    # train_acc_list - 훈련제이터가 학습되면서 에폭당 비용함수 값을 저장하는 리스트
    train_cost_list = []
    train_acc_list = []


    # RNN 훈련 메소드
    def run_train(session, train_x, train_y):
        print("\nStart training")

        avg_cost = 0.

        # total epoch이 소진될때 까지 훈련
        for epoch in range(total_epoch):
            total_batch = int(train_x.shape[0] / batch_size)  # total_batch는 전체 훈련데이터 / 배치 사이즈
            # total_batch만큼 반복
            for i in range(total_batch):
                # 한번에 batchc_size만큼 학습
                batch_x = train_x[i * batch_size:(i + 1) * batch_size]
                batch_y = train_y[i * batch_size:(i + 1) * batch_size]

                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                cost_tmp = sess.run(cost, feed_dict={X: batch_x, Y: batch_y}) / total_batch
                avg_cost+=cost_tmp
            if epoch % display_step == 0:
                print("Epoch #%d " % (epoch), "cost =", "{:.9f}".format(cost_tmp))
                summary_str = sess.run(summary_op, feed_dict={X: batch_x, Y: batch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))


    def kfold_div(idx, data_x, data_y):
        index = [0, 1, 2, 3, 4]
        val_data_x = np.array(data_x[idx])
        val_data_y = np.array(data_y[idx])
        index.pop(idx)

        train_data_x = np.array(data_x[index[0]])
        train_data_y = np.array(data_y[index[0]])
        index.pop(0)
        for i in index:
            train_data_x = np.vstack([train_data_x, data_x[i]])
            train_data_y = np.vstack([train_data_y, data_y[i]])
        return train_data_x, train_data_y, val_data_x, val_data_y


    # k fold cross vaildation -> 교차검증 수행
    def cross_validate(session):
        acc_results = []
        data_x = [dataset1_x, dataset2_x, dataset3_x, dataset4_x, dataset5_x]
        data_y = [dataset1_y_onehot, dataset2_y_onehot, dataset3_y_onehot, dataset4_y_onehot, dataset5_y_onehot]
        for i in range(0, 5):
            train_data_x, train_data_y, val_data_x, valt_data_y = kfold_div(i, data_x, data_y)
            run_train(session, train_data_x, train_data_y)

            accuracy = sess.run(eval_op, feed_dict={X: val_data_x, Y: valt_data_y})
            print("val acc"+str(accuracy))
            acc_results.append(accuracy)


        return acc_results

    result = cross_validate(sess)
    prediction,accuracy = sess.run([corr_op,eval_op], feed_dict={X: dataset6_x, Y: dataset6_y_onehot})
    print("Test Accuracy:", accuracy)

    # matrix confusion을 위해 onehot -> label로 바꿈
    onehot_to_label = sess.run(tf.argmax(dataset6_y_onehot, axis=1))
    '''print(onehot_to_label)'''

    # cnt - 테스트 데이터의 클래스별 갯수 확인 리스트
    # cnt_T - 테스트 데이터의 예측값을 저장 - 올바르게 예측했을때
    # cnt_F - 테스트 데이터의 예측값을 저장 - 예측을 실패했을때
    cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cnt_T = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cnt_F = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(onehot_to_label)):
        tmp = onehot_to_label[i]
        cnt[tmp] += 1
        if prediction[i] == True:
            cnt_T[tmp] += 1
        else:
            cnt_F[tmp] += 1

    print(cnt)
    print("true")
    print(cnt_T)
    print("false")
    print(cnt_F)

    # metrics.confusion_matrix을 하기위해 테스트 데이터 예측값 필요
    prediction = tf.argmax(model, 1)
    best = sess.run([prediction], feed_dict={X: dataset6_x})
    '''print(best[0])'''

    # metrics.confusion_matrix그리기 - normalization 수행
    LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

    confusion_matrix = metrics.confusion_matrix(onehot_to_label, best[0])
    '''print(confusion_matrix)'''
    cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 16))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt=".2f")
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    saver.save(sess, "rnn_model_50hz_16/model-checkpoint_har", global_step=global_step)
    saver.save(sess, "./rnn_model.ckpt")




