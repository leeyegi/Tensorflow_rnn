import tensorflow as tf
from tensorflow.contrib import rnn

#import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)

#output_dir='/data3/frankzhu/tmp/mnistmodel'
output_dir='save1'

#하이퍼파라미터
time_steps=28
num_units=128
n_input=28
learning_rate=0.001
n_classes=10
batch_size=128

#weight와 bias
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]),name="weights")
out_bias=tf.Variable(tf.random_normal([n_classes]),name="bias")

#placeholder선언
x=tf.placeholder("float",[None,time_steps,n_input])
y=tf.placeholder("float",[None,n_classes])

#input_tensor
input=tf.unstack(x ,time_steps,1,name="input_tensor")

#lstm model
#lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
#lstm_layer=rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units) for _ in range(3)])
#lstm_layer=rnn.LSTMBlockCell(num_units,forget_bias=1)
lstm_layer=tf.nn.rnn_cell.BasicLSTMCell(num_units)
#lstm_layer=tf.nn.rnn_cell.GRUCell(num_units)
#lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#lstm으로 학습후 prediction
prediction=tf.add(tf.matmul(outputs[-1],out_weights), out_bias, name="output")

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    iter=1
    while iter<100:
        #배치 사이즈 만큼 뽑아서 학습 진행
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
        batch_x=batch_x.reshape((batch_size,time_steps,n_input))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        if not tf.gfile.Exists('save1'):
            tf.gfile.MakeDirs('save1')
        tf.train.write_graph(sess.graph_def, '.', output_dir+'/model.pbtxt')

        # 모델저장
        saver = tf.train.Saver()
        filename = saver.save(sess, output_dir + '/model.ckpt')

        iter=iter+1

