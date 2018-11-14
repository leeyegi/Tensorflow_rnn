import tensorflow as tf
from datatools import input_data
mnist = input_data.read_data_sets("../data", one_hot=True)
from tensorflow.python.tools import freeze_graph


n_classes=16
time_steps=50
n_input=6
num_units=64
learning_rate=0.01
batch_size=100
output_dir="../"

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]),name="weights")
out_bias=tf.Variable(tf.random_normal([n_classes]),name="bias")

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1,name="input_tensor")

#defining the network
#lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
#lstm_layer=rnn.MultiRNNCell([rnn.BasicLSTMCell(num_units) for _ in range(3)])
#lstm_layer=rnn.LSTMBlockCell(num_units,forget_bias=1)
lstm_layer=tf.nn.rnn_cell.BasicLSTMCell(num_units)
#lstm_layer=tf.nn.rnn_cell.GRUCell(num_units)
#lstm_layer=tf.nn.rnn_cell.LSTMCell(num_units,forget_bias=1)
outputs,_=tf.contrib.rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
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
    while iter<800:
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

        # added
        saver = tf.train.Saver()
        filename = saver.save(sess, output_dir + '/model.ckpt')

        iter=iter+1

#========================================================================================
