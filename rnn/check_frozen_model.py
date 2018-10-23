import tensorflow as tf

g=tf.GraphDef()


print(g.ParseFromString(open("../frozen_rnn_model_check.pb",'r',encoding="UTF8")))

