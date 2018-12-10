import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

#ckpt, pbtxt -> pb로 만들어주는 모듈

MODEL_NAME = 'rnnraw'

# Freeze the graph
input_graph_path = '../rnnraw_shape40_overlapmax_e300_lr0.0001_98/' + MODEL_NAME+'.pbtxt'
checkpoint_path = '../rnnraw_shape40_overlapmax_e300_lr0.0001_98/' +MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "out_"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = '../frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = '../optimized_'+MODEL_NAME+'.pb'
clear_devices = True

#체크포인트와 pbtxt파일을 가지고와 model을 freeze시키는 메소드
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["in_"], # an array of the input node(s)
        ["out_"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())

# tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)

'''
MODEL_NAME = 'rnn_v2_model_check'
input_graph_path = MODEL_NAME+'.pbtxt'
checkpoint_path = './'+MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "y_"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True


freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
'''
'''
input_graph_path = "../checkpoint_har/har.pbtxt"
checkpoint_path = '../checkpoint_har/har.ckpt'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const"
output_frozen_graph_name = '../checkpoint_har/frozen_rnn_v2_model_check.pb'

freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=False, input_checkpoint=checkpoint_path,
                          output_node_names="y_", restore_op_name=restore_op_name,
                          filename_tensor_name=filename_tensor_name,
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")
'''
'''
freeze_graph.freeze_graph(input_graph = "../rnn_model.pbtxt",  input_saver = "",
             input_binary = False, input_checkpoint = "../rnn_model.ckpt", output_node_names = "Y",
             restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0",
             output_graph = "frozen_har.pb", clear_devices = True, initializer_nodes = "")
'''

#"frozen_har.pb".encode('utf-8')
#---------------------------------------------
"""

input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_har.pb', "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        [""],
        ["Y"],
        tf.float32.as_datatype_enum)

f = tf.gfile.FastGFile("optimized_frozen_har.pb", "w")
f.write(output_graph_def.SerializeToString())
"""
#---------------------------------------
"""
meta_path = './rnn_model.ckpt.meta' # Your .meta file
output_node_names = ['Y']    # Output nodes

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('.'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('output_graph.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
      """

"""
saver = tf.train.import_meta_graph('./rnn_model.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./rnn_model.ckpt")

output_node_names="y_pred"
output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session
            input_graph_def, # input_graph_def is useful for retrieving the nodes
            output_node_names.split(",")
)

output_graph = "./random_normal.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
"""

"""


MODEL_NAME = 'rnn_model.ckpt'

#Model_Name = 'rnn_model_50Hz_16'   #모델의 각 부분별로 체크포인트파일이 담겨있는 디렉토리


input_graph_path = MODEL_NAME+'.index'
checkpoint_path = MODEL_NAME
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'


freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=False, input_checkpoint=checkpoint_path,
                          output_node_names="Y", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

"""
