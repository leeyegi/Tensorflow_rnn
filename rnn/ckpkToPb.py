import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'rnn_model'

input_graph_path = "../rnn_model_check.pbtxt"
checkpoint_path = '../rnn_model_check.ckpt'
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = '../frozen_rnn_model_check.pb'

freeze_graph.freeze_graph(input_graph_path, input_saver="",
                          input_binary=False, input_checkpoint=checkpoint_path,
                          output_node_names="y_", restore_op_name="save/restore_all",
                          filename_tensor_name="save/Const:0",
                          output_graph=output_frozen_graph_name, clear_devices=True, initializer_nodes="")

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
