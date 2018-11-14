#This is code I used for freezing the graph:

'''
Code from https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
'''

import os, argparse

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))

def my_freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    print(model_dir)
    print(output_node_names)
    print(tf.gfile.Exists(model_dir))


    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    print(absolute_model_dir)
    print(output_graph)
    print(absolute_model_dir)

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )
        print(output_graph)
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        '''
        output_node_name = "output"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        clear_devices = True

        print("output path")
        print(absolute_model_dir)

        output_frozen_graph_path=absolute_model_dir + "/fb_lstm_model.pb"
        input_graph =absolute_model_dir + "/frozen_model.pbtxt"
        checkpoint_path = absolute_model_dir + "/model.ckpt"
        freeze_graph.freeze_graph(input_graph, '', False,checkpoint_path,
                              absolute_model_dir, output_node_name, restore_op_name,
                              filename_tensor_name, output_frozen_graph_path,
                              clear_devices, "")
        '''
        input_graph_path = absolute_model_dir + "/model.pbtxt"
        checkpoint_path = absolute_model_dir + "/model.ckpt"
        input_saver_def_path = ""
        input_binary = False
        output_node_names = "output"
        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        output_frozen_graph_name = absolute_model_dir + "/fb_lstm_model.pb"
        clear_devices = True

        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, checkpoint_path, output_node_names,
                                  restore_op_name, filename_tensor_name,
                                  output_frozen_graph_name, clear_devices, "")

    return output_graph_def
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="C:/Users/yegilee/Documents/PycharmProjects/Tensorflow/save1/", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="output", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()
    #print(tf.gfile.Exists("C:/Users/yegilee/Documents/PycharmProjects/Tensorflow/save1/"))

    my_freeze_graph(args.model_dir, args.output_node_names)


'''
output_node_name = "output"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = True

    (directory, fn, ext) = splitDirFilenameExt(input_graph_path)
    output_frozen_graph_path = os.path.join(directory, fn + '_frozen.pb')

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary,
                              checkpoint_path, output_node_name, restore_op_name,
                              filename_tensor_name, output_frozen_graph_path,
                              clear_devices, "")
'''
'''
if __name__ == '__main__':
    output_node_name="output"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = True
    input_graph_path = "./save/model.ckpt.meta"

    #(directory, fn, ext) = splitDirFilenameExt(input_graph_path)
    output_frozen_graph_path = './save/rnn_frozen.pb'

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary,
                              checkpoint_path, output_node_name, restore_op_name,
                              filename_tensor_name, output_frozen_graph_path,
                              clear_devices, "")
'''


