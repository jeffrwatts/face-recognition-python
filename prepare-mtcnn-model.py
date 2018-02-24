import os
import sys
import numpy as np
import pandas as pd
from scipy import misc
from shutil import copyfile
import tensorflow as tf
print(tf.__version__)

sys.path.append("../facenet/src")
import facenet
import align.detect_face

from tensorflow.python.framework.graph_util import convert_variables_to_constants

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

# Create the models output directory if it is not present.
output_models_dir = "./models"

if not os.path.exists(output_models_dir):
    os.makedirs(output_models_dir)

# Load pnet cnn and freeze graph.  (Code adapted from facenet/src/align/det_face.py)
pnet_output_dir = os.path.join(output_models_dir, "pnet")

if not os.path.exists(pnet_output_dir):
    os.makedirs(pnet_output_dir)

pnet_model_file = os.path.join(pnet_output_dir, "pnet.pb")

outputs = ["pnet/conv4-2/BiasAdd", "pnet/prob1"]

with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=None, log_device_placement=False))
    with sess.as_default():
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = align.detect_face.PNet({'data':data})
            pnet.load("../facenet/src/align/det1.npy", sess)

        frozen_graph = freeze_session(sess, keep_var_names=None, output_names=outputs)
        tf.train.write_graph(frozen_graph, "", pnet_model_file, as_text=False)

# Load rnet cnn and freeze graph.  (Code adapted from facenet/src/align/det_face.py)
rnet_output_dir = os.path.join(output_models_dir, "rnet")

if not os.path.exists(rnet_output_dir):
    os.makedirs(rnet_output_dir)

rnet_model_file = os.path.join(rnet_output_dir, "rnet.pb")

outputs = ["rnet/conv5-2/conv5-2", "rnet/prob1"]

with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=None, log_device_placement=False))
    with sess.as_default():
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = align.detect_face.RNet({'data':data})
            rnet.load(os.path.join("../facenet/src/align/det2.npy"), sess)
        frozen_graph = freeze_session(sess, keep_var_names=None, output_names=outputs)
        tf.train.write_graph(frozen_graph, "", rnet_model_file, as_text=False)

# Load onet cnn and freeze graph.  (Code adapted from facenet/src/align/det_face.py)
onet_output_dir = os.path.join(output_models_dir, "onet")

if not os.path.exists(onet_output_dir):
    os.makedirs(onet_output_dir)

onet_model_file = os.path.join(onet_output_dir, "onet.pb")

outputs = ["onet/conv6-2/conv6-2", "onet/conv6-3/conv6-3", "onet/prob1"]
with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=None, log_device_placement=False))
    with sess.as_default():
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = align.detect_face.ONet({'data':data})
            onet.load(os.path.join("../facenet/src/align/det3.npy"), sess)
        frozen_graph = freeze_session(sess, keep_var_names=None, output_names=outputs)
        tf.train.write_graph(frozen_graph, "", onet_model_file, as_text=False)

facenet_output_dir = os.path.join(output_models_dir, "facenet")

if not os.path.exists(facenet_output_dir):
    os.makedirs(facenet_output_dir)

copyfile("./20170512-110547/20170512-110547.pb", "./models/facenet/facenet.pb")

print('Complete')
