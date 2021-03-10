import keras
from keras.models import load_model, Model, Sequential
from keras.layers import Input, concatenate
from keras.layers.core import Reshape
import imageio
import os
import numpy as np
from scipy.misc import imresize
import sys, getopt, re, argparse
import tensorflow as tf
import matplotlib.cm as cm
from time import time
import cv2

def load_multigpu_model(model_to_load):
	mgpu_net = load_model(model_to_load, custom_objects={'tf':tf}, compile=False)
	return mgpu_net.layers[-2]

# Actually removes the last layer in the network (softmax)...
def consensus_models_softmax(list_of_models, model_load_function=load_model):
	all_models = [model_load_function(model_name) for model_name in list_of_models]
	for model in all_models:
		model.pop()
	new_model_input = Input(shape=(16, 112, 112, 1))
	all_outputs = [indv_model(new_model_input) for indv_model in all_models]
	if len(all_outputs)==1:
		new_model = Model(input=new_model_input, output=all_outputs[0])
	else:
		new_model = Model(input=new_model_input, output=Reshape((len(list_of_models),2))(concatenate(all_outputs, axis=-1)))
	new_model.compile('adam', 'categorical_crossentropy')
	# Re-frame the model?
	config = new_model.get_config()
	weights = new_model.get_weights()
	# Re-build a model where the learning phase is now hard-coded to 0.
	new_new_model = Model.from_config(config)
	new_new_model.set_weights(weights)
	return new_new_model


keras.backend.set_learning_phase(False)

net = consensus_models_softmax(['3Dconv_Model1.h5','3Dconv_Model2.h5','3Dconv_Model3.h5','3Dconv_Model4.h5'], load_multigpu_model)

saver2 = tf.train.Saver(tf.global_variables())
checkpoint_path = saver2.save(keras.backend.get_session(), './exported/KerasConsensusModel')
tf.train.write_graph(keras.backend.get_session().graph, '.', "./exported/KerasConsensusModel.pb", as_text=False)
# Plaintext version to find the exact tensor name that you're interested in...
tf.train.write_graph(keras.backend.get_session().graph, '.', "./exported/KerasConsensusModel.pbtxt", as_text=True)
