import imageio
import os
import numpy as np
from scipy.misc import imresize
import sys, getopt, re, argparse
import tensorflow as tf
from time import time

sys.path.append('../../Training/')
from ReadHDF5 import *

TFVERSION = tf.__version__.split('.')

if int(TFVERSION[1]) < 4 and int(TFVERSION[0]) <= 1:
	import keras
	from keras.layers import Input, concatenate
	from keras.layers.core import Reshape
	from keras.models import Model, load_model
elif int(TFVERSION[1]) == 4:
	from tensorflow import keras
	from tensorflow.python.keras._impl.keras.layers import Input, concatenate
	from tensorflow.python.keras._impl.keras.layers.core import Reshape
	from tensorflow.python.keras._impl.keras.models import Model, load_model
else:
	from tensorflow import keras
	from tensorflow.python.keras.layers import Inpur, concatenate
	from tensorflow.python.keras.layers.core import Reshape
	from tensorflow.python.keras.models import Model, load_model


# Keras' definition converted to numpy...
def softmax(x, axis=-1):
	ndim = np.ndim(x)
	if ndim >= 2:
		e = np.exp(x - np.max(x, axis=axis, keepdims=True))
		s = np.sum(e, axis=axis, keepdims=True)
		return e / s
	else:
		raise ValueError('Cannot apply softmax to a tensor that is 1D')

def load_multigpu_model(model_to_load):
	mgpu_net = load_model(model_to_load, custom_objects={'tf':tf}, compile=False)
	return mgpu_net.layers[-2]

# Loads the models as-is
def consensus_models(list_of_models, model_load_function=load_model):
	all_models = [model_load_function(model_name) for model_name in list_of_models]
	new_model_input = Input(shape=(16, 112, 112, 1))
	all_outputs = [indv_model(new_model_input) for indv_model in all_models]
	if len(all_outputs)==1:
		new_model = Model(inputs=new_model_input, outputs=all_outputs[0])
	else:
		new_model = Model(inputs=new_model_input, outputs=Reshape((len(list_of_models),2))(concatenate(all_outputs, axis=-1)))
	new_model.compile('adam','categorical_crossentropy')
	return new_model

# Actually removes the last layer in the network (softmax)...
def consensus_models_softmax(list_of_models, model_load_function=load_model):
	all_models = [model_load_function(model_name) for model_name in list_of_models]
	for model in all_models:
		model.pop()
	new_model_input = Input(shape=(16, 112, 112, 1))
	all_outputs = [indv_model(new_model_input) for indv_model in all_models]
	if len(all_outputs)==1:
		new_model = Model(inputs=new_model_input, outputs=all_outputs[0])
	else:
		new_model = Model(inputs=new_model_input, outputs=Reshape((len(list_of_models),2))(concatenate(all_outputs, axis=-1)))
	new_model.compile('adam','categorical_crossentropy')
	return new_model

def flip_input_batch(batch_input_single):
	assert len(np.shape(batch_input_single))==4 or (len(np.shape(batch_input_single))==5 and np.shape(batch_input_single)[0]==1)
	if len(np.shape(batch_input_single))==5:
		transpose_shape = (0,1,3,2,4)
	else:
		transpose_shape = (0,2,1,3)
	batch_input = np.reshape([batch_input_single,
							np.flipud(batch_input_single),
							np.fliplr(batch_input_single),
							np.fliplr(np.flipud(batch_input_single)),
							np.transpose(batch_input_single,transpose_shape),
							np.transpose(np.flipud(batch_input_single),transpose_shape),
							np.transpose(np.fliplr(batch_input_single),transpose_shape),
							np.transpose(np.fliplr(np.flipud(batch_input_single)),transpose_shape)], [8, np.shape(batch_input_single)[-4], np.shape(batch_input_single)[-3], np.shape(batch_input_single)[-2], np.shape(batch_input_single)[-1]])
	return batch_input

# Function to process all the data based on an image iterator
def process_video_frames(net, data_iterator, validation_dataset, output_name):
	file_consensus = open(os.path.splitext(output_name)[0] + '_meancons.csv', 'w')
	framenum = 0
	while True:
		try:
			start_time = time()
			current_data = next(data_iterator)
			frames = current_data[0]
			batch_input_single = np.reshape(frames,[16, 112, 112, 1])
			batch_input = flip_input_batch(batch_input_single)
			# Time logging...
			if framenum % 1000 == 0:
				print('Batch ' + str(framenum))
				print('Batch Assembled in: ' + str(time()-start_time))
			start_time = time()
			# Run the prediction
			results_nosoftmax = net.predict(batch_input, batch_size=8)
			# Time logging...
			if framenum % 1000 == 0:
				print('Batch Computed in: ' + str(time()-start_time))
			start_time = time()
			# Compute the other items from the prediction (softmax, argmax)
			results_nosoftmax = np.reshape(results_nosoftmax, [-1, 2])
			results = softmax(results_nosoftmax)
			predictions = np.argmax(results, 1)
			# Consensus predictions
			mean_pred = np.mean(results[:,1])
			# ','.join([str(current_data[2]), str(mean_pred)])
			file_consensus.write(str(current_data[2]) + ', ' + str(mean_pred) + '\n')
			file_consensus.flush()
			# Time logging...
			if framenum % 1000 == 0:
				print('Batch Saved in: ' + str(time()-start_time))
			framenum = framenum + 1
		except StopIteration:
			break
	file_consensus.close()

# Wrapper for cropped video processing
def process_dataset(net, dataset_name, output_name):
	dataset = Groom_Dataset(dataset_name, 16, 8, 2, 2, True, False, False)
	data_iterator = dataset.get_full_valid_generator()
	process_video_frames(net, data_iterator, dataset_name, output_name)

def main(argv):
	parser = argparse.ArgumentParser(description='Obtain Validation Inferences for 3DConv Models')
	parser.add_argument('--dataset', help='Name of dataset to process')
	parser.add_argument('--output', help='Output file', default='_ellfit.npy')
	parser.add_argument('--network', '--networks', help='Networks to use during inference', default='3Dconv_Keras.h5', nargs='+')

	args = parser.parse_args()
	arg_dict = args.__dict__

	# Actually load the models for full consensus
	net = consensus_models_softmax(args.network, load_multigpu_model)

	process_dataset(net, args.dataset, args.output)


if __name__ == '__main__':
	main(sys.argv[1:])

