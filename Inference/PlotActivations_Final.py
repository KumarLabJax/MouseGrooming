import imageio
import os
import numpy as np
from scipy.misc import imresize
import sys, getopt, re, argparse
import tensorflow as tf
import matplotlib.cm as cm
from time import time

# Keras' definition converted to numpy...
def softmax(x, axis=-1):
	ndim = np.ndim(x)
	if ndim >= 2:
		e = np.exp(x - np.max(x, axis=axis, keepdims=True))
		s = np.sum(e, axis=axis, keepdims=True)
		return e / s
	else:
		raise ValueError('Cannot apply softmax to a tensor that is 1D')

def main(argv):
	parser = argparse.ArgumentParser(description='Plotting 3DConv 4 Model Activations')
	parser.add_argument('--mov_name', help='Name of movie to process')

	args = parser.parse_args()
	arg_dict = args.__dict__

	if 'mov_name' in arg_dict.keys() and arg_dict['mov_name'] is not None:
		video_pattern = os.path.splitext(args.mov_name)[0]

	reader = imageio.get_reader(video_pattern+'.avi')
	writer = imageio.get_writer(video_pattern+'_out_Activations.avi', fps=30, codec='mpeg4', quality=10)
	im_iter = reader.iter_data()

	label_file = video_pattern+'_crop_raw.npy'
	label_data=[]
	with open(label_file,'rb') as file:
		while True:
			try:
				if sys.version_info[0]==2:
					label_data.append(np.load(file))
				else:
					label_data.append(np.load(file, encoding = 'bytes', allow_pickle = False))
			except IOError:
				break
			except ValueError:
				break

	label_data = np.reshape(label_data, [-1, np.shape(label_data)[-1]])

	# Save the new format because it's that much faster
	np.save(label_file, label_data, allow_pickle=False)

	label_data = np.reshape(label_data, [-1, int(np.shape(label_data)[-1]/2), 2])
	label_data_softmax = softmax(label_data)
	predictions_all = label_data_softmax[:,:,1]
	mean_pred = np.mean(predictions_all, axis=1)
	mean_pred = np.convolve(mean_pred, np.ones((46))/(46), mode='same')

	input_size = 112
	time_depth = 16
	frames = [np.zeros([input_size, input_size, 1]) for x in range(time_depth)]

	frame_num = 0
	n_models = 4
	prediction_size = 16
	while True:
		try:
			frame = np.uint8(next(im_iter))
			new_frame = np.copy(frame)
			new_frame = np.pad(new_frame, ((0,0),((1+n_models*prediction_size),0),(0,0)), 'constant', constant_values=0)

			# Plot the activations...
			predictions = predictions_all[frame_num,:]
			mean_cons = mean_pred[frame_num]
			# Black out the background behind...
			new_frame[0:(1+8*prediction_size+n_models*prediction_size),0:(1+n_models*prediction_size),:] = 0
			for i in range(int(np.shape(predictions)[0]/n_models)):
				for j in range(n_models):
					cur_result = predictions[i*n_models+j]
					new_frame[(1+prediction_size*i):(prediction_size+prediction_size*i),(1+prediction_size*j):(prediction_size+prediction_size*j),:] = np.multiply(cm.bwr(cur_result)[0:3],255)
			# Consensus prediction
			if mean_cons > 0.4811055:
				new_frame[1+8*prediction_size:(8*prediction_size+n_models*prediction_size),1:n_models*prediction_size,:] = [200,0,0]
			else:
				new_frame[1+8*prediction_size:(8*prediction_size+n_models*prediction_size),1:n_models*prediction_size,:] = [0,0,200]
			writer.append_data(new_frame.astype('u1'))
			frame_num = frame_num + 1

		except StopIteration:
			break
	reader.close()
	writer.close()



if __name__ == '__main__':
	main(sys.argv[1:])
