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
	parser = argparse.ArgumentParser(description='Plotting 3DConv Mean Consensus Probability')
	parser.add_argument('--mov_name', help='Name of movie to process')

	args = parser.parse_args()
	arg_dict = args.__dict__

	if 'mov_name' in arg_dict.keys() and arg_dict['mov_name'] is not None:
		video_pattern = os.path.splitext(args.mov_name)[0]

	reader = imageio.get_reader(video_pattern+'.avi')
	writer = imageio.get_writer(video_pattern+'_out_Consensus.avi', fps=30, codec='mpeg4', quality=10)
	im_iter = reader.iter_data()

	label_file = video_pattern+'_crop_meancons.npy'
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

	input_size = 112
	time_depth = 16
	frames = [np.zeros([input_size, input_size, 1]) for x in range(time_depth)]

	frame_num = 0
	while True:
		try:
			frame = np.uint8(next(im_iter))
			new_frame = np.copy(frame)

			# Consensus predictions
			prob = label_data[0,frame_num]
			new_frame[:,0:10,:] = 255
			level = 112-int(prob*112)
			if prob < 0.50000001:
				new_frame[level:111,0:10,0] = 200
				new_frame[level:111,0:10,1] = 0
				new_frame[level:111,0:10,2] = 0
			else:
				new_frame[level:111,0:10,1] = 200
				new_frame[level:111,0:10,0] = 0
				new_frame[level:111,0:10,2] = 0
			# Tick marks at 25, 50, 75%
			new_frame[int(112/2),0:5,:] = 0
			new_frame[int(3*112/4),0:5,:] = 0
			new_frame[int(112/4),0:5,:] = 0
			writer.append_data(new_frame.astype('u1'))
			frame_num = frame_num + 1

		except StopIteration:
			break
	reader.close()
	writer.close()



if __name__ == '__main__':
	main(sys.argv[1:])
