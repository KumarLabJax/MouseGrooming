import keras
from keras.models import load_model
import imageio
import os
import numpy as np
from scipy.misc import imresize
import sys, getopt, re, argparse
import tensorflow as tf
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
	parser = argparse.ArgumentParser(description='Inference 3DConv Models')
	parser.add_argument('--mov_name', help='Name of movie to process')

	args = parser.parse_args()
	arg_dict = args.__dict__

	# Helix
	net = load_model('3Dconv_Keras.h5')
	# Remove the softmax layer...
	net.pop()
	net.compile('adam','categorical_crossentropy') # rebuild to set the output correctly
	print(net.summary())

	if 'mov_name' in arg_dict.keys() and arg_dict['mov_name'] is not None:
		video_pattern = os.path.splitext(args.mov_name)[0]

	reader = imageio.get_reader(video_pattern+'.avi')
	writer = imageio.get_writer(video_pattern+'_out_keras_BN.avi', fps=30, codec='mpeg4', quality=10)
	im_iter = reader.iter_data()

	input_size = 112
	time_depth = 16
	frames = [np.zeros([input_size, input_size, 1]) for x in range(time_depth)]

	framenum = 0
	while True:
		try:
			start_time = time()
			frames[0:time_depth-1] = np.copy(frames[1:time_depth])
			frame = np.uint8(next(im_iter))
			frame = imresize(frame, (input_size, input_size, 3))
			frame = frame[:,:,0]
			frame = np.reshape(frame, [input_size, input_size, 1])
			frames[time_depth-1] = frame
			batch_input_single = np.reshape(frames,[time_depth, input_size, input_size, 1])
			batch_input = np.reshape([batch_input_single,
									np.flipud(batch_input_single),
									np.fliplr(batch_input_single),
									np.fliplr(np.flipud(batch_input_single)),
									np.transpose(batch_input_single,(0,2,1,3)),
									np.transpose(np.flipud(batch_input_single),(0,2,1,3)),
									np.transpose(np.fliplr(batch_input_single),(0,2,1,3)),
									np.transpose(np.fliplr(np.flipud(batch_input_single)),(0,2,1,3))], [8, time_depth, input_size, input_size, 1])
			if framenum % 1000 == 0:
				print('Batch ' + str(framenum))
				print('Batch Assembled in: ' + str(time()-start_time))
			start_time = time()
			results_nosoftmax = net.predict(batch_input, batch_size=8)
			if framenum % 1000 == 0:
				print('Batch Computed in: ' + str(time()-start_time))
			start_time = time()
			results = softmax(results_nosoftmax)
			#print(results)
			predictions = np.argmax(results, 1)
			#prob = np.exp(results)[:,1] / np.sum(np.exp(results), axis=1)
			new_frame = np.zeros([input_size, input_size, 3])
			new_frame[:,:,0] = np.squeeze(frame)
			new_frame[:,:,1] = np.squeeze(frame)
			new_frame[:,:,2] = np.squeeze(frame)
			new_frame[:,0:15,:] = 255
			for i in range(np.shape(results)[0]):
				if results[i][1] < 0.50000001: # Negative
					level = 112-int(results[i][1]*112)
					new_frame[level:111,(0+2*i):(1+2*i),0] = 200
					new_frame[level:111,(0+2*i):(1+2*i),1] = 0
					new_frame[level:111,(0+2*i):(1+2*i),2] = 0
				else: # Positive
					level = 112-int(results[i][1]*112)
					new_frame[level:111,(0+2*i):(1+2*i),1] = 200
					new_frame[level:111,(0+2*i):(1+2*i),0] = 0
					new_frame[level:111,(0+2*i):(1+2*i),2] = 0
			# Tick marks at 25, 50, 75%
			new_frame[int(112/2),0:5,:] = 0
			new_frame[int(3*112/4),0:5,:] = 0
			new_frame[int(112/4),0:5,:] = 0
			# Consensus predictions
			mean_pred = np.mean(results[:,1]) > 0.5
			vote_pred = np.sum(predictions) > 4 # only majority, not half
			maxpool_pred = np.argmax(np.diag(results_nosoftmax[np.argmax(results_nosoftmax,0)])) == 1
			if mean_pred:
				new_frame[1:6,20:25,1] = 200
				new_frame[1:6,20:25,0] = 0
				new_frame[1:6,20:25,2] = 0
			else:
				new_frame[1:6,20:25,0] = 200
				new_frame[1:6,20:25,1] = 0
				new_frame[1:6,20:25,2] = 0
			if vote_pred:
				new_frame[1:6,30:35,1] = 200
				new_frame[1:6,30:35,0] = 0
				new_frame[1:6,30:35,2] = 0
			else:
				new_frame[1:6,30:35,0] = 200
				new_frame[1:6,30:35,1] = 0
				new_frame[1:6,30:35,2] = 0
			if maxpool_pred:
				new_frame[1:6,40:45,1] = 200
				new_frame[1:6,40:45,0] = 0
				new_frame[1:6,40:45,2] = 0
			else:
				new_frame[1:6,40:45,0] = 200
				new_frame[1:6,40:45,1] = 0
				new_frame[1:6,40:45,2] = 0
			writer.append_data(new_frame.astype('u1'))
			if framenum % 1000 == 0:
				print('Batch Saved in: ' + str(time()-start_time))
			framenum = framenum + 1
		except StopIteration:
			break
	reader.close()
	writer.close()



if __name__ == '__main__':
	main(sys.argv[1:])
