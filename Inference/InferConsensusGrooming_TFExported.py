# Note: The exported model is generated from the ExportTFGraph.py function in this folder.

import imageio
import os
import numpy as np
from scipy.misc import imresize
import sys, getopt, re, argparse
import tensorflow as tf
import matplotlib.cm as cm
from time import time
from CompressNPY import read_data
import cv2

# Keras' definition converted to numpy...
def softmax(x, axis=-1):
	ndim = np.ndim(x)
	if ndim >= 2:
		e = np.exp(x - np.max(x, axis=axis, keepdims=True))
		s = np.sum(e, axis=axis, keepdims=True)
		return e / s
	else:
		raise ValueError('Cannot apply softmax to a tensor that is 1D')

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
def process_video_frames(net, im_iter, video_pattern):
	with tf.Session() as sess:
		loader = tf.train.import_meta_graph(net + '.meta')
		sess.run(tf.global_variables_initializer())
		loader = loader.restore(sess, net)
		batch_placeholder = tf.get_default_graph().get_tensor_by_name('input_1:0')
		groom_classification = tf.get_default_graph().get_tensor_by_name('reshape_1/Reshape:0') # ?x4x2 -- batch = flips, 4 models, 2 softmax dimension

		file_raw = open(video_pattern + '_raw.npy', 'ab')
		file_consensus = open(video_pattern + '_meancons.npy', 'ab')

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
				batch_input = flip_input_batch(batch_input_single)
				# Time logging...
				if framenum % 1000 == 0:
					print('Batch ' + str(framenum))
					print('Batch Assembled in: ' + str(time()-start_time))
				start_time = time()

				# Run the prediction
				results_nosoftmax = sess.run([groom_classification], {batch_placeholder: batch_input})
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
				# mean_pred = np.mean(results[:,1]) > 0.5
				# vote_pred = np.sum(predictions) > int(np.shape(results)[0]/2) # only majority, not half
				# maxpool_pred = np.argmax(np.diag(results_nosoftmax[np.argmax(results_nosoftmax,0)])) == 1
				# Just incase we get a better post-processing than mean_pred...
				raw_out = np.reshape(results_nosoftmax, -1)

				np.save(file_raw, raw_out, allow_pickle=False)
				np.save(file_consensus, mean_pred, allow_pickle=False)

				# Time logging...
				if framenum % 1000 == 0:
					print('Batch Saved in: ' + str(time()-start_time))
				framenum = framenum + 1
			except StopIteration:
				break

		file_raw.close()
		file_consensus.close()

# Wrapper for cropped video processing
def process_cropped_movie(net, video_pattern):
	reader = imageio.get_reader(video_pattern+'.avi')
	im_iter = reader.iter_data()
	process_video_frames(net, im_iter, video_pattern)
	reader.close()

# Wrapper for not-cropped video processing
def process_full_movie(net, video_pattern, ellfit_extension):
	reader = imageio.get_reader(video_pattern+'.avi')
	im_iter = reader.iter_data()
	track_data = read_data(video_pattern + ellfit_extension)
	frame_iter = crop_frame(im_iter, track_data)
	process_video_frames(net, frame_iter, video_pattern)
	reader.close()

# Applies a crop based on center location in tracking data
def crop_frame(im_iter, track_data):
	track_iter = np.nditer(track_data)
	while True:
		frame = next(im_iter)
		ell_data = np.array([next(track_iter) for x in range(6)])
		# Apply the crop
		affine_mat = np.float32([[1,0,-ell_data[0]+112/2],[0,1,-ell_data[1]+112/2]])
		crop_frame = cv2.warpAffine(frame, affine_mat, (112, 112));
		yield crop_frame

def main(argv):
	parser = argparse.ArgumentParser(description='Inference 3DConv Models')
	parser.add_argument('--mov_name', help='Name of movie to process')
	parser.add_argument('--mov_list', help='File containing a list of movies to process')
	parser.add_argument('--fullframe_video', help='Video is full-frame (not cropped)', dest='video_cropped', action='store_false', default=True)
	parser.add_argument('--ellfit_extension', help='Ellipse-fit data extension', default='_ellfit.npy')
	parser.add_argument('--network', help='Networks to use during inference', default='exported/KerasConsensusModel')

	args = parser.parse_args()
	arg_dict = args.__dict__

	if 'mov_name' in arg_dict.keys() and arg_dict['mov_name'] is not None:
		video_pattern = os.path.splitext(args.mov_name)[0]
		if args.video_cropped:
			process_cropped_movie(args.network, video_pattern)
		else:
			process_full_movie(args.network, video_pattern, args.ellfit_extension)
	elif 'mov_list' in arg_dict.keys() and arg_dict['mov_list'] is not None:
		f = open(args.mov_list, 'r')
		lines = f.read().split('\n')
		lines = lines[0:-1] # Remove the last split '' string
		f.close()
		list_of_vids = [os.path.splitext(line)[0] for line in lines]
		for video_pattern in list_of_vids:
			if args.video_cropped:
				process_cropped_movie(args.network, video_pattern)
			else:
				process_full_movie(args.network, video_pattern, args.ellfit_extension)

if __name__ == '__main__':
	main(sys.argv[1:])

