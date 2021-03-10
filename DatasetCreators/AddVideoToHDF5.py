import h5py
import sys, getopt, os, re, argparse
import imageio
import numpy as np

def read_default_video_data(f_pattern):
	mov_file = f_pattern
	try:
		mov_reader = imageio.get_reader(mov_file)
		mov_iter = mov_reader.iter_data()
	except:
		print('Failed loading video data: ' + f_pattern)
		return None
	# Read in the data to return
	mov_data = np.array([frame for frame in mov_iter])
	mov_data = mov_data[:,:,:,0]
	# Crop the data to the shortest available...
	data_dim = np.min([np.shape(mov_data)[0]])
	mov_data = mov_data[0:data_dim,:,:]
	lab_data = np.zeros([data_dim]).astype(np.uint8)
	mask_data = np.zeros([data_dim]).astype(bool)
	nlabs = 0
	return mov_data, mask_data, lab_data, data_dim, nlabs

def main(argv):
	parser = argparse.ArgumentParser(description='Adds a video to a hdf5 grooming dataset file')
	parser.add_argument('--mov_name', help='Name of movie to add', required=True)
	parser.add_argument('--video_key', help='Key to assign to the video', default=None)
	parser.add_argument('--input_hdf5', help='HDF5 file to append the video data to', required=True)
	parser.add_argument('--set', help='Training/Validation set', choices=['training','validation'], default='training')

	args = parser.parse_args()

	h5out_file = h5py.File(args.input_hdf5,'a',libver='latest')
	try:
		add_to_group = h5out_file[args.set]
	except:
		add_to_group = h5out_file.create_group(args.set)

	if args.video_key is None:
		vid_name = os.path.splitext(os.path.split(args.mov_name)[-1])[0]
	else:
		vid_name = args.video_key

	data = read_default_video_data(args.mov_name)
	if data is not None:
		try:
			cur_group = add_to_group.create_group(vid_name)
		except:
			print('Video already exists in this dataset, cannot add. Exiting.')
			sys.exit()
		cur_group.create_dataset('video', data=data[0])
		cur_group.create_dataset('mask', data=data[1])
		cur_group.create_dataset('label', data=data[2])
		cur_group.create_dataset('nframe', data=data[3])
		cur_group.create_dataset('nlabelers', data=data[4])
	h5out_file.close()

if __name__ == '__main__':
	main(sys.argv[1:])
