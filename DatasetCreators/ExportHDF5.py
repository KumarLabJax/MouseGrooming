import h5py
import sys, getopt, os, re, argparse
import imageio
import numpy as np

def read_video_data(f_pattern, mov_folder='Videos/', lab_folder='Labels/'):
	mov_file = mov_folder + f_pattern + '.avi'
	label_file = lab_folder + f_pattern + '.txt'
	try:
		mov_reader = imageio.get_reader(mov_file)
		mov_iter = mov_reader.iter_data()
		label_file = open(label_file)
	except:
		print('Failed loading video data: ' + f_pattern + '.txt')
		return None
	# Read in the data to return
	mov_data = np.array([frame for frame in mov_iter])
	mov_data = mov_data[:,:,:,0]
	lab_data = np.array([label.strip('\n').split('\t') for label in label_file.readlines()])
	mask_data = np.array([np.all(lab_data[ln,:] == lab_data[ln,:][0]) for ln in range(np.shape(lab_data)[0])])
	nlabs = np.shape(lab_data[1])[0]
	lab_data = np.uint8(lab_data[:,0])
	# Crop the data to the shortest available...
	data_dim = np.min([np.shape(mov_data)[0], np.shape(lab_data)[0]])
	mov_data = mov_data[0:data_dim,:,:]
	lab_data = lab_data[0:data_dim]
	mask_data = mask_data[0:data_dim]
	return mov_data, mask_data, lab_data, data_dim, nlabs


h5_filename = 'GroomingDataset_2017-08-21.hdf5'
h5out_file = h5py.File(h5_filename,'a',libver='latest')
train_group = h5out_file.create_group('training')

train_set = open('trainlist_2017-08-21.txt').read().splitlines()
valid_set = open('validlist_2017-08-21.txt').read().splitlines()

for vid_name in train_set:
	data = read_video_data(vid_name)
	if data is not None:
		cur_group = train_group.create_group(vid_name)
		cur_group.create_dataset('video', data=data[0])
		cur_group.create_dataset('mask', data=data[1])
		cur_group.create_dataset('label', data=data[2])
		cur_group.create_dataset('nframe', data=data[3])
		cur_group.create_dataset('nlabelers', data=data[4])

valid_group = h5out_file.create_group('validation')
for vid_name in valid_set:
	data = read_video_data(vid_name)
	if data is not None:
		cur_group = valid_group.create_group(vid_name)
		cur_group.create_dataset('video', data=data[0])
		cur_group.create_dataset('mask', data=data[1])
		cur_group.create_dataset('label', data=data[2])
		cur_group.create_dataset('nframe', data=data[3])
		cur_group.create_dataset('nlabelers', data=data[4])

h5out_file.close()
