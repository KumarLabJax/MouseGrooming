import h5py
import numpy as np
import argparse
import sys
import cv2

class Edit_Groom_Dataset:
	def __init__(self, h5_filename):
		# Setup the internal variables
		self.__h5_filename = h5_filename
		# Open the file
		self.__h5_file = h5py.File(self.__h5_filename,'r+')
		# Setup the training info
		self.__train_key = 'training'
		self.__train_vids = [key for key in self.__h5_file[self.__train_key].keys()]
		# And setup the validation info
		self.__valid_key = 'validation'
		self.__valid_vids = [key for key in self.__h5_file[self.__valid_key].keys()]
		# Merge them together into a full key list
		self.__all_keys = np.concatenate([[self.__train_key + '/' + item for item in self.__train_vids], [self.__valid_key + '/' + item for item in self.__valid_vids]])
	# Getter for the all available keys
	def list_keys(self):
		return self.__all_keys
	# Getter for keys that aren't currently curated
	def list_keys_not_curated(self):
		return list([[i, key] for i, key in enumerate(self.__all_keys) if not 'curated' in self.__h5_file[key].keys()])
	# Getter for keys that have a specific number of labelers
	def list_keys_num_labelers(self, nlabelers):
		return list([[i, key] for i, key in enumerate(self.__all_keys) if nlabelers == self.__h5_file[key + '/nlabelers'][...]])
	# Returns the video, mask, and label data from a given key
	def load_video(self, key):
		vid_data = self.__h5_file[key + '/video'][:,:,:]
		mask_data = self.__h5_file[key + '/mask'][:]
		label_data = self.__h5_file[key + '/label'][:]
		nlabelers = self.__h5_file[key + '/nlabelers'][...]
		return vid_data, mask_data, label_data, nlabelers
	# Updates labeling data for a video
	def update_labels(self, key, labels, mask, nlabelers):
		self.__h5_file[key + '/mask'][...] = mask
		self.__h5_file[key + '/label'][...] = labels
		self.__h5_file[key + '/nlabelers'][...] = nlabelers
		try:
			self.__h5_file[key + '/curated'][...] = True
		except KeyError:
			group = self.__h5_file[key]
			group.create_dataset('curated', data=True)
		return True

def process_hdf5(dataset):
	video_keys = dataset.list_keys()
	#print('\n'.join([str(element[0]) + ' ' + element[1] for element in enumerate(video_keys)]))
	print('Select from videos 1 through ' + str(len(video_keys)))
	while True:
		video_select = input("Select Video (h for help): ")
		process_video = False
		if video_select == 'q':
			break
		elif video_select == 'p':
			print('\n'.join([str(element[0]+1) + ' ' + element[1] for element in enumerate(video_keys)]))
		elif video_select == 'c':
			print('\n'.join([str(item[0]+1) + ' ' + item[1] for item in dataset.list_keys_not_curated()]))
			pass
		elif video_select == 'n':
			try:
				test_num_labelers = int(input('Enter the number of labelers:'))
				print('\n'.join([str(item[0]+1) + ' ' + item[1] for item in dataset.list_keys_num_labelers(test_num_labelers)]))
			except:
				print('Try using an actual number request next time.')
			pass
		elif video_select == 'h':
			print("Current options:" + "\n\th = show this menu" + "\n\tq = quit the program safely" + "\n\tp = print all the available video keys to select from" + "\n\tc = show only videos that haven't yet been curated" + "\n\tn = show only videos with a specific number of labelers")
		else:
			try:
				video_select = int(video_select)
				assert video_select > 0
				assert video_select <= len(dataset.list_keys())
				vid, mask, label, nlabelers = dataset.load_video(video_keys[video_select-1])
				print("Loaded video: " + video_keys[video_select-1])
				label, mask, nlabelers, overwrite = display_movie(vid, label, mask, nlabelers)
				if overwrite:
					dataset.update_labels(video_keys[video_select-1], label, mask, nlabelers)
					print("Saved Labels for: " + video_keys[video_select-1])
				else:
					print("Canceling Saved Labels for video.")
			except:
				print("Video not found, try again (q to quit).")
				pass

# Enables controls for viewing/editing labels on movies
def display_movie(vid, label, mask, nlabelers):
	current_frame = 0
	max_frames = np.shape(vid)[0]
	cv2.namedWindow('Frame', cv2.WINDOW_GUI_EXPANDED)
	mask_overwrite = -1
	label_overwrite = -1
	overwrite = False
	gamma = 1.0

	print('Controls:\n\t0-9 for assigning new labels\n\twasd for seeking\n\tm and n for mask assignment\n\t+- for number of labelers\n\tq or esc to quit (+save)\n\te to quit no-save')
	while True:
		# Dummy processing to overwrite labels if necessary
		current_frame = seek_and_write(current_frame, label, mask, 0, mask_overwrite, label_overwrite)
		show_frame(vid, label, mask, current_frame, nlabelers, mask_overwrite, label_overwrite, gamma)
		keystroke = cv2.waitKey(0)
		# print(str(keystroke) + ' ' + str(current_frame))
		if keystroke == 27 or keystroke == ord('q'): # Escape or 'q'
			overwrite = True
			break
		if keystroke == ord('e'): # exit without saving...
			break
		# Basic seeking
		elif keystroke == ord('a'):
			current_frame = seek_and_write(current_frame, label, mask, -1, mask_overwrite, label_overwrite)
		elif keystroke == ord('d'):
			current_frame = seek_and_write(current_frame, label, mask, 1, mask_overwrite, label_overwrite)
		elif keystroke == ord('s'):
			current_frame = seek_and_write(current_frame, label, mask, 15, mask_overwrite, label_overwrite)
		elif keystroke == ord('w'):
			current_frame = seek_and_write(current_frame, label, mask, -15, mask_overwrite, label_overwrite)
		# Modify labeler value...
		elif keystroke == ord('+'):
			nlabelers = nlabelers + 1
		elif keystroke == ord('-'):
			nlabelers = nlabelers - 1
		# Gamma correction...
		elif keystroke == ord('.'):
			gamma = gamma*1.1
		elif keystroke == ord(','):
			gamma = gamma*0.9
		elif keystroke == ord('/'):
			gamma = 1.0
		# Toggle masking value
		elif keystroke == ord('m'):
			if mask_overwrite == 1:
				mask_overwrite = -1
			else:
				mask_overwrite = 1
		elif keystroke == ord('n'):
			if mask_overwrite == 0:
				mask_overwrite = -1
			else:
				mask_overwrite = 0
		# Toggle the label values
		elif keystroke == ord('1'):
			if label_overwrite == 1:
				label_overwrite = -1
			else:
				label_overwrite = 1
		elif keystroke == ord('2'):
			if label_overwrite == 2:
				label_overwrite = -1
			else:
				label_overwrite = 2
		elif keystroke == ord('3'):
			if label_overwrite == 3:
				label_overwrite = -1
			else:
				label_overwrite = 3
		elif keystroke == ord('4'):
			if label_overwrite == 4:
				label_overwrite = -1
			else:
				label_overwrite = 4
		elif keystroke == ord('5'):
			if label_overwrite == 5:
				label_overwrite = -1
			else:
				label_overwrite = 5
		elif keystroke == ord('6'):
			if label_overwrite == 6:
				label_overwrite = -1
			else:
				label_overwrite = 6
		elif keystroke == ord('7'):
			if label_overwrite == 7:
				label_overwrite = -1
			else:
				label_overwrite = 7
		elif keystroke == ord('8'):
			if label_overwrite == 8:
				label_overwrite = -1
			else:
				label_overwrite = 8
		elif keystroke == ord('9'):
			if label_overwrite == 9:
				label_overwrite = -1
			else:
				label_overwrite = 9
		elif keystroke == ord('0'):
			if label_overwrite == 0:
				label_overwrite = -1
			else:
				label_overwrite = 0
	cv2.destroyAllWindows()
	return label, mask, nlabelers, overwrite

# Subroutine for showing frames
def show_frame(vid, label, mask, frame_num, nlabelers, mask_overwrite, label_overwrite, gamma):
	side_pad = 25
	bottom_pad = 30
	vid_height = np.shape(vid)[1]
	vid_width = np.shape(vid)[2]

	# Add in the video frame
	frame = np.zeros([vid_height + bottom_pad, vid_width + side_pad, 3], dtype=np.uint8)
	frame[0:vid_width,0:vid_height,:] = np.repeat(np.expand_dims(vid[frame_num,:,:], -1), 3, -1)
	frame = adjust_gamma(frame, gamma)

	# Add in the label data
	#colors = [[64,64,64],[255,255,255],[255,255,0],[255,0,255],[0,255,255],[0,0,255],[0,255,0],[255,0,0],[128,128,128],[255,128,255]]
	colors = [[227,206,166],[180,120,31],[138,223,178],[153,154,251],[28,26,227],[111,191,253],[0,127,255],[214,178,202],[154,61,106],[44,160,51]]
	frame = plot_side_panel(frame, side_pad, colors)
	frame = plot_bottom_panel(frame, label, mask, frame_num, bottom_pad, colors)

	# Add in extra metrics (frame #, nlabelers, total frames)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, str(frame_num), (0, 20), font, 0.75, (255,255,0), 1, cv2.LINE_AA)
	cv2.putText(frame, str(nlabelers), (0, 40), font, 0.75, (255,0,255), 1, cv2.LINE_AA)
	cv2.putText(frame, str(len(mask)-1), (0, 105), font, 0.75, (0,255,255), 1, cv2.LINE_AA)

	# Add in the mask/label pointers
	col_size = side_pad-2
	if col_size * np.shape(colors)[0] > np.shape(frame)[1]:
		col_size = int(np.shape(frame)[1]/np.shape(colors)[0])
	cv2.putText(frame, '*', (np.shape(vid)[1], int((col_size-3)*label_overwrite+col_size-1)), font, 0.65, (255,255,255), 1, cv2.LINE_AA)

	cv2.putText(frame, '*', (0, np.shape(vid)[2]+17), font, 1, (0,0,0), 2, cv2.LINE_AA)
	if mask_overwrite == 0:
		cv2.putText(frame, '*', (0, np.shape(vid)[2]+17), font, 1, (38,0,165), 1, cv2.LINE_AA)
	elif mask_overwrite == 1:
		cv2.putText(frame, '*', (0, np.shape(vid)[2]+17), font, 1, (55,104,0), 1, cv2.LINE_AA)

	cv2.imshow('Frame', frame)

# Plots the colors on the side
def plot_side_panel(frame, side_pad, colors):
	col_size = side_pad-2
	if col_size * np.shape(colors)[0] > np.shape(frame)[1]:
		col_size = int(np.shape(frame)[1]/np.shape(colors)[0])

	side = np.pad(np.reshape(np.tile(colors, [col_size*np.shape(colors)[0]]), [-1, col_size, 3]), ((1,1),(1,1),(0,0)), 'constant', constant_values=0)
	frame[0:np.shape(side)[0], np.shape(frame)[1]-np.shape(side)[1]:, :] = side
	return frame

# Plots the history of labels on the bottom
def plot_bottom_panel(frame, label, mask, frame_num, bottom_pad, colors, temporal_size=33):
	col_size = bottom_pad
	expand_row_size = int(np.shape(frame)[1]/temporal_size/2)

	# Grab the labels to plot
	temp = np.arange(frame_num-temporal_size, frame_num+temporal_size)
	temp = temp[temp>=0]
	temp = temp[temp<len(label)-1]
	# Plot the labels...
	bottom = np.pad(np.reshape(np.tile(np.array(colors)[label[temp]], [int(bottom_pad*expand_row_size/2)]), [-1, int(bottom_pad/2), 3]), ((1,1),(0,0),(0,0)), 'constant', constant_values=0)
	bottom = np.transpose(bottom, [1,0,2])
	# Plot the masks...
	bottom2 = np.pad(np.reshape(np.tile(np.array([[38,0,165],[55,104,0]])[np.array(mask[temp], dtype=int)], [int(bottom_pad*expand_row_size/2)]), [-1, int(bottom_pad/2), 3]), ((1,1),(0,0),(0,0)), 'constant', constant_values=0)
	bottom2 = np.transpose(bottom2, [1,0,2])

	# Merge
	bottom = np.concatenate([bottom2, bottom], 0)

	# Plot it centered
	center = np.where(temp == frame_num)[0][0]
	# Copy it over to the frame to display
	frame[np.shape(frame)[0]-np.shape(bottom)[0]:, int(np.shape(frame)[1]/2)-center*expand_row_size:int((np.shape(frame)[1])/2)-center*expand_row_size+np.shape(bottom)[1], :] = bottom
	# Mark the current frame...
	frame[int(np.shape(frame)[0]-(np.shape(bottom)[0])*3/4):int(np.shape(frame)[0]-(np.shape(bottom)[0])/4), int(np.shape(frame)[1]/2), :] = [0,0,0]
	frame[int(np.shape(frame)[0]-(np.shape(bottom)[0])*3/4):int(np.shape(frame)[0]-(np.shape(bottom)[0])/4), int(np.shape(frame)[1]/2)+expand_row_size, :] = [0,0,0]
	frame[int(np.shape(frame)[0]-(np.shape(bottom)[0])*3/4), int(np.shape(frame)[1]/2):int(np.shape(frame)[1]/2)+expand_row_size, :] = [0,0,0]
	frame[int(np.shape(frame)[0]-(np.shape(bottom)[0])/4), int(np.shape(frame)[1]/2):int(np.shape(frame)[1]/2)+expand_row_size, :] = [0,0,0]

	return frame

# Applies a "safe seek"
def seek_and_write(current_frame, label, mask, change, mask_overwrite, label_overwrite):
	next_frame = current_frame + change
	# Safely seek
	if current_frame < 0:
		next_frame = 0
	elif next_frame > len(label) - 2:
		next_frame = len(label) - 2

	# Edit the mask/label values (if necessary)
	if mask_overwrite >= 0:
		mask[min(current_frame,next_frame):max(current_frame,next_frame)+1] = bool(mask_overwrite)
	if label_overwrite >= 0:
		label[min(current_frame,next_frame):max(current_frame,next_frame)+1] = label_overwrite
	# return the next frame
	return next_frame

# Adjusts gamma value
def adjust_gamma(image, gamma=1.0):
	# Build LUT
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def main(argv):
	parser = argparse.ArgumentParser(description='Edits a hdf5 dataset')
	parser.add_argument('input_file', help='Name of hdf5 file to process')

	args = parser.parse_args()
	dataset = Edit_Groom_Dataset(args.input_file)

	process_hdf5(dataset)

if __name__ == '__main__':
	main(sys.argv[1:])

