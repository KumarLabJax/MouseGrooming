import h5py
import numpy as np
import multiprocessing as mp

# Usage:
# Create a Groom_Dataset object.
# Use the get_train_generator() and get_valid_generator() functions to read the data.
class Groom_Dataset:
	def __init__(self, h5_filename, time_size, batch_size, n_classes=2, min_labels=1, ignore_masks=False, balance_classes=False, curate_only=False, class_list=None, target_train_frames=-1, target_valid_frames=-1):
		# Setup the internal variables
		self.__h5_filename = h5_filename
		self.time_size = time_size
		self.batch_size = batch_size
		self.__n_classes = n_classes
		self.__train_queue = None
		self.__train_mp_pool = None
		self.__valid_queue = None
		self.__valid_mp_pool = None
		# Setup class re-mapping
		self.__class_map = {}
		if class_list is None:
			class_list = [[x] for x in range(n_classes)]
		else:
			# Override if class_list defined
			self.__n_classes = len(class_list)
		temp = list(dict((item,i) for item in e) for i,e in enumerate(class_list))
		for i in temp:
			self.__class_map.update(i)
		# Open the file
		h5_file = h5py.File(self.__h5_filename, 'r')
		# Setup the training info
		self.__train_key = 'training'
		self.__train_vids = [key for key in h5_file[self.__train_key].keys()]
		# And setup the validation info
		self.__valid_key = 'validation'
		self.__valid_vids = [key for key in h5_file[self.__valid_key].keys()]
		# Split out method of setting up info based on number of labeles required...
		if min_labels > 1:
			self.__train_vids = self.__set_min_labels(self.__train_key, self.__train_vids, min_labels)
			self.__valid_vids = self.__set_min_labels(self.__valid_key, self.__valid_vids, min_labels)
		# Split out for curated data
		if curate_only:
			self.__train_vids = self.__set_curate_only(self.__train_key, self.__train_vids)
			self.__valid_vids = self.__set_curate_only(self.__valid_key, self.__valid_vids)
		# Prime the counts
		self.__train_inds, self.__train_sample_inds, self.__train_sample_prob = self.__set_visible_frames(self.__train_key, self.__train_vids, ignore_masks, balance_classes)
		self.__valid_inds, self.__valid_sample_inds, self.__valid_sample_prob = self.__set_visible_frames(self.__valid_key, self.__valid_vids, ignore_masks, balance_classes)
		# Trim down the videos until it's less than the target frames
		if target_train_frames > 0:
			vid_lengths = [int(h5_file[self.__train_key + '/' + vid + '/nframe'][...]) for vid in self.__train_vids]
			while target_train_frames < np.sum(vid_lengths):
				to_remove = np.random.randint(len(vid_lengths))
				vid_lengths = np.delete(vid_lengths, to_remove)
				self.__train_vids = list(np.delete(self.__train_vids, to_remove))
			# Reset the video calculations
			self.__train_inds, self.__train_sample_inds, self.__train_sample_prob = self.__set_visible_frames(self.__train_key, self.__train_vids, ignore_masks, balance_classes)
		if target_valid_frames > 0:
			vid_lengths = [int(h5_file[self.__valid_key + '/' + vid + '/nframe'][...]) for vid in self.__valid_vids]
			while target_valid_frames < np.sum(vid_lengths):
				to_remove = np.random.randint(len(vid_lengths))
				vid_lengths = np.delete(vid_lengths, to_remove)
				self.__valid_vids = list(np.delete(self.__valid_vids, to_remove))
			# Reset the video calculations
			self.__valid_inds, self.__valid_sample_inds, self.__valid_sample_prob = self.__set_visible_frames(self.__valid_key, self.__valid_vids, ignore_masks, balance_classes)
		h5_file.close()
	# Assigns the visible items for sampling
	# Note: key and vids must match which they refer to
	# Returns:
	#    inds: vector of length (number of frame) that contains the video index for the sample
	#    sample_inds: vector of length (number of frames) that contains a sample index
	#    sample_prob: vector same size as sample_inds with the probability for selecting that sample
	###### Note: sample_inds contains gaps when masking is used.
	def __set_visible_frames(self, key, vids, ignore_masks, balance_classes):
		h5_file = h5py.File(self.__h5_filename, 'r')
		vid_lengths = [int(h5_file[key + '/' + vid + '/nframe'][...]) for vid in vids]
		inds = np.cumsum(vid_lengths)
		sample_inds = np.arange(0, vid_lengths[-1])
		video_inds_expanded = np.concatenate([np.repeat(i, vid_length) for i,vid_length in enumerate(vid_lengths)])
		# Optimize reading to drop masked labels...
		if ignore_masks:
			all_masks = np.concatenate([h5_file[key + '/' + vid + '/mask'][...] for vid in vids])
			# Alternative Form...
			#sample_inds = sample_inds[np.where(all_masks == True)]
			sample_inds = np.reshape(np.where(all_masks == True), [-1])
		# Transform the data into more convenient stuff...
		video_inds_expanded = video_inds_expanded[sample_inds]
		sample_inds = sample_inds - inds[video_inds_expanded] + np.array(vid_lengths)[video_inds_expanded]
		# Set the probabilities of selecting an example
		if not balance_classes:
			# Give equal probability (uniform distribution) to all samples
			sample_prob = np.ones(len(sample_inds))/len(sample_inds)
		else: # balance all the classes
			all_classes = [h5_file[key + '/' + str(vids[vid]) + '/label'][frame] for vid,frame in zip(video_inds_expanded, sample_inds)]
			# Remap
			all_classes = [self.__class_map[int(x)] for x in all_classes]
			class_percentage = np.unique(all_classes, return_counts = True)[1]
			# Set probability of example equal to 1/class_examples/num_classes (ie equal chance per class and equal chance of example inside each class)
			sample_prob = 1/class_percentage[all_classes]/self.__n_classes
		h5_file.close()
		return video_inds_expanded, sample_inds, sample_prob
	# Removes keys based on number of labels
	# Note: key and vids must match which they refer to
	def __set_min_labels(self, key, vids, min_labels):
		h5_file = h5py.File(self.__h5_filename, 'r')
		# Filter out the videos with < min_labels present
		n_labels_vids = [int(h5_file[key + '/' + vid + '/nlabelers'][...]) for vid in vids]
		h5_file.close()
		return np.array(vids)[np.where(np.array(n_labels_vids) >= min_labels)[0]]
	# Removes keys based on curated flag
	# Note: key and vids must match which they refer to
	def __set_curate_only(self, key, vids):
		h5_file = h5py.File(self.__h5_filename, 'r')
		# Search for 'curated' flag existing first (default to no)
		vids = [vid for vid in vids if 'curated' in h5_file[key + '/' + vid].keys()]
		# Make sure the flag is actually True
		curated_vids = [int(h5_file[key + '/' + vid + '/curated'][...]) for vid in vids]
		h5_file.close()
		return np.array(vids)[np.where(np.array(curated_vids) == True)[0]]
	# Reads chunks of data
	def __retrieve_data(self, data_path, data_chunk):
		#print("Reading in data: " + data_path + " for frame " + str(data_chunk[-1]))
		temp_h5file = h5py.File(self.__h5_filename, 'r')
		vid_data = temp_h5file[data_path + '/video'][data_chunk,:,:]
		mask_data = temp_h5file[data_path + '/mask'][data_chunk]
		label_data = temp_h5file[data_path + '/label'][data_chunk]
		temp_h5file.close()
		# Remap the data labels
		label_data = [self.__class_map[int(x)] for x in label_data]
		#print("Read in data: " + data_path + " for frame " + str(data_chunk[-1]))
		return vid_data, mask_data, label_data
	# Fetches 1 example from the training set
	def __get_train_example(self):
		rand_flip = np.random.random_integers(0, 7)
		# Pick a sample
		sample = np.random.choice(np.arange(0,len(self.__train_sample_inds)), 1, p=self.__train_sample_prob)[0]
		# Grab the video index and the frame number
		vid_ind = self.__train_inds[sample]
		sample = self.__train_sample_inds[sample]
		data_min = sample - self.time_size
		data_path = self.__train_key + '/' + str(self.__train_vids[vid_ind])
		if data_min < 0:
			raw_data = self.__retrieve_data(data_path, range(0, sample+1))
			frames = np.pad(raw_data[0], [(np.abs(data_min+1),0),(0,0),(0,0)], mode='constant')
		else:
			raw_data = self.__retrieve_data(data_path, range(data_min+1, sample+1))
			frames = raw_data[0]
		# Random flips
		if rand_flip % 2 == 1:
			frames = frames[:,::-1,:]
		if (rand_flip/2) % 2 == 1:
			frames = frames[:,:,::-1]
		if (rand_flip/4) % 2 == 1:
			frames = np.transpose(frames, (0,2,1))
		return np.expand_dims(frames, axis=-1), raw_data[1][-1], raw_data[2][-1]
	# Fetches 1 example from the validation set
	def __get_valid_example(self):
		rand_flip = np.random.random_integers(0, 7)
		# Pick a sample
		sample = np.random.choice(np.arange(0,len(self.__valid_sample_inds)), 1, p=self.__valid_sample_prob)[0]
		# Grab the video index and the frame number
		vid_ind = self.__valid_inds[sample]
		sample = self.__valid_sample_inds[sample]
		data_min = sample - self.time_size
		data_path = self.__valid_key + '/' + str(self.__valid_vids[vid_ind])
		if data_min < 0:
			raw_data = self.__retrieve_data(data_path, range(0, sample+1))
			frames = np.pad(raw_data[0], [(np.abs(data_min+1),0),(0,0),(0,0)], mode='constant')
		else:
			raw_data = self.__retrieve_data(data_path, range(data_min+1, sample+1))
			frames = raw_data[0]
		# Random flips
		if rand_flip % 2 == 1:
			frames = frames[:,::-1,:]
		if (rand_flip/2) % 2 == 1:
			frames = frames[:,:,::-1]
		if (rand_flip/4) % 2 == 1:
			frames = np.transpose(frames, (0,2,1))
		return np.expand_dims(frames, axis=-1), raw_data[1][-1], raw_data[2][-1]
	# Fetches a batch of training data
	def get_train_generator(self):
		while True:
			raw_batch = [self.__get_train_example() for x in range(self.batch_size)]
			frames = np.stack([x[0] for x in raw_batch])
			masks = np.stack([x[1] for x in raw_batch])
			labels = np.stack([x[2] for x in raw_batch])
			yield frames, np.eye(self.__n_classes)[labels], masks
	# Fetches a batch of validation data
	def get_valid_generator(self):
		while True:
			raw_batch = [self.__get_valid_example() for x in range(self.batch_size)]
			frames = np.stack([x[0] for x in raw_batch])
			masks = np.stack([x[1] for x in raw_batch])
			labels = np.stack([x[2] for x in raw_batch])
			yield frames, np.eye(self.__n_classes)[labels], masks
	# Fetches 1 example from the training set and places it into the queue
	def __get_train_example_queue(self, train_queue):
		# Be sure to seed each thread correctly...
		import random
		seed = random.randrange(4294967295)
		np.random.seed(seed=seed)
		while True:
			try:
				rand_flip = np.random.random_integers(0, 7)
				# Pick a sample
				sample = np.random.choice(np.arange(0,len(self.__train_sample_inds)), 1, p=self.__train_sample_prob)[0]
				# Grab the video index and the frame number
				vid_ind = self.__train_inds[sample]
				sample = self.__train_sample_inds[sample]
				data_min = sample - self.time_size
				data_path = self.__train_key + '/' + str(self.__train_vids[vid_ind])
				#print("Attempting to read in data: " + data_path + " with video index " + str(vid_ind) + " and frame " + str(sample))
				if data_min < 0:
					raw_data = self.__retrieve_data(data_path, range(0, sample+1))
					frames = np.pad(raw_data[0], [(np.abs(data_min+1),0),(0,0),(0,0)], mode='constant')
				else:
					raw_data = self.__retrieve_data(data_path, range(data_min+1, sample+1))
					frames = raw_data[0]
				# Random flips
				if rand_flip % 2 == 1:
					frames = frames[:,::-1,:]
				if (rand_flip/2) % 2 == 1:
					frames = frames[:,:,::-1]
				if (rand_flip/4) % 2 == 1:
					frames = np.transpose(frames, (0,2,1))
				train_queue.put([np.expand_dims(frames, axis=-1), raw_data[1][-1], raw_data[2][-1]])
			# Odd exceptions that get thrown during race condition?
			except Exception as e:
				print('An error occurred: ' + str(type(e)) + " : " + str(e))
				pass
	# Fetches 1 example from the validation set and places it into the queue
	def __get_valid_example_queue(self, valid_queue):
		# Be sure to seed each thread correctly...
		import random
		seed = random.randrange(4294967295)
		np.random.seed(seed=seed)
		while True:
			try:
				rand_flip = np.random.random_integers(0, 7)
				# Pick a sample
				sample = np.random.choice(np.arange(0,len(self.__valid_sample_inds)), 1, p=self.__valid_sample_prob)[0]
				# Grab the video index and the frame number
				vid_ind = self.__valid_inds[sample]
				sample = self.__valid_sample_inds[sample]
				data_min = sample - self.time_size
				data_path = self.__valid_key + '/' + str(self.__valid_vids[vid_ind])
				if data_min < 0:
					raw_data = self.__retrieve_data(data_path, range(0, sample+1))
					frames = np.pad(raw_data[0], [(np.abs(data_min+1),0),(0,0),(0,0)], mode='constant')
				else:
					raw_data = self.__retrieve_data(data_path, range(data_min+1, sample+1))
					frames = raw_data[0]
				# Random flips
				if rand_flip % 2 == 1:
					frames = frames[:,::-1,:]
				if (rand_flip/2) % 2 == 1:
					frames = frames[:,:,::-1]
				if (rand_flip/4) % 2 == 1:
					frames = np.transpose(frames, (0,2,1))
				valid_queue.put([np.expand_dims(frames, axis=-1), raw_data[1][-1], raw_data[2][-1]])
			# Odd exceptions that get thrown during race condition?
			except Exception as e:
				print('An error occurred: ' + str(type(e)) + " : " + str(e))
				pass
	# Fetches a parallel enqueue-ing generator
	def get_train_generator_parallel(self, n_threads):
		# Queue size to be batch_size*15
		if self.__train_mp_pool is None:
			#self.__train_queue = mp.Manager().Queue(self.batch_size*15)
			self.__train_queue = mp.Queue(self.batch_size*15)
			#self.__train_mp_pool = mp.Pool(n_threads)
			self.__train_mp_pool = [mp.Process(target=self.__get_train_example_queue, args=(self.__train_queue,), daemon=True) for i in range(n_threads)]
			#self.__train_mp_pool.map(self.__get_train_example_queue, range(n_threads))
			for pool_index in self.__train_mp_pool:
				pool_index.start()
		while True:
			raw_batch = [self.__train_queue.get() for x in range(self.batch_size)]
			frames = np.stack([x[0] for x in raw_batch])
			masks = np.stack([x[1] for x in raw_batch])
			labels = np.stack([x[2] for x in raw_batch])
			yield frames, np.eye(self.__n_classes)[labels], masks
	# Fetches a parallel enqueue-ing generator
	def get_valid_generator_parallel(self, n_threads):
		# Queue size to be batch_size*15
		if self.__valid_mp_pool is None:
			self.__valid_queue = mp.Queue(self.batch_size*15)
			self.__valid_mp_pool = [mp.Process(target=self.__get_valid_example_queue, args=(self.__valid_queue,), daemon=True) for i in range(n_threads)]
			for pool_index in self.__valid_mp_pool:
				pool_index.start()
		while True:
			raw_batch = [self.__valid_queue.get() for x in range(self.batch_size)]
			frames = np.stack([x[0] for x in raw_batch])
			masks = np.stack([x[1] for x in raw_batch])
			labels = np.stack([x[2] for x in raw_batch])
			yield frames, np.eye(self.__n_classes)[labels], masks
	# Returns a generator for the full training set
	# One at a time
	# Throws "StopIteration" when the dataset is extinguished
	def get_full_train_generator(self):
		for example_num in np.arange(0,len(self.__train_sample_inds)):
			# Grab the video index and the frame number
			vid_ind = self.__train_inds[example_num]
			sample = self.__train_sample_inds[example_num]
			data_min = sample - self.time_size
			data_path = self.__train_key + '/' + str(self.__train_vids[vid_ind])
			if data_min < 0:
				raw_data = self.__retrieve_data(data_path, range(0, sample+1))
				frames = np.pad(raw_data[0], [(np.abs(data_min+1),0),(0,0),(0,0)], mode='constant')
			else:
				raw_data = self.__retrieve_data(data_path, range(data_min+1, sample+1))
				frames = raw_data[0]
			yield np.expand_dims(frames, axis=-1), raw_data[1][-1], raw_data[2][-1]
	# Returns a generator for the full validation set
	# One at a time
	# Throws "StopIteration" when the dataset is extinguished
	def get_full_valid_generator(self):
		for example_num in np.arange(0,len(self.__valid_sample_inds)):
			# Grab the video index and the frame number
			vid_ind = self.__valid_inds[example_num]
			sample = self.__valid_sample_inds[example_num]
			data_min = sample - self.time_size
			data_path = self.__valid_key + '/' + str(self.__valid_vids[vid_ind])
			if data_min < 0:
				raw_data = self.__retrieve_data(data_path, range(0, sample+1))
				frames = np.pad(raw_data[0], [(np.abs(data_min+1),0),(0,0),(0,0)], mode='constant')
			else:
				raw_data = self.__retrieve_data(data_path, range(data_min+1, sample+1))
				frames = raw_data[0]
			yield np.expand_dims(frames, axis=-1), raw_data[1][-1], raw_data[2][-1]
	# Returns the detected number of examples in the training set size
	def get_train_size(self):
		return len(self.__train_sample_inds)
	# Returns the detected number of examples in the validation set size
	def get_valid_size(self):
		return len(self.__valid_sample_inds)
	# Helper for retrieving distributions
	def __get_distribution(self, key):
		h5_file = h5py.File(self.__h5_filename, 'r')
		if key == self.__train_key:
			video_names = self.__train_vids
			video_set = self.__train_inds
			frame_set = self.__train_sample_inds
		else:
			video_names = self.__valid_vids
			video_set = self.__valid_inds
			frame_set = self.__valid_sample_inds
		all_labels = [h5_file[key + '/' + str(video_names[vid]) + '/label'][frame] for vid,frame in zip(video_set, frame_set)]
		all_labels = [self.__class_map[int(x)] for x in all_labels]
		h5_file.close()
		return np.unique(all_labels, return_counts = True)
	# Returns the distribution of labels in the training set
	def get_train_distribution(self):
		return self.__get_distribution(self.__train_key)
	# Returns the distribution of labels in the validation set
	def get_valid_distribution(self):
		return self.__get_distribution(self.__valid_key)
	# Returns the number of videos in the train set
	def get_train_numvids(self):
		return len(self.__train_vids)
	# Returns the number of videos in the validation set
	def get_valid_numvids(self):
		return len(self.__valid_vids)
	# Returns the available videos in the training set
	def get_train_vid_names(self):
		return self.__train_vids
	# Returns the available videos in the validation set
	def get_valid_vid_names(self):
		return self.__valid_vids
