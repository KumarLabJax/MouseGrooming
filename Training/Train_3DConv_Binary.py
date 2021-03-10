import tensorflow as tf
import imageio
import numpy as np
import queue as Queue
import random
import sys, getopt, os, re, argparse
import skvideo.io
from scipy.misc import imresize
from KerasMultiGPU import *
import h5py
from ReadHDF5 import Groom_Dataset

TFVERSION = tf.__version__.split('.')

# Other poor imports because keras changed places
if int(TFVERSION[1]) < 4 and int(TFVERSION[0]) <= 1:
	import keras
	from keras.models import Sequential, load_model
	from keras.layers.core import Activation, Dense, Flatten, Dropout
	from keras.layers.convolutional import Conv3D
	from keras.layers.pooling import MaxPooling3D
	from keras.layers.normalization import BatchNormalization
	from keras.utils import to_categorical
elif int(TFVERSION[1]) == 4:
	from tensorflow import keras
	from tensorflow.python.keras._impl.keras.models import Sequential, load_model
	from tensorflow.python.keras._impl.keras.layers.core import Activation, Dense, Flatten, Dropout
	from tensorflow.python.keras._impl.keras.layers.convolutional import Conv3D
	from tensorflow.python.keras._impl.keras.layers.pooling import MaxPooling3D
	from tensorflow.python.keras._impl.keras.layers.normalization import BatchNormalization
	from tensorflow.python.keras._impl.keras.utils import to_categorical
else:
	from tensorflow import keras
	from tensorflow.python.keras.models import Sequential, load_model
	from tensorflow.python.keras.layers.core import Activation, Dense, Flatten, Dropout
	from tensorflow.python.keras.layers.convolutional import Conv3D
	from tensorflow.python.keras.layers.pooling import MaxPooling3D
	from tensorflow.python.keras.layers.normalization import BatchNormalization
	from tensorflow.python.keras.utils import to_categorical

def main(argv):
	parser = argparse.ArgumentParser(description='Train 3DConv Models')
	parser.add_argument('--data_dir', help='Root directory of the data', default='.')
	parser.add_argument('--data_name', help='Name of the dataset', default='GroomingDataset_2017-08-21.hdf5')
	parser.add_argument('--log_dir', help='Root directory of the logging', default='.')
	parser.add_argument('--batch_size', help='Per-gpu size of batch', default=64)
	parser.add_argument('--train_set_frame_num', help='Value to downsample the training set size (-1 for no downsample)', default=-1)

	args = parser.parse_args()

	n_gpus = 2
	time_depth = 16
	input_size = 112
	n_threads = 12
	batch_size = int(int(args.batch_size)*n_gpus)
	dataset = Groom_Dataset(args.data_dir + '/' + args.data_name, time_depth, batch_size, ignore_masks=True, target_train_frames=int(args.train_set_frame_num))
	print('Labeled Videos Used:')
	np.set_printoptions(threshold=np.inf)
	print(dataset.get_train_vid_names())
	np.set_printoptions(threshold=1000) # return to default again...
	ckpts_per_epoch = 1
	train_samples = dataset.get_train_size()
	train_steps_per_epoch = int(train_samples/batch_size/ckpts_per_epoch)
	valid_samples = dataset.get_valid_size()
	valid_steps_per_epoch = int(valid_samples/batch_size/ckpts_per_epoch)
	filters = 4
	kernel_size = (3,3,3)
	dropout_rate = 0.5

	net = Sequential()
	net.add(Conv3D(filters, kernel_size, padding='same', use_bias=False, input_shape=(time_depth, input_size, input_size, 1)))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Conv3D(filters*1, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(MaxPooling3D())
	net.add(Conv3D(filters*2, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Conv3D(filters*2, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(MaxPooling3D())
	net.add(Conv3D(filters*4, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Conv3D(filters*4, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(MaxPooling3D())
	net.add(Conv3D(filters*8, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Conv3D(filters*8, kernel_size, padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(MaxPooling3D())
	net.add(Conv3D(filters*16, (1,3,3), padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Conv3D(filters*16, (1,3,3), padding='same', use_bias=False))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Flatten())
	net.add(Dropout(dropout_rate))
	net.add(Dense(128))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Dropout(dropout_rate))
	net.add(Dense(128))
	net.add(BatchNormalization())
	net.add(Activation('relu'))
	net.add(Dropout(dropout_rate))
	net.add(Dense(2))
	net.add(Activation('softmax'))

	#print(net.summary())

	log_folder = args.log_dir + '/'
	ckpt_saver = keras.callbacks.ModelCheckpoint(log_folder + '3Dconv_Keras.h5', monitor='val_acc', verbose=1, save_best_only=True)
	tensorboard_out = keras.callbacks.TensorBoard(log_dir=log_folder)
	early_stopper = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=10)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, min_lr=1e-8, verbose=1)

	adam_opt = keras.optimizers.Adam(lr=1e-5)
	#net.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])
	multigpunet = make_parallel(net, n_gpus)
	multigpunet.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])
	# Keras model is now fit for running.
	multigpunet.fit_generator(dataset.get_train_generator_parallel(n_threads), train_steps_per_epoch, epochs=200, verbose=2, validation_data=dataset.get_valid_generator_parallel(n_threads), validation_steps=valid_steps_per_epoch, initial_epoch=0, workers=1, callbacks=[ckpt_saver, tensorboard_out, early_stopper, reduce_lr])
	#net.save(log_folder + '3Dconv_Keras_singleGPU.h5')


if __name__ == '__main__':
	main(sys.argv[1:])

