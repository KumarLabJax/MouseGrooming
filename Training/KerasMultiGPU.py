import tensorflow as tf

TFVERSION = tf.__version__.split('.')

if int(TFVERSION[1]) < 4 and int(TFVERSION[0]) <= 1:
	from keras.layers import merge
	from keras.layers.core import Lambda
	from keras.models import Model
elif int(TFVERSION[1]) == 4:
	from tensorflow.python.keras._impl.keras.layers import merge
	from tensorflow.python.keras._impl.keras.layers.core import Lambda
	from tensorflow.python.keras._impl.keras.models import Model
else:
	from tensorflow.python.keras.layers import merge
	from tensorflow.python.keras.layers.core import Lambda
	from tensorflow.python.keras.models import Model

def make_parallel(model, gpu_count):
	def get_slice(data, idx, parts):
		shape = tf.shape(data)
		size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
		stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
		start = stride * idx
		return tf.slice(data, start, size)
	outputs_all = []
	for i in range(len(model.outputs)):
		outputs_all.append([])
	#Place a copy of the model on each GPU, each getting a slice of the batch
	for i in range(gpu_count):
		with tf.device('/gpu:%d' % i):
			with tf.name_scope('tower_%d' % i) as scope:
				inputs = []
				#Slice each input into a piece for processing on this GPU
				for x in model.inputs:
					input_shape = tuple(x.get_shape().as_list())[1:]
					if int(TFVERSION[1]) < 4 and int(TFVERSION[0]) <= 1:
						slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
					else:
						slice_n = Lambda(get_slice, arguments={'idx':i,'parts':gpu_count})(x)
					inputs.append(slice_n)
				outputs = model(inputs)
				if not isinstance(outputs, list):
					outputs = [outputs]
				#Save all the outputs for merging back together later
				for l in range(len(outputs)):
					outputs_all[l].append(outputs[l])
	# merge outputs on CPU
	with tf.device('/cpu:0'):
		merged = []
		for outputs in outputs_all:
			if int(TFVERSION[1]) < 4 and int(TFVERSION[0]) <= 1:
				merged.append(merge(outputs, mode='concat', concat_axis=0))
				returnVal = Model(input=model.inputs, output=merged)
			else:
				merged.append(merge.concatenate(outputs, axis=0))
				returnVal = Model(model.inputs, merged)
		return returnVal

