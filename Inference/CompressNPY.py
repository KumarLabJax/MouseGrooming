import sys
import numpy as np
import sys, getopt, re, argparse

# Reads appended npy data
def read_data(filename):
	return_data = []
	with open(filename,'rb') as file:
		while True:
			try:
				if sys.version_info[0]==2:
					return_data.append(np.load(file))
				else:
					return_data.append(np.load(file, encoding = 'bytes', allow_pickle = False))
			except (IOError, ValueError):
				break
	return_data = np.reshape(return_data, [-1, np.shape(return_data)[-1]])
	# Overwrite because the data when appended at initial write-time is super slow to process.
	# This makes the second time you read in the data fast
	np.save(filename, return_data, allow_pickle=False)
	return return_data


def main(argv):
	parser = argparse.ArgumentParser(description='Reshape appended npy to a better compressed (and faster loading) npy')
	parser.add_argument('npy', help='NPY filename to re-export')

	args = parser.parse_args()

	read_data(args.npy)

if __name__ == '__main__':
	main(sys.argv[1:])
