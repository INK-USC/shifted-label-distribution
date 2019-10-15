import os

def load_data(dir, set):
	'''
	if set=train, expect train_x.txt and train_y.txt
	'''

	x_file = open(os.path.join(dir, set + '_x.txt'), 'r')
	y_file = open(os.path.join(dir, set + '_y.txt'), 'r')

	n = len(y_file.readlines())
	y_file = open(os.path.join(dir, set + '_y.txt'), 'r')

	x = []
	y = []

	for i in range(n-1):
		x_line = x_file.readline().strip().split('\t')
		x_id = x_line[0]
		x_features = map(int, x_line[1].split(','))
		# x_features = x_file.readline().strip().split(',')
		y_line = y_file.readline().strip().split('\t')
		if len(y_line) > 1:
			y_id = y_line[0]
			y_label = map(int, y_line[1].split(','))

			# print x_id, x_features, y_line, y_id, y_label
			assert(x_id == y_id)
			x.append(x_features)
			y.append(y_label)

	return x, y

def load_info(dir):
	feature_file = open(os.path.join(dir, 'feature.map'), 'r')
	feature_size = len(feature_file.readlines())

	type_file = open(os.path.join(dir, 'type.txt'), 'r')
	type_size = len(type_file.readlines())

	return feature_size, type_size

def get_none_id(type_filename):
	with open(type_filename) as type_file:
		for line in type_file:
			ls = line.strip().split()
			if ls[0] == "None":
				return int(ls[1])