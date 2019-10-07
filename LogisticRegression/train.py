import pickle
import os
import utils
from Logistic import Logistic
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/intermediate/KBP/rm', help='directory containing train/dev/test file.')
parser.add_argument('--save_dir', type=str, default='../dumped_model', help='directory to save test results.')
parser.add_argument('--save_filename', type=str, default='result_kbp.pkl', help='filename of test results.')
parser.add_argument('--default_thres', type=float, default=0.5, help='filename of test results.')
args = parser.parse_args()
opt = vars(args)

data_dir = opt['data_dir']

def main():
	x_train, y_train = utils.load_data(data_dir, 'train')
	x_dev, y_dev = utils.load_data(data_dir, 'dev')
	x_test, y_test = utils.load_data(data_dir, 'test')

	feature_size, label_size = utils.load_info(data_dir)

	model = Logistic(feature_size, label_size, opt['default_thres'])

	# no hyper-prams needed so combining train and dev to train the model.
	# x_train += x_dev
	# y_train += y_dev
	model.fit(x_train, y_train)

	# test set prediction
	y_pred = map(lambda x: int(model.predict(x)[0]), x_test)
	y_test = map(lambda x: x[0], y_test)

	results = {
		'pred': y_pred,
		'gold': y_test
	}

	filename = os.path.join(opt['save_dir'], opt['save_filename'])
	pickle.dump(results, open(filename, 'wb'))

if __name__ == '__main__':
	main()
