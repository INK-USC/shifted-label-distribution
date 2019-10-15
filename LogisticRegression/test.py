import pickle
import argparse
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../data/intermediate/KBP/rm', help='directory containing train/dev/test file.')
parser.add_argument('--save_dir', type=str, default='../dumped_model', help='directory to save test results.')
parser.add_argument('--save_filename', type=str, default='result_kbp.pkl', help='filename of test results.')
args = parser.parse_args()
opt = vars(args)

noneId = utils.get_none_id(os.path.join(opt['data_dir'], 'type.txt'))

def main():
	filename = os.path.join(opt['save_dir'], opt['save_filename'])
	results = pickle.load(open(filename, 'rb'))
	true = results['gold']
	pred = results['pred']

	interested = 0.0
	predicted = 0.0
	correct = 0.0

	for i, a in enumerate(true):
		b = pred[i]
		if a != noneId:
			interested += 1
		if b != noneId:
			predicted += 1
		if a == b and a != noneId:
			correct += 1

	precision = correct / predicted
	recall = correct / interested
	f1 = (precision * recall * 2) / (precision + recall)

	print(precision, recall, f1)

if __name__ == '__main__':
	main()
