import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='../dumped_model', help='directory to save test results.')
parser.add_argument('--save_filename', type=str, default='result_kbp.pkl', help='filename of test results.')
args = parser.parse_args()
opt = vars(args)

noneId = 0

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
