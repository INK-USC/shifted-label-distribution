'''
Training script for Position-Aware LSTM for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
from model import Model
import utils
from utils import Dataset
import argparse
import pickle
import numpy as np
import os
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./dumped_models', help='Root dir for saving models.')
parser.add_argument('--info', type=str, default='KBP_default', help='Optional info for the experiment.')
parser.add_argument('--repeat', type=int, default=5, help='Train the model for multiple times.')
# if info == 'KBP_default' and repeat == 5, we will evaluate 5 models 'KBP_default_1' ... 'KBP_default_5'

parser.add_argument('--thres_ratio', type=float, default=0.2, help='proportion of data to tune thres.')
parser.add_argument('--bias_ratio', type=float, default=0.2, help='proportion of data to estimate bias.')
parser.add_argument('--cvnum', type=int, default=100, help='# samples to tune thres or estimate bias')

parser.add_argument('--fix_bias', dest='fix_bias', action='store_true', help='Train model with fix bias.')
parser.set_defaults(fix_bias=False)

args_new = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the config of trained model
model_file = os.path.join(args_new.save_dir, args_new.info)
params = torch.load(model_file + '_1.pkl')
args = params['config']
print(args)

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

with open(args.vocab_dir + '/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
word2id = {}
for idx, word in enumerate(vocab):
	word2id[word] = idx

emb_file = args.vocab_dir + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == len(vocab)
assert emb_matrix.shape[1] == args.emb_dim
args.vocab_size = len(vocab)


rel2id = utils.load_rel2id('%s/relation2id.json' % args.data_dir)
none_id = rel2id['no_relation']
print('Reading data......')
train_filename = '%s/train.json' % args.data_dir
test_filename = '%s/test.json' % args.data_dir

train_dset = Dataset(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True,
					 mask_with_type=args.mask_with_type)
test_dset = Dataset(test_filename, args, word2id, device, rel2id=rel2id)
train_lp = torch.from_numpy(train_dset.log_prior).to(device)

for runid in range(1, args.repeat + 1):
	model = Model(args, device, word_emb=emb_matrix, rel2id=rel2id)
	print('loading model %d ......' % runid)
	model.load('%s_%d.pkl' % (model_file, runid))

	if not args_new.fix_bias:
		print('Evaluating original / max_thres / ent_thres / set_bias.')

		# Original
		test_loss, (prec, recall, f1) = model.eval(test_dset)
		print('Original:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))

		# Max Thres
		test_loss, f1, recall, prec = model.TuneMaxThres(test_dset, none_id,
																  ratio=args_new.thres_ratio,
																  cvnum=args_new.cvnum)
		print('Max Thres:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))

		# Entropy Thres
		test_loss, f1, recall, prec = model.TuneEntropyThres(test_dset, none_id,
																	  ratio=args_new.thres_ratio,
																	  cvnum=args_new.cvnum)
		print('Entropy Thres:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))


		# Set bias
		results = []
		for j in tqdm(range(args_new.cvnum)):
			# splitting test set into clean dev and actual test
			cdev_dset, test_dset = utils.get_cv_dataset(test_filename, args, word2id, device, rel2id,
													   dev_ratio=args_new.thres_ratio)
			cdev_lp = torch.from_numpy(cdev_dset.log_prior).to(device)
			bias_old = model.get_bias()
			bias_new = bias_old - train_lp + cdev_lp
			model.set_bias(bias_new)
			test_loss, (prec, recall, f1) = model.eval(test_dset)
			results.append((test_loss, prec, recall, f1))
			model.set_bias(bias_old)
		results = np.array(results, dtype=np.float32)
		test_loss, prec, recall, f1 = np.mean(results, axis=0)
		print('Set bias:')
		print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))

	else:
		print('Evaluating fix bias.')