'''
Training script for Position-Aware LSTM for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
from model import Model, Wrapper
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

# train_dset = Dataset(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True,
# 					 mask_with_type=args.mask_with_type)
test_dset = Dataset(test_filename, args, word2id, device, rel2id=rel2id, use_bag=False)
# train_lp = torch.from_numpy(train_dset.log_prior).to(device)

for runid in range(1, args.repeat + 1):
	model = Model(args, device, rel2id, emb_matrix)
	wrapper = Wrapper(model, args, device, rel2id)
	wrapper.load('%s_%d.pkl' % (model_file, runid))

	# Original
	test_loss, (prec, recall, f1), _, _, _ = wrapper.eval(test_dset)
	print('Original:')
	print('Test loss %.4f, Precision %.4f, Recall %.4f, F1 %.4f' % (test_loss, prec, recall, f1))

