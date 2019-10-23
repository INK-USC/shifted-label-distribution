'''
Training script with ramdom splitting dev set
'''
__author__ = 'Maosen'
import torch
from model import Model, Wrapper
import utils
from utils import Dataset
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os
import random

torch.backends.cudnn.deterministic = True


def train(args):
	# Training
	logging.info(str(args))

	model = Model(args, device, rel2id, emb_matrix)
	wrapper = Wrapper(model, args, device, train_dset.rel2id)

	max_dev_f1 = 0.0
	test_result_on_max_dev_f1 = (0.0, 0.0, 0.0)

	for iter in range(niter):
		# print('Iteration %d:' % iter)
		loss = 0.0
		for idx, batch in enumerate(tqdm(train_dset.batched_data)):
			scope = train_dset.batched_scope[idx]
			loss_batch = wrapper.update(batch, scope)
			loss += loss_batch
		loss /= len(train_dset.batched_data)

		valid_loss, (dev_prec, dev_recall, dev_f1), _, _, _ = wrapper.eval(dev_dset)
		logging.info('Iteration %d, Train loss %f' % (iter, loss))
		logging.info(
			'Dev loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(valid_loss, dev_prec, dev_recall,
																		dev_f1))
		test_loss, (test_prec, test_recall, test_f1), _, _, _ = wrapper.eval(test_dset)
		logging.info(
			'Test loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(test_loss, test_prec, test_recall,
																		 test_f1))
		if dev_f1 > max_dev_f1:
			max_dev_f1 = dev_f1
			test_result_on_max_dev_f1 = (test_prec, test_recall, test_f1)
			save_filename = os.path.join(args.save_dir, '%s_%d.pkl' % (args.info, runid))
			wrapper.save(save_filename, iter)

		wrapper.update_lr(valid_loss)

	logging.info('Max dev F1: %f' % max_dev_f1)
	test_p, test_r, test_f1 = test_result_on_max_dev_f1
	logging.info('Test P, R, F1 on best epoch: {:.4f}, {:.4f}, {:.4f}'.format(test_p, test_r, test_f1))
	logging.info('\n')

	return max_dev_f1, test_result_on_max_dev_f1


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='data/neural_att/KBP')
	parser.add_argument('--vocab_dir', type=str, default='data/neural/vocab')

	parser.add_argument('--encoder', type=str, default='pcnn', help='Model')

	parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
	parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
	parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
	parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
	parser.add_argument('--position_dim', type=int, default=30, help='Position encoding dimension.')

	parser.add_argument('--hidden', type=int, default=230, help='RNN hidden state size.')
	parser.add_argument('--window_size', type=int, default=3, help='Convolution window size')
	parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')

	parser.add_argument('--bidirectional', dest='bidirectional', action='store_true', help='Bidirectional RNN.')
	parser.set_defaults(bidirectional=True)

	# Data Loading & Pre-processing
	parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
	parser.add_argument('--no-lower', dest='lower', action='store_false')
	parser.set_defaults(lower=True)
	parser.add_argument('--batch_size', type=int, default=64)

	# Optimization
	parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
	parser.add_argument('--lr_decay', type=float, default=0.9)
	parser.add_argument('--num_epoch', type=int, default=30)
	parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

	# Optimization - Dropout
	parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
	parser.add_argument('--in_drop', type=float, default=0.5, help='Input dropout rate.')
	parser.add_argument('--intra_drop', type=float, default=0.3, help='Intra-layer dropout rate.')
	parser.add_argument('--out_drop', type=float, default=0.7, help='Output dropout rate.')

	# Other options
	parser.add_argument('--seed', type=int, default=7698)
	parser.add_argument('--repeat', type=int, default=5)
	parser.add_argument('--save_dir', type=str, default='./dumped_models', help='Root dir for saving models.')
	parser.add_argument('--info', type=str, default='KBP_default_ATT', help='Optional info for the experiment.')

	args = parser.parse_args()

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	logger = logging.getLogger()
	logger.setLevel(logging.INFO)

	# Load vocab file (id2word)
	with open(args.vocab_dir + '/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
	word2id = {}
	for idx, word in enumerate(vocab):
		word2id[word] = idx

	# Load word embedding
	emb_file = args.vocab_dir + '/embedding.npy'
	emb_matrix = np.load(emb_file)
	assert emb_matrix.shape[0] == len(vocab)
	assert emb_matrix.shape[1] == args.emb_dim
	args.vocab_size = len(vocab)
	niter = args.num_epoch

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('Using device: %s' % device.type)

	print('Reading data......')
	rel2id = utils.load_rel2id('%s/relation2id.json' % args.data_dir)
	train_filename = '%s/train.json' % args.data_dir
	test_filename = '%s/test.json' % args.data_dir
	dev_filename = '%s/dev.json' % args.data_dir
	train_dset = Dataset(train_filename, args, word2id, device, rel2id=rel2id, shuffle=True, use_bag=True)
	test_dset = Dataset(test_filename, args, word2id, device, rel2id=rel2id, use_bag=False)
	dev_dset = Dataset(dev_filename, args, word2id, device, rel2id=rel2id, use_bag=False)

	if not os.path.isdir(args.save_dir):
		os.makedirs(args.save_dir)

	for runid in range(1, args.repeat + 1):
		logging.info('Run model %d times......' % runid)
		dev_f1, test_result = train(args)
		logging.info('')
