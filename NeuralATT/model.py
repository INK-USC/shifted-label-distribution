import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler

import utils
from network import encoder
from network.embedding import Embedding
from network.selector import AttentionSelector
import json

class Model(nn.Module):
	def __init__(self, args, device, rel2id, word_emb=None):
		super(Model, self).__init__()
		self.embedding = Embedding(args, word_emb)
		if args.encoder == 'pcnn':
			self.encoder = encoder.PCNN(args)
		elif args.encoder == 'bgru':
			self.encoder = encoder.BGRU(args)
		else:
			raise NotImplementedError
		self.selector = AttentionSelector(args, device, rel2id, self.encoder.outsize, use_bag=True)

	def forward(self, inputs, scope=None, labels=None, is_training=False):
		input_embs = self.embedding(inputs)
		repres = self.encoder(inputs, input_embs)
		logits = self.selector(repres, scope, labels, is_training)
		return logits


class Wrapper(object):
	def __init__(self, model, args, device, rel2id, fix_bias=False):
		lr = args.lr
		lr_decay = args.lr_decay
		self.cpu = torch.device('cpu')
		self.device = device
		self.args = args
		self.rel2id = rel2id
		self.max_grad_norm = args.max_grad_norm
		self.model = model.to(device)


		self.criterion = nn.CrossEntropyLoss()
		if fix_bias:
			self.model.selector.bias.requires_grad = False
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		# self.parameters = self.model.parameters()
		self.optimizer = torch.optim.SGD(self.parameters, lr)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=lr_decay)

	def update_lr(self, valid_loss):
		self.scheduler.step(valid_loss)

	def update(self, batch, scope):
		inputs = [p.to(self.device) for p in batch[:5]]
		labels = batch[5].to(self.device)
		self.model.train()
		logits = self.model(inputs, scope, labels, is_training=True)
		scope_starts = torch.LongTensor([x[0] for x in scope]).to(self.device)
		bag_labels = torch.take(labels, scope_starts)
		loss = self.criterion(logits, bag_labels)
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
		self.optimizer.step()
		return loss.item()

	def get_bias(self):
		return self.model.selector.bias.data

	def set_bias(self, bias):
		self.model.selector.bias.data = bias

	def predict(self, batch):
		inputs = [p.to(self.device) for p in batch[:5]]
		labels = batch[5].to(self.cpu)
		orig_idx = batch[6]
		logits = self.model(inputs).to(self.cpu)
		loss = self.criterion(logits, labels)
		pred = torch.argmax(logits, dim=1).to(self.cpu)
		# corrects = torch.eq(pred, labels)
		# acc_cnt = torch.sum(corrects, dim=-1)
		recover_idx = utils.recover_idx(orig_idx)
		logits = [logits[idx].tolist() for idx in recover_idx]
		pred = [pred[idx].item() for idx in recover_idx]
		labels = [labels[idx].item() for idx in recover_idx]
		return logits, pred, labels, loss.item()

	def eval(self, dset, vocab=None, output_false_file=None, output_label_file=None):
		rel_labels = ['']*len(dset.rel2id)
		for label, id in dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		loss = 0.0
		acc_cnt = 0.0
		trans_mat = np.zeros([len(self.rel2id), len(self.rel2id)], dtype=np.int32)
		pred_distr = np.ones(len(self.rel2id), dtype=np.float32)
		for idx, batch in enumerate(tqdm(dset.batched_data)):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b
			labels += labels_b
			loss += loss_b
			for j in range(len(labels_b)):
				trans_mat[labels_b[j]][pred_b[j]] += 1
				pred_distr[pred_b[j]] += 1
				if pred_b[j] == labels_b[j]:
					acc_cnt += 1

		pred_distr = np.log(pred_distr)
		max_log = np.max(pred_distr)
		pred_distr = pred_distr - max_log

		loss /= len(dset.batched_data)
		return loss, utils.eval(pred, labels), trans_mat, pred_distr, acc_cnt/len(pred)

	def save(self, filename, epoch):
		params = {
			'model': self.model.state_dict(),
			'config': self.args,
			'epoch': epoch
		}
		try:
			torch.save(params, filename)
			print("Epoch {}, model saved to {}".format(epoch, filename))
		except BaseException:
			print("[Warning: Saving failed... continuing anyway.]")
		# json.dump(vars(self.args), open('%s.json' % filename, 'w'))

	def count_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def load(self, filename):
		params = torch.load(filename)
		if type(params).__name__ == 'dict' and 'model' in params:
			self.model.load_state_dict(params['model'])
		else:
			self.model.load_state_dict(params)






