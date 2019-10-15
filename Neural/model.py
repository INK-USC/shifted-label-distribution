'''
Model wrapper for Relation Extraction
Author: Maosen Zhang
Email: zhangmaosen@pku.edu.cn
'''
__author__ = 'Maosen'
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import torch.optim.lr_scheduler as lr_scheduler

import utils
from models.position_aware_lstm import PositionAwareLSTM
from models.bgru import BGRU
from models.cnn import CNN
from models.pcnn import PCNN
from models.lstm import LSTM
import json


class Model(object):
	def __init__(self, args, device, rel2id, word_emb=None):
		lr = args.lr
		lr_decay = args.lr_decay
		self.cpu = torch.device('cpu')
		self.device = device
		self.args = args
		self.rel2id = rel2id
		self.max_grad_norm = args.max_grad_norm
		if args.model == 'pa_lstm':
			self.model = PositionAwareLSTM(args, rel2id, word_emb)
		elif args.model == 'bgru':
			self.model = BGRU(args, rel2id, word_emb)
		elif args.model == 'cnn':
			self.model = CNN(args, rel2id, word_emb)
		elif args.model == 'pcnn':
			self.model = PCNN(args, rel2id, word_emb)
		elif args.model == 'lstm':
			self.model = LSTM(args, rel2id, word_emb)
		else:
			raise ValueError
		self.model.to(device)
		self.criterion = nn.CrossEntropyLoss()
		if args.fix_bias:
			self.model.flinear.bias.requires_grad = False
		self.parameters = [p for p in self.model.parameters() if p.requires_grad]
		# self.parameters = self.model.parameters()
		self.optimizer = torch.optim.SGD(self.parameters, lr)
		self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=lr_decay)

	def update_lr(self, valid_loss):
		self.scheduler.step(valid_loss)

	def update(self, batch, penalty=False, weight=1.0):
		inputs = [p.to(self.device) for p in batch[:5]]
		labels = batch[5].to(self.device)
		self.model.train()
		logits = self.model(inputs)
		loss = self.criterion(logits, labels)
		# batch_ent = utils.calcEntropy(logits)
		# ent = torch.sum(batch_ent) / len(batch_ent)
		# if penalty:
		# 	loss = loss - ent*weight
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
		self.optimizer.step()
		return loss.item()

	def get_bias(self):
		return self.model.flinear.bias.data

	def set_bias(self, bias):
		self.model.flinear.bias.data = bias

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

	def eval(self, dset, vocab=None, output_false_file=None, output_label_file=None, weights=None):
		if weights is None:
			weights = [1.0] * len(dset.rel2id)

		rel_labels = ['']*len(dset.rel2id)
		for label, id in dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		loss = 0.0

		for idx, batch in enumerate(dset.batched_data):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b
			labels += labels_b
			loss += loss_b

			if output_false_file is not None and vocab is not None:
				all_words, pos, ner, subj_pos, obj_pos, labels_, _ = batch
				all_words = all_words.tolist()
				output_false_file.write('\n')
				for i, word_ids in enumerate(all_words):
					if labels[i] != pred[i]:
						length = 0
						for wid in word_ids:
							if wid != utils.PAD_ID:
								length += 1
						words = [vocab[wid] for wid in word_ids[:length]]
						sentence = ' '.join(words)

						subj_words = []
						for sidx in range(length):
							if subj_pos[i][sidx] == 0:
								subj_words.append(words[sidx])
						subj = '_'.join(subj_words)

						obj_words = []
						for oidx in range(length):
							if obj_pos[i][oidx] == 0:
								obj_words.append(words[oidx])
						obj = '_'.join(obj_words)

						output_false_file.write('%s\t%s\t%s\t%s\t%s\n' % (sentence, subj, obj, rel_labels[pred[i]], rel_labels[labels[i]]))

		if output_label_file is not None and vocab is not None:
			output_label_file.write(json.dumps(pred) + '\n')
			output_label_file.write(json.dumps(labels) + '\n')


		loss /= len(dset.batched_data)
		return loss, utils.eval(pred, labels, weights)

	def TuneEntropyThres(self, test_dset, noneInd=utils.NO_RELATION, ratio=0.2, cvnum=100):
		'''
		Tune threshold on test set
		'''
		rel_labels = [''] * len(test_dset.rel2id)
		for label, id in test_dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		scores = []
		loss = 0.0
		for idx, batch in enumerate(test_dset.batched_data):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b
			labels += labels_b
			scores += scores_b
			loss += loss_b
		loss /= len(test_dset.batched_data)

		# start tuning
		scores = torch.tensor(scores)
		f1score = 0.0
		recall = 0.0
		precision = 0.0

		pre_ind = utils.calcInd(scores)
		pre_entropy = utils.calcEntropy(scores)
		valSize = int(np.floor(ratio * len(pre_ind)))
		data = [[pre_ind[ind], pre_entropy[ind], labels[ind]] for ind in range(0, len(pre_ind))]

		for cvind in tqdm(range(cvnum)):
			random.shuffle(data)
			val = data[0:valSize]
			eva = data[valSize:]

			# find best threshold
			max_ent = max(val, key=lambda t: t[1])[1]
			min_ent = min(val, key=lambda t: t[1])[1]
			stepSize = (max_ent - min_ent) / 100
			thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
			ofInterest = 0
			for ins in val:
				if ins[2] != noneInd:
					ofInterest += 1
			bestThreshold = float('nan')
			bestF1 = float('-inf')
			for threshold in thresholdList:
				corrected = 0
				predicted = 0
				for ins in val:
					if ins[1] < threshold and ins[0] != noneInd:
						predicted += 1
						if ins[0] == ins[2]:
							corrected += 1
				curF1 = 2.0 * corrected / (ofInterest + predicted)
				if curF1 > bestF1:
					bestF1 = curF1
					bestThreshold = threshold
			ofInterest = 0
			corrected = 0
			predicted = 0
			for ins in eva:
				if ins[2] != noneInd:
					ofInterest += 1
				if ins[1] < bestThreshold and ins[0] != noneInd:
					predicted += 1
					if ins[0] == ins[2]:
						corrected += 1

			f1score += (2.0 * corrected / (ofInterest + predicted))
			recall += (1.0 * corrected / ofInterest)
			precision += (1.0 * corrected / (predicted + 0.00001))


		f1score /= cvnum
		recall /= cvnum
		precision /= cvnum

		return loss, f1score, recall, precision

	def TuneMaxThres(self, test_dset, noneInd=utils.NO_RELATION, ratio=0.2, cvnum=100):
		'''
		Tune threshold on test set
		'''
		rel_labels = [''] * len(test_dset.rel2id)
		for label, id in test_dset.rel2id.items():
			rel_labels[id] = label
		self.model.eval()
		pred = []
		labels = []
		scores = []
		loss = 0.0
		for idx, batch in enumerate(test_dset.batched_data):
			scores_b, pred_b, labels_b, loss_b = self.predict(batch)
			pred += pred_b
			labels += labels_b
			scores += scores_b
			loss += loss_b
		loss /= len(test_dset.batched_data)

		# start tuning
		scores = torch.tensor(scores)
		f1score = 0.0
		recall = 0.0
		precision = 0.0

		pre_prob, pre_ind = torch.max(scores, 1)
		valSize = int(np.floor(ratio * len(pre_ind)))
		data = [[pre_ind[ind], pre_prob[ind], labels[ind]] for ind in range(0, len(pre_ind))]
		for cvind in tqdm(range(cvnum)):
			random.shuffle(data)
			val = data[0:valSize]
			eva = data[valSize:]

			# find best threshold
			max_ent = max(val, key=lambda t: t[1])[1]
			min_ent = min(val, key=lambda t: t[1])[1]
			stepSize = (max_ent - min_ent) / 100
			thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
			ofInterest = 0
			for ins in val:
				if ins[2] != noneInd:
					ofInterest += 1
			bestThreshold = float('nan')
			bestF1 = float('-inf')
			for threshold in thresholdList:
				corrected = 0
				predicted = 0
				for ins in val:
					if ins[1] > threshold and ins[0] != noneInd:
						predicted += 1
						if ins[0] == ins[2]:
							corrected += 1
				curF1 = 2.0 * corrected / (ofInterest + predicted)
				if curF1 > bestF1:
					bestF1 = curF1
					bestThreshold = threshold

			ofInterest = 0
			corrected = 0
			predicted = 0
			for ins in eva:
				if ins[2] != noneInd:
					ofInterest += 1
				if ins[1] > bestThreshold and ins[0] != noneInd:
					predicted += 1
					if ins[0] == ins[2]:
						corrected += 1
			f1score += (2.0 * corrected / (ofInterest + predicted))
			recall += (1.0 * corrected / ofInterest)
			precision += (1.0 * corrected / (predicted + 0.00001))

		f1score /= cvnum
		recall /= cvnum
		precision /= cvnum

		return loss, f1score, recall, precision

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






