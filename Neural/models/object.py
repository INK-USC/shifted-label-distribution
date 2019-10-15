import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random

def soft_max(vec, mask):
	batch_size = vec.size(0)
	_, idx = torch.max(vec, 1)  # B * 1
	idx = idx.view(batch_size, 1)
	max_score = torch.gather(vec, 1, idx)  # B * 1
	max_score = max_score.view(batch_size, 1)
	exp_score = torch.exp(vec - max_score.expand_as(vec))  # B * L
	exp_score = exp_score * mask  # B * L
	exp_score_sum = torch.sum(exp_score, 1).view(batch_size, 1).expand_as(exp_score)
	prob_score = exp_score / exp_score_sum
	return prob_score

class softCE(nn.Module):
	def __init__(self, if_average=False):
		super(softCE, self).__init__()
		self.logSoftmax = nn.LogSoftmax()
		self.if_average = if_average

	def forward(self, scores, target):
		# scores: B * L
		# target: B * L
		supervision_p = soft_max(scores, target)
		scores_logp = self.logSoftmax(scores)
		CE = (-supervision_p * scores_logp).sum()
		if self.if_average:
			CE = CE / scores.size(0)
		return CE


class softCE_S(nn.Module):
	def __init__(self, if_average=False):
		super(softCE_S, self).__init__()
		self.logSoftmax = nn.LogSoftmax()
		self.if_average = if_average

	def forward(self, scores, target):
		# scores: B * L
		# target: B * L

		supervision_p = soft_max(scores, target)
		supervision_ng = supervision_p.detach()
		scores_logp = self.logSoftmax(scores)
		CE = -(supervision_ng * scores_logp * target).sum()
		if self.if_average:
			CE = CE / scores.size(0)
		return CE


class partCE(nn.Module):
	def __init__(self, if_average=False):
		super(partCE, self).__init__()
		self.crit = nn.CrossEntropyLoss(size_average=if_average)
		self.maximum_score = 100000

	def forward(self, scores, target):
		# scores: B * L
		# target: B * L
		par_scores = scores - (1 - target) * self.maximum_score
		_, max_ind = torch.max(par_scores, 1)
		max_ind = max_ind.detach()
		loss = self.crit(scores, max_ind)
		# print(scores)
		# print(target)
		# print(par_scores)
		# print(max_ind)
		# input()
		return loss


class softKL(nn.Module):
	def __init__(self, if_average=False):
		super(softKL, self).__init__()
		self.logSoftmax = nn.LogSoftmax()
		self.if_average = if_average

	def forward(self, scores, target):
		# scores: B * L
		# target: B * L
		supervision_p = soft_max(scores, target)
		supervision_logp = (supervision_p + 0.000001 * (1 - target)).log()
		scores_logp = self.logSoftmax(scores)
		CE = (supervision_p * supervision_logp - supervision_p * scores_logp).sum()
		if self.if_average:
			CE = CE / scores.size(0)
		return CE