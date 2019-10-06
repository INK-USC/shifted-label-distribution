import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random
import model.utils as utils

class softCE(nn.Module):
    def __init__(self, if_average = False):
        super(softCE, self).__init__()
        self.logSoftmax = nn.LogSoftmax()
        self.if_average = if_average

    def forward(self, scores, target):
    	#scores: B * L
    	#target: B * L
    	supervision_p = utils.soft_max(scores, target)
    	scores_logp = self.logSoftmax(scores)
    	CE = (-supervision_p * scores_logp).sum()
    	if self.if_average:
    		CE = CE / scores.size(0)
    	return CE

class softCE_S(nn.Module):
    def __init__(self, if_average = False):
        super(softCE_S, self).__init__()
        self.logSoftmax = nn.LogSoftmax()
        self.if_average = if_average

    def forward(self, scores, target):
    	#scores: B * L
    	#target: B * L
    	
    	supervision_p = utils.soft_max(scores, target)
    	supervision_ng = supervision_p.detach()
    	scores_logp = self.logSoftmax(scores)
    	CE = -(supervision_ng * scores_logp * target).sum()
    	if self.if_average:
    		CE = CE / scores.size(0)
    	return CE

class partCE(nn.Module):
    def __init__(self, if_average = False):
        super(partCE, self).__init__()
        self.crit = nn.CrossEntropyLoss(size_average=if_average)
        self.maximum_score = 100000

    def forward(self, scores, target):
        #scores: B * L
        #target: B * L
        par_scores = scores - (1 - target) * self.maximum_score
        _, max_ind = torch.max(par_scores, 1)
        max_ind = max_ind.detach()
        loss = self.crit(scores, max_ind)
        return loss
        
class softKL(nn.Module):
    def __init__(self, if_average = False):
        super(softKL, self).__init__()
        self.logSoftmax = nn.LogSoftmax()
        self.if_average = if_average

    def forward(self, scores, target):
    	#scores: B * L
    	#target: B * L
    	supervision_p = utils.soft_max(scores, target)
    	supervision_logp = (supervision_p + 0.000001 * (1 - target)).log()
    	scores_logp = self.logSoftmax(scores)
    	CE = (supervision_p * supervision_logp -supervision_p * scores_logp).sum()
    	if self.if_average:
    		CE = CE / scores.size(0)
    	return CE