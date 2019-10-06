import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random

class NCE_loss(nn.Module):
    def __init__(self, wordSize, embedLen, if_average = False):

        super(NCE_loss, self).__init__()

        self.wordSize = wordSize
        self.embedLen = embedLen

        self.out_embed = nn.Embedding(self.wordSize, self.embedLen)

        self.if_average = if_average

        # self.in_embed = nn.Embedding(self.wordSize, self.embedLen)

    def forward(self, input_emb, pos_word, neg_sample, batch_size):
        #input_emb: (B * R) * L
        #pos_word: (B * R)

        tot_size = pos_word.size(0)
        # [batch_size, resample] = pos_word.size()

        output_emb = self.out_embed(pos_word)
        
        #gpu model
        noise = autograd.Variable(torch.Tensor(tot_size, neg_sample).uniform_(0, self.wordSize - 1).long()).cuda()
        #un comment for cpu mode
        # noise = autograd.Variable(torch.Tensor(tot_size, neg_sample).uniform_(0, self.wordSize - 1).long())

        noise_emb = self.out_embed(noise).neg()

        log_target = (input_emb * output_emb).sum(1).squeeze().sigmoid().log().sum()

        sum_log_sampled = torch.bmm(noise_emb, input_emb.unsqueeze(2)).sigmoid().log().sum()

        loss = log_target + sum_log_sampled

        if self.if_average:
            loss = loss / batch_size
        return -loss


    def load_neg_embedding(self, pre_embeddings):
        self.out_embed.weight = nn.Parameter(pre_embeddings)