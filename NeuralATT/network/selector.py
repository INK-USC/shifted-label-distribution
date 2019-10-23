
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class SelectorBase(nn.Module):
	def __init__(self, args, device, rel2id, repre_size, use_bag=True):
		super(SelectorBase, self).__init__()
		# hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, window_size, num_layers, dropout = \
		# 	args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
		# 	args.position_dim, args.attn_dim, args.window_size, args.num_layers, args.dropout

		self.rel_size = len(rel2id)
		self.repre_size = repre_size
		self.relation_mat = nn.Parameter(torch.Tensor(len(rel2id), repre_size))
		# torch.nn.init.xavier_uniform_(self.relation_mat.data)
		self.relation_mat.data.normal_(std=0.001)
		self.bias = nn.Parameter(torch.zeros(len(rel2id)))
		self.bias.data.zero_()
		self.use_bag = use_bag
		self.device = device

	def flinear(self, input):
		return F.linear(input, self.relation_mat, self.bias)



class AttentionSelector(SelectorBase):
	def __init__(self, args, device, rel2id, repre_size, use_bag=True):
		super(AttentionSelector, self).__init__(args, device, rel2id, repre_size, use_bag)

	def forward(self, repre, scope=None, labels=None, is_training=False):
		'''
		:param repre: batch of repre
		:param scope: [start, end)
		:param labels: batch of label (in the same bag, labels are repeating)
		:param is_training:
		:return:
		'''
		if self.use_bag and is_training:
			assert labels is not None
			assert scope is not None
			batch_size = labels.size()[0]
			labels = labels.view(batch_size, 1)
			label_onehot = torch.zeros(batch_size, self.rel_size).to(self.device).scatter_(1, labels, 1)
			rel_vecs = torch.mm(label_onehot, self.relation_mat)
			rel_logits = torch.bmm(repre.view(batch_size, 1, self.repre_size),
								   rel_vecs.view(batch_size, self.repre_size, 1)).view(batch_size)
			att_repres = []
			for start, end in scope:
				bag_repres = repre[start:end, :]
				bag_logits = rel_logits[start:end]
				bag_scores = torch.unsqueeze(F.softmax(bag_logits, dim=0), 1)
				att_repre = torch.sum(bag_scores*bag_repres, dim=0)
				att_repres.append(att_repre.view(1, self.repre_size))
			att_repres = torch.cat(att_repres, dim=0)
			logits = self.flinear(att_repres)
			return logits
		else:
			logits = self.flinear(repre)
			return logits

