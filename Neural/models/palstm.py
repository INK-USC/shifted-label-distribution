import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import utils
# from utils import pos2id, ner2id, PAD_ID, MAX_LEN

class PositionAwareRNN(nn.Module):
	""" A sequence model for relation extraction. """

	def __init__(self, opt, rel2id, emb_matrix=None):
		super(PositionAwareRNN, self).__init__()

		opt = vars(opt)
		opt['attn'] = True

		self.input_drop = nn.Dropout(opt['in_drop'])
		self.output_drop = nn.Dropout(opt['out_drop'])
		self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=utils.PAD_ID)
		if opt['pos_dim'] > 0:
			self.pos_emb = nn.Embedding(len(utils.pos2id), opt['pos_dim'],
					padding_idx=utils.PAD_ID)
		if opt['ner_dim'] > 0:
			self.ner_emb = nn.Embedding(len(utils.ner2id), opt['ner_dim'],
					padding_idx=utils.PAD_ID)

		input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
		self.rnn = nn.LSTM(input_size, opt['hidden'], opt['num_layers'], batch_first=True,\
				dropout=opt['intra_drop'])
		self.flinear = nn.Linear(opt['hidden'], len(rel2id))

		if opt['attn']:
			self.attn_layer = PositionAwareAttention(opt['hidden'],
					opt['hidden'], 2*opt['position_dim'], opt['attn_dim'])
			self.pe_emb = nn.Embedding(utils.MAXLEN * 2 + 1, opt['position_dim'])

		self.opt = opt
		self.topn = self.opt.get('topn', 1e10)
		self.use_cuda = opt['cuda']
		self.emb_matrix = emb_matrix
		self.init_weights()

	def init_weights(self):
		if self.emb_matrix is None:
			self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
		else:
			self.emb_matrix = torch.from_numpy(self.emb_matrix)
			self.emb.weight.data.copy_(self.emb_matrix)
		if self.opt['pos_dim'] > 0:
			self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
		if self.opt['ner_dim'] > 0:
			self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

		self.flinear.bias.data.fill_(0)
		init.xavier_uniform(self.flinear.weight, gain=1) # initialize linear layer
		if self.opt['attn']:
			self.pe_emb.weight.data.uniform_(-1.0, 1.0)

		# decide finetuning
		if self.topn <= 0:
			print("Do not finetune word embedding layer.")
			self.emb.weight.requires_grad = False
		elif self.topn < self.opt['vocab_size']:
			print("Finetune top {} word embeddings.".format(self.topn))
			self.emb.weight.register_hook(lambda x: \
					utils.keep_partial_grad(x, self.topn))
		else:
			print("Finetune all embeddings.")

	def zero_state(self, batch_size):
		state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden'])
		h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
		if self.use_cuda:
			return h0.cuda(), c0.cuda()
		else:
			return h0, c0

	def forward(self, inputs):
		words, pos, ner, subj_pos, obj_pos = inputs  # unpack
		masks = torch.eq(words, 0)
		seq_lens = list(masks.data.eq(utils.PAD_ID).long().sum(1).squeeze())
		batch_size = words.size()[0]

		# embedding lookup
		word_inputs = self.emb(words)
		inputs = [word_inputs]
		if self.opt['pos_dim'] > 0:
			inputs += [self.pos_emb(pos)]
		if self.opt['ner_dim'] > 0:
			inputs += [self.ner_emb(ner)]
		inputs = self.input_drop(torch.cat(inputs, dim=2)) # add dropout to input
		input_size = inputs.size(2)

		# rnn
		h0, c0 = self.zero_state(batch_size)
		inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
		outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
		outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
		hidden = self.output_drop(ht[-1,:,:]) # get the outmost layer h_n
		outputs = self.output_drop(outputs)

		# attention
		if self.opt['attn']:
			# convert all negative PE numbers to positive indices
			# e.g., -2 -1 0 1 will be mapped to 98 99 100 101
			subj_pe_inputs = self.pe_emb(subj_pos + utils.MAXLEN)
			obj_pe_inputs = self.pe_emb(obj_pos + utils.MAXLEN)
			pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
			final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
		else:
			final_hidden = hidden

		logits = self.flinear(final_hidden)
		return logits

class PositionAwareAttention(nn.Module):
	"""
	A position-augmented attention layer where the attention weight is
	a = T' . tanh(Ux + Vq + Wf)
	where x is the input, q is the query, and f is additional position features.
	"""

	def __init__(self, input_size, query_size, feature_size, attn_size):
		super(PositionAwareAttention, self).__init__()
		self.input_size = input_size
		self.query_size = query_size
		self.feature_size = feature_size
		self.attn_size = attn_size
		self.ulinear = nn.Linear(input_size, attn_size)
		self.vlinear = nn.Linear(query_size, attn_size, bias=False)
		if feature_size > 0:
			self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
		else:
			self.wlinear = None
		self.tlinear = nn.Linear(attn_size, 1)
		self.init_weights()

	def init_weights(self):
		self.ulinear.weight.data.normal_(std=0.001)
		self.vlinear.weight.data.normal_(std=0.001)
		if self.wlinear is not None:
			self.wlinear.weight.data.normal_(std=0.001)
		self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

	def forward(self, x, x_mask, q, f):
		"""
		x : batch_size * seq_len * input_size
		q : batch_size * query_size
		f : batch_size * seq_len * feature_size
		"""
		batch_size, seq_len, _ = x.size()

		x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
			batch_size, seq_len, self.attn_size)
		q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
			batch_size, self.attn_size).unsqueeze(1).expand(
			batch_size, seq_len, self.attn_size)
		if self.wlinear is not None:
			f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
				batch_size, seq_len, self.attn_size)
			projs = [x_proj, q_proj, f_proj]
		else:
			projs = [x_proj, q_proj]
		scores = self.tlinear(F.tanh(sum(projs)).view(-1, self.attn_size)).view(
			batch_size, seq_len)

		# mask padding
		scores.data.masked_fill_(x_mask.data, -float('inf'))
		weights = F.softmax(scores)
		# weighted average input vectors
		outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
		return outputs

