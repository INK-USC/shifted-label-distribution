__author__ = 'Maosen'
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import pos2id, ner2id
import sys
from tqdm import tqdm

class BGRU(nn.Module):
	def __init__(self, args, rel2id, word_emb=None):
		super(BGRU, self).__init__()
		# arguments
		hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, num_layers = \
			args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
			args.position_dim, args.attn_dim, args.num_layers

		# bidirectional = args.bidirectional
		in_drop, intra_drop, state_drop, out_drop, bias = \
			args.in_drop, args.intra_drop, args.state_drop, args.out_drop, args.bias

		# embeddings
		if word_emb is not None:
			assert vocab_size, emb_dim == word_emb.shape
			self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID, _weight=torch.from_numpy(word_emb).float())
			# self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
			# self.word_emb.weight.requires_grad = False
		else:
			self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID)
			self.word_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

		self.pos_dim = pos_dim
		self.ner_dim = ner_dim
		self.hidden = hidden
		if pos_dim > 0:
			self.pos_emb = nn.Embedding(len(pos2id), pos_dim, padding_idx=utils.PAD_ID)
			self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
		if ner_dim > 0:
			self.ner_emb = nn.Embedding(len(ner2id), ner_dim, padding_idx=utils.PAD_ID)
			self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

		self.position_dim = position_dim
		if position_dim > 0:
			self.position_emb = nn.Embedding(utils.MAXLEN*2, position_dim)
			self.position_emb.weight.data.uniform_(-1.0, 1.0)

		# GRU
		# input_size = emb_dim + pos_dim + ner_dim
		input_size = emb_dim + position_dim*2 + pos_dim + ner_dim
		self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=intra_drop, bidirectional=True)

		self.input_dropout = nn.Dropout(in_drop)
		self.output_dropout = nn.Dropout(out_drop)

		# linear parameters of Position-aware attention
		feat_dim = hidden*2 + position_dim*2
		self.attn_dim = attn_dim
		self.feat_dim = feat_dim
		# self.wlinear = nn.Linear(feat_dim, attn_dim, bias=False)
		# self.vlinear = nn.Linear(attn_dim, 1, bias=False)
		self.flinear = nn.Linear(hidden*2, len(rel2id), bias=bias)
		# self.wlinear.weight.data.normal_(std=0.001)
		# self.vlinear.weight.data.zero_()
		self.flinear.weight.data.normal_(std=0.001)



	def forward(self, inputs):
		words, pos, ner, subj_pos, obj_pos = inputs
		# pos_subj and pos_obj are relative position to subject/object
		batch, maxlen = words.size()

		masks = torch.eq(words, utils.PAD_ID)
		seq_lens = masks.eq(utils.PAD_ID).long().sum(1)
		seq_lens = seq_lens.tolist()

		emb_words = self.word_emb(words)

		input_embs = [emb_words]

		if self.position_dim > 0:
			emb_subj_pos = self.position_emb(subj_pos + utils.MAXLEN)
			emb_obj_pos = self.position_emb(obj_pos + utils.MAXLEN)
			input_embs.append(emb_subj_pos)
			input_embs.append(emb_obj_pos)
		if self.pos_dim > 0:
			emb_pos = self.pos_emb(pos)
			input_embs.append(emb_pos)
		if self.ner_dim > 0:
			emb_ner = self.ner_emb(ner)
			input_embs.append(emb_ner)

		input = torch.cat(input_embs, dim=2).contiguous()
		input = self.input_dropout(input)

		input = nn.utils.rnn.pack_padded_sequence(input, seq_lens, batch_first=True)
		output, hn = self.gru(input)  # default: zero state
		output, output_lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

		# output = self.output_dropout(output)

		final_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
		final_hidden = self.output_dropout(final_hidden)

		logits = self.flinear(final_hidden)

		return logits





