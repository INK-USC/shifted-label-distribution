import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import pos2id, ner2id
import sys
from tqdm import tqdm

class Embedding(nn.Module):
	def __init__(self, args, word_emb=None):
		super(Embedding, self).__init__()
		# arguments
		hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, window_size, num_layers, dropout = \
			args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
			args.position_dim, args.attn_dim, args.window_size, args.num_layers, args.dropout

		# embeddings
		if word_emb is not None:
			assert vocab_size, emb_dim == word_emb.shape
			self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID, _weight=torch.from_numpy(word_emb).float())
			# self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
			# self.word_emb.weight.requires_grad = False
		else:
			self.word_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=utils.PAD_ID)
			# torch.nn.init.xavier_uniform_(self.word_emb.weight.data[1:, :])
			self.word_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

		self.pos_dim = pos_dim
		self.ner_dim = ner_dim
		self.hidden = hidden
		if pos_dim > 0:
			self.pos_emb = nn.Embedding(len(pos2id), pos_dim, padding_idx=utils.PAD_ID)
			# torch.nn.init.xavier_uniform_(self.pos_emb.weight.data[1:, :])
			self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
		if ner_dim > 0:
			self.ner_emb = nn.Embedding(len(ner2id), ner_dim, padding_idx=utils.PAD_ID)
			# torch.nn.init.xavier_uniform_(self.ner_emb.weight.data[1:, :])
			self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

		self.position_dim = position_dim
		if position_dim > 0:
			self.position_emb = nn.Embedding(utils.MAXLEN*2, position_dim)
			# torch.nn.init.xavier_uniform_(self.position_emb.weight.data)
			self.position_emb.weight.data.uniform_(-1.0, 1.0)


	def forward(self, inputs):
		words, pos, ner, subj_pos, obj_pos = inputs
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

		input_embs = torch.cat(input_embs, dim=2).contiguous()
		return input_embs





