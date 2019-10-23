import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import pos2id, ner2id
import sys
from tqdm import tqdm

class PCNN(nn.Module):
	def __init__(self, args):
		super(PCNN, self).__init__()
		# arguments
		hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, window_size, num_layers, dropout = \
			args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
			args.position_dim, args.attn_dim, args.window_size, args.num_layers, args.dropout

		# CNN
		input_size = emb_dim + position_dim*2 + pos_dim + ner_dim
		self.input_size = input_size
		self.cnn = torch.nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=(input_size, window_size), stride=1)
		# torch.nn.init.xavier_uniform_(self.cnn.weight.data)
		# self.cnn.bias.data.zero_()
		self.pooling = torch.nn.AdaptiveMaxPool1d(1)
		self.dropout = nn.Dropout(dropout)
		self.outsize = hidden*3


	def masked_conv(self, input, mask):
		batch, timest, input_size = input.size()
		mask = mask.unsqueeze(2)
		mask = mask.expand(batch, timest, input_size)
		mask = mask.to(torch.float32)
		input = mask * input

		input = torch.transpose(input, 1, 2)  # [batch, input_size, seq_len]
		input = torch.unsqueeze(input, dim=1)  # [batch, 1, input_size, seq_len]

		output = self.cnn(input)  # [batch, hidden, 1, time']
		output = torch.squeeze(output, 2)  # [batch, hidden, time']
		output = self.pooling(output)  # [batch, hidden, 1]
		output = torch.squeeze(output, 2)  # [batch, hidden]
		output = torch.nn.functional.tanh(output)
		return output

	def forward(self, inputs, input_embs):
		words, pos, ner, subj_pos, obj_pos = inputs
		# pos_subj and pos_obj are relative position to subject/object
		batch, maxlen = words.size()

		masks = torch.eq(words, utils.PAD_ID)
		seq_lens = masks.eq(utils.PAD_ID).long().sum(1).squeeze().tolist()

		piece3mask = (subj_pos >= 0) & (obj_pos >= 0)
		piece1mask = (subj_pos <= 0) & (obj_pos <= 0)
		piece2mask = (obj_pos > 0) ^ (subj_pos > 0)


		piece1 = self.masked_conv(input_embs, piece1mask)
		piece2 = self.masked_conv(input_embs, piece2mask)
		piece3 = self.masked_conv(input_embs, piece3mask)

		output = torch.cat([piece1, piece2, piece3], dim=1)
		output = self.dropout(output)

		return output



class BGRU(nn.Module):
	def __init__(self, args):
		super(BGRU, self).__init__()
		# arguments
		hidden, vocab_size, emb_dim, pos_dim, ner_dim, position_dim, attn_dim, num_layers = \
			args.hidden, args.vocab_size, args.emb_dim, args.pos_dim, args.ner_dim, \
			args.position_dim, args.attn_dim, args.num_layers

		# bidirectional = args.bidirectional
		in_drop, intra_drop, out_drop = \
			args.in_drop, args.intra_drop, args.out_drop

		# GRU
		input_size = emb_dim + position_dim*2 + pos_dim + ner_dim
		self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, num_layers=num_layers, batch_first=True, dropout=intra_drop, bidirectional=True)

		self.input_dropout = nn.Dropout(in_drop)
		self.output_dropout = nn.Dropout(out_drop)
		self.outsize = hidden*2

	def forward(self, inputs, input_embs):
		words, pos, ner, subj_pos, obj_pos = inputs
		# pos_subj and pos_obj are relative position to subject/object
		batch, maxlen = words.size()

		masks = torch.eq(words, utils.PAD_ID)
		seq_lens = masks.eq(utils.PAD_ID).long().sum(1)
		seq_lens = seq_lens.tolist()

		input = self.input_dropout(input_embs)

		input = nn.utils.rnn.pack_padded_sequence(input, seq_lens, batch_first=True)
		output, hn = self.gru(input)  # default: zero state
		output, output_lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

		# output = self.output_dropout(output)

		final_hidden = torch.cat([hn[-2], hn[-1]], dim=1)
		final_hidden = self.output_dropout(final_hidden)

		return final_hidden


