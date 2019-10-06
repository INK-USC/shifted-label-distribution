import torch
import torch.nn as nn
from torch.nn import functional as F
import model.object as obj

class noCluster(nn.Module):
    def __init__(self, opt):
        super(noCluster, self).__init__()
        self.opt = opt
        self.emblen = opt['emb_len']
        self.word_size = opt['word_size']
        self.type_size = opt['type_size']
        self.bag_weighting = opt['bag_weighting']
        self.label_distribution = opt['label_distribution']

        self.word_emb_bag = nn.EmbeddingBag(opt['word_size'], opt['emb_len'])
        self.word_emb_bag.weight = self.word_emb_bag.weight



        if opt['bias'] == 'fix':
            self.linear = nn.Linear(opt['emb_len'], opt['type_size'], bias=False)
            self.linear.weight.data.zero_()
            self.linear_bias = torch.log(torch.autograd.Variable(torch.cuda.FloatTensor(opt['label_distribution']), requires_grad=False))
        else:
            self.linear = nn.Linear(opt['emb_len'], opt['type_size'], bias=True)
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()

        self.crit = obj.partCE(if_average=opt['if_average'])
        self.drop_prob = opt['output_dropout']
        # self.crit = obj.softCE_S(if_average=if_average)
        # self.crit = obj.softCE(if_average=if_average)
        # self.crit = obj.softKL(if_average=if_average)


    def load_word_embedding(self, pre_embeddings):
        self.word_emb_bag.weight = nn.Parameter(pre_embeddings)

    def loss(self, typeTensor, feaDrop, offsetDrop):
        scores = self(feaDrop, offsetDrop)
        loss = self.crit(scores, typeTensor)
        return loss

    def forward(self, feature_seq, offset_seq):
        men_embedding = self.word_emb_bag(feature_seq, offset_seq)
        ret = self.linear(F.dropout(men_embedding, p=self.drop_prob, training=self.training))
        if self.opt['bias'] == 'fix':
            return ret + self.linear_bias
        else:
            return ret

    def test_with_bias(self, feature_seq, offset_seq, bias):
        bias = torch.log(torch.autograd.Variable(torch.cuda.FloatTensor(bias), requires_grad=False))
        men_embedding = self.word_emb_bag(feature_seq, offset_seq)
        return self.linear(F.dropout(men_embedding, p=self.drop_prob,
                                      training=self.training)) + bias

