import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import random
from model.utils import *
import functools


class repack():
    def __init__(self, ratio, resampleNum, if_cuda):
        self.ratio = ratio
        self.cur_dropout = functools.partial(dropout, ratio=self.ratio)
        self.resampleNum = resampleNum
        self.cur_resample = functools.partial(resample, resample_num=self.resampleNum)
        self.if_cuda = if_cuda

    def repack(self, feature_list, type_list):
        type_tensor = torch.cat(type_list, dim=0)
        scope_list = [0]
        feature_list_dropout = list(map(self.cur_dropout, feature_list))

        scope_list = torch.LongTensor(scope_list)
        offset_dropout_tensor = torch.LongTensor(
            [0] + list(map(lambda t: t.size(0), feature_list_dropout[:-1]))).cumsum(0)
        feature_list_dropout_tensor = torch.cat(feature_list_dropout, dim=0)

        feature_list_resample_tensor_1 = torch.cat(list(map(self.cur_resample, feature_list)), dim=0)
        feature_list_resample_tensor_2 = torch.cat(list(map(self.cur_resample, feature_list)), dim=0)

        if self.if_cuda:
            return autograd.Variable(type_tensor).cuda(), autograd.Variable(
                feature_list_resample_tensor_1).cuda(), autograd.Variable(
                feature_list_resample_tensor_2).cuda(), autograd.Variable(
                feature_list_dropout_tensor).cuda(), autograd.Variable(offset_dropout_tensor).cuda(), \
                autograd.Variable(scope_list).cuda()
        else:
            return autograd.Variable(type_tensor), autograd.Variable(feature_list_resample_tensor_1), autograd.Variable(
                feature_list_resample_tensor_2), autograd.Variable(feature_list_dropout_tensor), autograd.Variable(
                offset_dropout_tensor), autograd.Variable(scope_list).cuda()

    def repack_eva(self, feature_list):
        offset_tensor = torch.LongTensor([0] + list(map(lambda t: t.size(0), feature_list[:-1]))).cumsum(0)
        feature_list_tensor = torch.cat(feature_list, dim=0)

        if self.if_cuda:
            return autograd.Variable(feature_list_tensor).cuda(), autograd.Variable(offset_tensor).cuda()
        else:
            return autograd.Variable(feature_list_tensor), autograd.Variable(offset_tensor)
