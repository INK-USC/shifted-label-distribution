import torch
import torch.nn as nn
import itertools
import numpy as np
import random

zip = getattr(itertools, 'izip', zip)

def get_none_id(type_filename):
    with open(type_filename, encoding='utf-8') as type_file:
        for line in type_file:
            ls = line.strip().split()
            if ls[0] == "None":
                return int(ls[1])

def get_distribution(type_filename):
    with open(type_filename, encoding='utf-8') as type_file:
        sum = 0
        prob_lst = []
        for line in type_file:
            ls = line.strip().split()
            sum += int(ls[2])
            prob_lst.append(int(ls[2]))
        prob_lst = [(c+1e-10) / float(sum) for c in prob_lst]
    return prob_lst

def get_distribution_from_list(labels, max_index=None):
    labels = [labels[i][0] for i in range(len(labels))]
    #print(labels)
    if max_index is None:
        max_index = max(labels) + 1
    count = [0] * (max_index)
    for i, item in enumerate(labels):
        count[item] += 1
    prob = [item / len(labels) for item in count]
    return prob


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def log_sum_exp(vec):
    # vec: B * L * M
    # output: B * 1 * M
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx)  # B * 1 * M
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1))  # B * 1 * M


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


def load_embedding(emb_file, randIni=True):
    fin = open(emb_file, 'r')
    line = fin.readline().split(' ')
    size = int(line[0])
    embLen = int(line[1])
    if randIni:
        bias = np.sqrt(3.0 / embLen)
        pos_embedding_tensor = torch.rand(size, embLen) * 2 * bias - bias
        neg_embedding_tensor = torch.FloatTensor(size, embLen).zero_()
    else:
        pos_embedding_array = list()
        neg_embedding_array = list()
        for ind in range(size):
            line = fin.readline()
            if line.isspace():
                break
            line = line.split(' ')
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line)))
            pos_embedding_array.append(vector)
        fin.readline()
        for ind in range(size):
            line = fin.readline()
            if line.isspace():
                break
            line = line.split(' ')
            vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line)))
            neg_embedding_array.append(vector)
        pos_embedding_tensor = torch.FloatTensor(np.asarray(pos_embedding_array))
        neg_embedding_tensor = torch.FloatTensor(np.asarray(neg_embedding_array))
    return size, embLen, pos_embedding_tensor, neg_embedding_tensor


def initialize_embedding(feature_file, embLen):
    fin = open(feature_file, 'r')
    lines = fin.readlines()
    size = len(lines)
    bias = np.sqrt(3.0 / embLen)
    embedding_tensor = torch.rand(size, embLen) * 2 * bias - bias
    return size, embedding_tensor


def load_corpus(corpus):
    fin = open(corpus, 'r')
    line = fin.readline()
    size = int(line)
    feature_list = list()
    label_list = list()
    type_size = -1
    for line in fin:
        if line.isspace():
            break
        line = line.split('\t')
        fv = torch.LongTensor()
        feature_list.append(torch.LongTensor(list(map(lambda t: int(t), line[2].split(' ')))))
        label_vec = list(map(lambda t: int(t), line[4].split(' ')))
        tmp = max(label_vec)
        if type_size < tmp:
            type_size = tmp
        #if if_cuda:
        #    label_list.append(torch.cuda.LongTensor(label_vec))
        #else:
        label_list.append(torch.LongTensor(label_vec))
    type_size = type_size + 1
    type_list = list()
    for label in label_list:
        type_vec = torch.FloatTensor(type_size).zero_()
        type_vec[label] = 1
        type_vec = type_vec.view(1, -1)
        type_list.append(type_vec)
    return size, type_size, feature_list, label_list, type_list

def load_qa_corpus(corpus):
    fin = open(corpus, 'r')
    size = 0
    feature_list = list()
    for line in fin:
        if line.isspace():
            break
        size += 1
        line = line.strip('\r\n').split('\t')
        feature_list.append(torch.LongTensor(list(map(lambda t: int(t), line[1].split(',')))))
    return size, feature_list


def load_question_info(mention_question_file):
    pos_qapairs = []
    pos_qapair_to_question = {}
    question_to_qapairs = {}
    with open(mention_question_file, 'r') as fin:
        for line in fin:
            if line.isspace():
                break
            line = line.strip('\r\n').split('\t')
            mid = int(line[0])
            qid = int(line[1])
            if float(line[2]) == 1.0:
                pos_qapairs.append(mid)
                pos_qapair_to_question[mid] = qid
                if qid in question_to_qapairs:
                    pos_mids = question_to_qapairs[qid][0]
                    pos_mids.append(mid)
                    question_to_qapairs[qid] = (pos_mids, question_to_qapairs[qid][1])
                else:
                    question_to_qapairs[qid] = ([mid], [])
            else:
                if qid in question_to_qapairs:
                    neg_mids = question_to_qapairs[qid][1]
                    neg_mids.append(mid)
                    question_to_qapairs[qid] = (question_to_qapairs[qid][0], neg_mids)
                else:
                    question_to_qapairs[qid] = ([], [mid])
    return pos_qapairs, pos_qapair_to_question, question_to_qapairs


def shuffle_data(ori_labels, ori_corpus):
    assert (len(ori_labels) == len(ori_corpus))
    index_shuf = list(range(len(ori_labels)))
    random.shuffle(index_shuf)
    labels = [ori_labels[i] for i in index_shuf]
    corpus = [ori_corpus[i] for i in index_shuf]
    return labels, corpus


def calcEntropy(batch_scores):
    # input: B * L
    # output: B
    batch_probs = nn.functional.softmax(batch_scores)
    return torch.sum(batch_probs * torch.log(batch_probs), 1).neg()

def calcInd(batch_probs):
    # input: B * L
    # output: B
    _, ind = torch.max(batch_probs, 1)
    return ind

def calcMaxProb(batch_scores):
    batch_probs = nn.functional.softmax(batch_scores)
    # batch_probs = batch_scores / sum(batch_scores)
    prob, _ = torch.max(batch_probs, 1)
    return prob

def dropout(input_tensor, ratio):
    keep_vec = torch.ge(torch.rand(input_tensor.size()), ratio)
    return input_tensor[keep_vec]


def resample(input_tensor, resample_num):
    resample_ind = torch.Tensor(resample_num).uniform_(0, input_tensor.size(0) - 1).long()
    return input_tensor[resample_ind].view(-1)


def clip_grad(parameters, max_value):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        mask = p.grad.data.gt(max_value)
        p.grad.data[mask] = max_value
        mask = p.grad.data.lt(-max_value)
        p.grad.data[mask] = -max_value


def CrossValidation(pre_ind, pre_entropy, true_ind, noneInd, ratio=0.1, cvnum=100):
    f1score = 0.0
    recall = 0.0
    precision = 0.0
    meanBestF1 = 0.0
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_entropy[ind], true_ind[ind]] for ind in range(0, len(pre_ind))]
    for cvind in range(cvnum):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2][0] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] < threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2][0]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold
        meanBestF1 += bestF1
        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2][0] != noneInd:
                ofInterest += 1
            if ins[1] < bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2][0]:
                    corrected += 1
        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))

    meanBestF1 /= cvnum
    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum
    return f1score, recall, precision, meanBestF1


def eval_score(pre_ind, true_ind_dev, noneInd):
    eva = [[pre_ind[ind], true_ind_dev[ind]] for ind in range(0, len(pre_ind))]

    ofInterest = 0
    corrected = 0
    predicted = 0

    for ins in eva:
        if ins[1][0] != noneInd:
            ofInterest += 1
        if ins[0] != noneInd:
            predicted += 1
            if ins[0] == ins[1][0]:
                corrected += 1
    f1score = (2.0 * corrected / (ofInterest + predicted))
    recall = (1.0 * corrected / ofInterest)
    precision = (1.0 * corrected / (predicted + 1e-10))

    return f1score, recall, precision


def TuneThres(pre_ind, pre_entropy, true_ind, noneInd, thres_type='max', ratio=0.2, cvnum=100):
    # > for 'max' and < for 'entropy'
    if thres_type == 'max':
        SIGN = 1
    else:
        SIGN = -1

    f1score = 0.0
    recall = 0.0
    precision = 0.0
    val_f1 = 0.0

    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_entropy[ind], true_ind[ind]] for ind in range(0, len(pre_ind))]

    for cvind in range(cvnum):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]

        ofInterest = 0
        for ins in val:
            if ins[2][0] != noneInd:
                ofInterest += 1

        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if SIGN * ins[1] > SIGN * threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2][0]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold

        val_f1 += bestF1

        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2][0] != noneInd:
                ofInterest += 1
            if SIGN * ins[1] > SIGN * bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2][0]:
                    corrected += 1

        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 1e-10))

    val_f1 /= cvnum
    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum

    return f1score, recall, precision, val_f1


def eval_score_with_thres(pre_ind, pre_entropy, true_ind, noneInd, threshold, thres_type='max'):
    # print prediction after thresholding to a file
    if thres_type == 'max':
        SIGN = 1
    else:
        SIGN = -1
        
    # calculate the f1 without threshold on ndev set
    f1score = 0.0
    recall = 0.0
    precision = 0.0
    data = [[pre_ind[ind], pre_entropy[ind], true_ind[ind]] for ind in range(0, len(pre_ind))]

    ofInterest = 0
    for ins in data:
        if ins[2][0] != noneInd:
            ofInterest += 1
    corrected = 0
    predicted = 0
    for ins in data:
        if SIGN * ins[1] > SIGN * threshold and ins[0] != noneInd:
            predicted += 1
            if ins[0] == ins[2][0]:
                corrected += 1
    f1score = (2.0 * corrected / (ofInterest + predicted))
    recall = (1.0 * corrected / ofInterest)
    precision = (1.0 * corrected / (predicted + 1e-8))

    print(f1score, recall, precision)
    print(corrected, ofInterest, predicted)

    pre_ind_cutoff = pre_ind.clone()
    for ind in range(0, len(pre_ind)):
        ins = data[ind]
        if SIGN * ins[1] > SIGN * threshold:
            pre_ind_cutoff[ind] = pre_ind[ind]
        else:
            pre_ind_cutoff[ind] = noneInd

    return pre_ind_cutoff


def SampleBias(model, packer, label_list, feature_list, ratio, type, cvnum=100):
    """Sample a ratio of test set as clean dev, estimate bias with clean dev, and evaluate the adjusted model"""
    assert type=='fix' or type=='set'
    results = []
    none_ind = model.opt['none_ind']
    for i in range(cvnum):
        labels, feature = shuffle_data(label_list, feature_list)
        eva_n = int(ratio * len(label_list))

        eva_label = labels[:eva_n]
        eva_label_distribution = get_distribution_from_list(eva_label, max_index=model.type_size)

        test_label = labels[eva_n:]
        test_feature = feature[eva_n:]

        fl_t, of_t = packer.repack_eva(test_feature)

        model.eval()
        ### fix bias (train: b = p_train, test: b = p_test)
        if type == 'fix':
            delta_bias = eva_label_distribution
        ### set bias (b' = b - p_train + p_test)
        else:
            delta_bias = np.array(eva_label_distribution) / np.array(model.opt['label_distribution'])
        scores = model.test_with_bias(fl_t, of_t, delta_bias)

        ind = calcInd(scores)

        f1score, recall, precision = eval_score(ind.data, test_label, none_ind)
        results.append([f1score, recall, precision])

    return np.average(results, axis=0)