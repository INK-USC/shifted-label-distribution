"""
Split train instances into train and dev set (for KBP and NYT),
Save to train_split.json and dev.json
"""
__author__ = 'Qinyuan Ye'
import sys
import random
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='KBP', help='name of the dataset.')
parser.add_argument('--ratio', type=float, default=0.1, help='ratio of dev set.')
parser.add_argument('--seed', type=int, default=1234, help='random seed.')

args = parser.parse_args()
opt = vars(args)

if __name__ == "__main__":
    dataset = opt['dataset']
    ratio = opt['ratio']
    random.seed(opt['seed'])

    dir = 'data/source/%s' % dataset
    original_train_json = dir + '/train.json'
    train_json = dir + '/train_split.json'
    dev_json = dir + '/dev.json'

    if dataset == 'TACRED':
        print('TACRED has a provided dev set, skip splitting ...')
        copyfile(original_train_json, train_json)
        exit(0)

    fin = open(original_train_json, 'r')
    lines = fin.readlines()
    dev_size = int(ratio * len(lines))

    random.shuffle(lines)

    dev = lines[:dev_size]
    train_split = lines[dev_size:]

    fout1 = open(dev_json, 'w')
    fout1.writelines(dev)

    fout2 = open(train_json, 'w')
    fout2.writelines(train_split)
