'''Re-organize data into bag-level setting for models using selective attention'''
import json
import random
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default='./data/neural/KBP')
parser.add_argument('--out_dir', type=str, default='./data/neural_att/KBP')
parser.add_argument('--split_ratio', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=99)
args = parser.parse_args()
opt = vars(args)


def reorg_data(instances, is_train=True):
	hash = {}
	for instance in instances:
		token = instance['token']
		ss, se = instance['subj_start'], instance['subj_end']
		os, oe = instance['obj_start'], instance['obj_end']
		subj = '_'.join(token[ss:se + 1])
		obj = '_'.join(token[os:oe + 1])
		relname = instance['relation']
		if relname in rel2id:
			relation = rel2id[relname]
			if is_train:
				key = subj + "#" + obj + "#" + str(relation)
			else:
				key = subj + "#" + obj
			if not key in hash:
				hash[key] = []
			hash[key].append(instance)
	data = list(hash.items())
	return data


def load_rel2id(fname):
	with open(fname, 'r') as f:
		relation2id = json.load(f)
		return relation2id


def split(data, devprop):
	datasize = len(data)
	devSize = int(datasize * devprop)
	# shuffle
	indices = list(range(datasize))
	random.shuffle(indices)
	data = [data[i] for i in indices]

	print('Original Train Size: ', datasize)
	print('Dev Set Size: ', devSize)
	print('New Train Set Size:', datasize - devSize)

	dev = data[:devSize]
	train = data[devSize:]

	return train, dev


def no_bag(data):
	instances = []
	for key, bag in data:
		instances += bag
	return instances


if __name__ == '__main__':
	assert opt['in_dir'] != opt['out_dir']
	if not os.path.exists(opt['out_dir']):
		os.makedirs(opt['out_dir'])

	rel2id = load_rel2id(os.path.join(opt['in_dir'], 'relation2id.json'))

	# merge train and dev because they should be split in bag-level
	with open(os.path.join(opt['in_dir'], 'train.json'), 'r') as fin:
		train_instances = json.load(fin)
	with open(os.path.join(opt['in_dir'], 'dev.json'), 'r') as fin:
		dev_instances = json.load(fin)
	instances = train_instances + dev_instances

	data = reorg_data(instances)
	train, dev = split(data, opt['split_ratio'])
	# still tune the model in sentence level.
	dev_new = no_bag(dev)

	with open(os.path.join(opt['out_dir'], 'train.json'), 'w') as fout:
		json.dump(train, fout)
	with open(os.path.join(opt['out_dir'], 'dev.json'), 'w') as fout:
		json.dump(dev_new, fout)
	shutil.copyfile(os.path.join(opt['in_dir'], 'test.json'), os.path.join(opt['out_dir'], 'test.json'))
	shutil.copyfile(os.path.join(opt['in_dir'], 'relation2id.json'), os.path.join(opt['out_dir'], 'relation2id.json'))
