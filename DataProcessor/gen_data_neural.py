'''
Prepare data for neural models
Author: Maosen Zhang / Qinyuan Ye
'''

from tqdm import tqdm
import os
import json
import argparse

rel_list = ['no_relation']

def find_index(sen_split, word_split):
	index1 = -1
	index2 = -1
	for i in range(len(sen_split)):
		if str(sen_split[i]) == str(word_split[0]):
			flag = True
			k = i
			for j in range(len(word_split)):
				if word_split[j] != sen_split[k]:
					flag = False
				if k < len(sen_split) - 1:
					k+=1
			if flag:
				index1 = i
				index2 = i + len(word_split)
				break
	return index1, index2


def transform_data(data, in_dir, out_dir):
	src_filename = '%s/%s_new.json' % (in_dir, data)
	dest_filename = '%s/%s.json' % (out_dir, data)
	instances = []

	with open(src_filename, 'r') as src_file:
		for idx, line in enumerate(tqdm(src_file.readlines())):
			try:
				sent = json.loads(line.strip())
				tokens = sent['tokens']
				pos_tags = sent['pos']
				ner_tags = sent['ner']
				for rm in sent['relationMentions']:
					start1, end1 = rm['em1Start'], rm['em1End'] - 1
					start2, end2 = rm['em2Start'], rm['em2End'] - 1
					labelset = rm['labels']
					for label in labelset:
						if label == 'None':
							label = 'no_relation'
						instance = {'id': sent['sentId'],
									'relation': label,
									'token': tokens,
									'subj_start': start1,
									'subj_end': end1,
									'obj_start': start2,
									'obj_end': end2,
									'subj_type': ner_tags[start1],
									'obj_type': ner_tags[start2],
									'stanford_pos': pos_tags,
									'stanford_ner': ner_tags
									}
						instances.append(instance)
						if not label in rel_list:
							rel_list.append(label)
			except:
				pass
	with open(dest_filename, 'w') as f:
		json.dump(instances, f)

	if data == 'train':
		out_filename = os.path.join(out_dir, 'relation2id.json')
		d = {item: idx for idx, item in enumerate(rel_list)}
		with open(out_filename, 'w') as fout:
			json.dump(d, fout)

	return instances

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', type=str, default='./data/intermediate/KBP/rm')
	parser.add_argument('--out_dir', type=str, default='./data/neural/KBP')
	args = vars(parser.parse_args())

	if not os.path.isdir(args['out_dir']):
		os.makedirs(args['out_dir'])

	for data in ['train', 'dev', 'test']:
		transform_data(data, args['in_dir'], args['out_dir'])