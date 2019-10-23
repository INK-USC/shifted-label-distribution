"""
Generate json format TACRED used for feature-based models
"""

import os
import json
import argparse

rel_list = ['None']

def convert_data(in_filename, out_filename):
	sentDict = {}
	cnt = 0
	with open(in_filename) as fin:
		json_examples = json.load(fin)

	for instance in json_examples:
		if instance['relation'] == 'no_relation':
			instance['relation'] = 'None'
		sent = ' '.join(instance['token'])
		em1 = ' '.join(instance['token'][instance['obj_start']:instance['obj_end'] + 1])
		em2 = ' '.join(instance['token'][instance['subj_start']:instance['subj_end'] + 1])
		if sent in sentDict:
			sentDict[sent]['relationMentions'].append({"em1Text": em1, "em2Text": em2, "label": instance['relation']})
			if em1 not in sentDict[sent]['entityDict']:
				em_cnt = len(sentDict[sent]['entityMentions'])
				sentDict[sent]['entityMentions'].append({
					"start": em_cnt,
					"label": instance['obj_type'],
					"text": em1
				})
				sentDict[sent]['entityDict'].append(em1)
			if em2 not in sentDict[sent]['entityDict']:
				em_cnt = len(sentDict[sent]['entityMentions'])
				sentDict[sent]['entityMentions'].append({
					"start": em_cnt,
					"label": instance['subj_type'],
					"text": em2
				})
				sentDict[sent]['entityDict'].append(em2)
		else:
			s = {
				"articleId": instance['docid'].replace('_', '-'),
				"relationMentions": [{"em1Text": em1, "em2Text": em2,
									  "label": instance['relation']}],
				"entityMentions": [{"text": em1, "start": instance['obj_start'], "label": instance['obj_type']},
								   {"text": em2, "start": instance['subj_start'], "label": instance['subj_type']}],
				"sentId": cnt,
				"entityDict": [em1, em2]
			}
			cnt += 1
			sentDict[sent] = s
			if not instance['relation'] in rel_list:
				rel_list.append(instance['relation'])
	with open(out_filename, 'w') as fout:
		for e in sentDict:
			sentDict[e].pop('entityDict')
			sentDict[e]['sentText'] = e
			fout.write(json.dumps(sentDict[e]) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in_dir', default='./data/neural/TACRED', type=str)
	parser.add_argument('--out_dir', default='./data/source/TACRED', type=str)
	args = parser.parse_args()
	opt = vars(args)

	assert opt['in_dir'] != opt['out_dir']

	if not os.path.exists(opt['out_dir']):
		os.makedirs(opt['out_dir'])

	for split in ['train', 'dev', 'test']:
		in_filename = os.path.join(opt['in_dir'], split + '.json')
		if split == 'train':
			out_filename = os.path.join(opt['out_dir'], 'train_split.json')
		else:
			out_filename = os.path.join(opt['out_dir'], split + '.json')
		convert_data(in_filename, out_filename)

		# relation2id.json is needed in neural models
		if split == 'train':
			rel2id_outfile = os.path.join(opt['in_dir'], 'relation2id.json')
			print(rel_list)
			rel_list[0] = 'no_relation'
			d = {item: idx for idx, item in enumerate(rel_list)}
			with open(rel2id_outfile, 'w') as fout:
				json.dump(d, fout)
