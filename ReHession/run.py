import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import model.utils as utils
import model.noCluster as noCluster
import model.pack as pack
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='KBP', help='name of the dataset.')
parser.add_argument('--output_dropout', type=float, default=0.5, help='dropout of output layer.')
parser.add_argument('--input_dropout', type=float, default=0.1, help='dropout of input layer.')
parser.add_argument('--batch_size', type=int, default=20, help='batch size.')
parser.add_argument('--emb_len', type=int, default=50, help='dimension of hidden layer.')
parser.add_argument('--bag_weighting', type=str, default='none', help='bag-level attention. (not used)')
parser.add_argument('--seed', type=int, default=1234, help='random seed.')
parser.add_argument('--info', type=str, default='KBP_default', help='description.')
parser.add_argument('--save_dir', type=str, default='./dumped_model', help='directory to save model.')
parser.add_argument('--bias', type=str, default='default', help='default/fix, fix=set linear layer bias to label distribution and fix during training.')
parser.add_argument('--data_dir', type=str, default='./data/intermediate')

args = parser.parse_args()
opt = vars(args)

# Setting random seed
SEED = opt['seed']
print('Using Random Seed: '+str(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# read data
data_dir = os.path.join(opt['data_dir'], opt['dataset'], 'rm')
train_file = os.path.join(data_dir, 'train.data')
dev_file = os.path.join(data_dir, 'dev.data')
test_file = os.path.join(data_dir, 'test.data')
feature_file = os.path.join(data_dir, 'feature.txt')
type_file = os.path.join(data_dir, 'type.txt')
type_file_test = os.path.join(data_dir, 'type_test.txt')
none_ind = utils.get_none_id(type_file)

word_size, pos_embedding_tensor = utils.initialize_embedding(feature_file, opt['emb_len'])
doc_size, type_size, feature_list, label_list, type_list = utils.load_corpus(train_file)
doc_size_test, _, feature_list_test, label_list_test, type_list_test = utils.load_corpus(test_file)
doc_size_dev, _, feature_list_dev, label_list_dev, type_list_dev = utils.load_corpus(dev_file)

# set up configs
opt['none_ind'] = none_ind
opt['label_distribution'] = utils.get_distribution(type_file)
opt['word_size'], opt['type_size'] = word_size, type_size
opt['if_average'] = False
bat_size = opt['batch_size']

# initialize model
nocluster = noCluster.noCluster(opt)
print('embLen, word_size, type_size: ', opt['emb_len'], word_size, type_size)
nocluster.load_word_embedding(pos_embedding_tensor)

optimizer = optim.SGD(nocluster.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
print('Total parameters: ', sum(p.numel() for p in nocluster.parameters() if p.requires_grad))
torch.cuda.set_device(0)
nocluster.cuda()

# set model saving path
save_path = os.path.join(opt['save_dir'], opt['info'])
save_filename = os.path.join(save_path, 'best_model.pth')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# pack features
packer = pack.repack(opt['input_dropout'], 20, True)
fl_t, of_t = packer.repack_eva(feature_list_test)
fl_d, of_d = packer.repack_eva(feature_list_dev)

best_f1, best_recall, best_precision, best_dev_f1 = float('-inf'), 0, 0, float('-inf')

# train
for epoch in range(200):
    print("epoch: " + str(epoch))
    nocluster.train()
    sf_tp, sf_fl = utils.shuffle_data(type_list, feature_list)
    for b_ind in range(0, len(sf_tp), bat_size):
        nocluster.zero_grad()
        if b_ind + bat_size > len(sf_tp):
            b_eind = len(sf_tp)
        else:
            b_eind = b_ind + bat_size
        t_t, _, _, fl_dt, off_dt, scope = packer.repack(sf_fl[b_ind: b_eind], sf_tp[b_ind: b_eind])
        loss = nocluster.loss(t_t, fl_dt, off_dt)
        loss.backward()
        nn.utils.clip_grad_norm(nocluster.parameters(), 5)
        optimizer.step()

    # evaluation mode
    nocluster.eval()
    scores = nocluster(fl_t, of_t)
    ind = utils.calcInd(scores)
    entropy = utils.calcEntropy(scores)

    scores_dev = nocluster(fl_d, of_d)
    ind_dev = utils.calcInd(scores_dev)
    entropy_dev = utils.calcEntropy(scores_dev)

    dev_f1, _, _ = utils.eval_score(ind_dev.data, label_list_dev, none_ind)
    f1score, recall, precision = utils.eval_score(ind.data, label_list_test, none_ind)
    scheduler.step(dev_f1)

    print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f' %
          (f1score,
           recall,
           precision,
           dev_f1))

    if dev_f1 > best_dev_f1:
        best_f1 = f1score
        best_recall = recall
        best_precision = precision
        best_dev_f1 = dev_f1

        params = {
            'model': nocluster.state_dict(),
            'config': opt,
            'epoch': epoch
        }
        torch.save(params, save_filename)

print('Best result: ')
print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f' %
      (best_f1,
       best_recall,
       best_precision,
       best_dev_f1))
