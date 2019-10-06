import torch
import numpy as np
import random
import model.utils as utils
import model.pack as pack
import model.noCluster as noCluster
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='KBP', help='name of the dataset.')
parser.add_argument('--seed', type=int, default=1234, help='random seed.')
parser.add_argument('--info', type=str, default='KBP_default', help='description.')
parser.add_argument('--save_dir', type=str, default='./dumped_model', help='directory to save model.')
parser.add_argument('--bias', type=str, default='default', help='default/fix, fix=set linear layer bias to label distribution and fix during training.')
parser.add_argument('--data_dir', type=str, default='./data/intermediate')
parser.add_argument('--thres_ratio', type=float, default=0.2, help='proportion of data to tune thres.')
parser.add_argument('--bias_ratio', type=float, default=0.2, help='proportion of data to estimate bias.')

args = parser.parse_args()
opt_new = vars(args)

# set random seed
SEED = opt_new['seed']
print('Using Random Seed: '+str(SEED))
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# load model
save_path = os.path.join(opt_new['save_dir'], opt_new['info'])
save_filename = os.path.join(save_path, 'best_model.pth')
loaded = torch.load(save_filename)
opt = loaded['config']

nocluster = noCluster.noCluster(opt)
nocluster.load_state_dict(loaded['model'], strict=False)

torch.cuda.set_device(0)
nocluster.cuda()
nocluster.eval()

data_dir = os.path.join(opt['data_dir'], opt['dataset'], 'rm')
dev_file = os.path.join(data_dir, 'dev.data')
test_file = os.path.join(data_dir, 'test.data')

feature_file = os.path.join(data_dir, 'feature.txt')
type_file = os.path.join(data_dir, 'type.txt')
type_file_test = os.path.join(data_dir, 'type_test.txt')

none_ind = nocluster.opt['none_ind']
print("None id:", none_ind)

label_distribution = utils.get_distribution(type_file)
label_distribution_test = utils.get_distribution(type_file_test)

packer = pack.repack(opt['input_dropout'], 20, if_cuda=True)

### 1. evaluate dev set
_, _, feature_list, label_list, _ = utils.load_corpus(dev_file)
fl_t, of_t = packer.repack_eva(feature_list)
scores = nocluster(fl_t, of_t)
ind = utils.calcInd(scores)
dev_f1, _, _ = utils.eval_score(ind.data, label_list, none_ind)

### 2. evaluate test set
_, _, feature_list, label_list, _ = utils.load_corpus(test_file)
fl_t, of_t = packer.repack_eva(feature_list)
scores = nocluster(fl_t, of_t)
entropy = utils.calcEntropy(scores)
maxprob = utils.calcMaxProb(scores)
ind = utils.calcInd(scores)
f1score, precision, recall = utils.eval_score(ind.data, label_list, none_ind)

print('Original \tF1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f' %
      (f1score,
       recall,
       precision,
       dev_f1))

### 3. max / entropy thres
if opt_new['bias'] == 'default':
    f1score, recall, precision, dev_f1 = utils.TuneThres(ind.data, maxprob.data, label_list, none_ind, thres_type='max', ratio=opt_new['thres_ratio'])
    print('Max Thres \tF1 = %.4f, recall = %.4f, precision = %.4f, clean val f1 = %.4f' % (f1score, recall, precision, dev_f1))

    f1score, recall, precision, dev_f1 = utils.TuneThres(ind.data, entropy.data, label_list, none_ind, thres_type='entropy', ratio=opt_new['thres_ratio'])
    print('Entropy \tF1 = %.4f, recall = %.4f, precision = %.4f, clean val f1 = %.4f' % (f1score, recall, precision, dev_f1))
### 4. set bias (b' = b - p_train + p_test)
elif opt_new['bias'] == 'set':
    f1score, recall, precision = utils.SampleBias(nocluster, packer, label_list, feature_list, type='set',
                                                  ratio=opt_new['bias_ratio'])
    print('Set bias \tF1 = %.4f, recall = %.4f, precision = %.4f' % (f1score, recall, precision))
### 5. fix bias (train: b = p_train, test: b = p_test)
elif opt_new['bias'] == 'fix':
    f1score, recall, precision = utils.SampleBias(nocluster, packer, label_list, feature_list, type='fix',
                                                  ratio=opt_new['bias_ratio'])
    print('Fix bias \tF1 = %.4f, recall = %.4f, precision = %.4f' % (f1score, recall, precision))

