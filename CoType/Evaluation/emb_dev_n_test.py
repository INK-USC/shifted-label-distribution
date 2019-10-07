__author__ = 'xiang'

import sys
from collections import  defaultdict
from evaluation import *
from emb_prediction import *
import pickle

def save_log(data, lr, iter, precision, recall, f1):
    if os.path.isfile('tune_log.pkl'):
        with open('tune_log.pkl', 'rb') as f:
            d = pickle.load(f)
    else:
        d = dict()
    d[(data, lr, iter)] = (precision, recall, f1)
    with open('tune_log.pkl', 'wb') as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print 'Usage: emb_test.py -TASK (classify/extract) \
        -DATA(BioInfer/NYT/Wiki) -METHOD(retype) -SIM(cosine/dot) -THRESHOLD -lr -iter'
        exit(-1)

    _task = sys.argv[1]
    _data = sys.argv[2]
    _method = sys.argv[3]
    _sim_func = sys.argv[4]
    _threshold = float(sys.argv[5])

    # predict dev set
    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '_dev.txt'
    ground_truth = load_labels(indir + '/mention_type_dev.txt')

    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index, val=True) # need modify predict function
    elif _task == 'classify':
        predict(indir, outdir, _method, _sim_func, _threshold, output, None, val=True)
    else:
        print 'wrong TASK argument!'
        exit(1)

    # print dev result, test without threshold
    predictions = load_labels(output)
    print 'Evaluation:'
    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        prec, rec, f1 = evaluate_rm_neg(predictions, ground_truth, none_label_index)
        print 'precision:', prec
        print 'recall:', rec
        print 'f1:', f1
    elif _task == 'classify':
        prec, rec, f1 = evaluate_rm(predictions, ground_truth)
        # print 'accuracy:', prec
    else:
        print 'wrong TASK argument.'
        exit(1)

    # save dev set result (tuning depend on dev)
    # save_log(_data, _lr, _iter, prec, rec, f1)

    # predict test set
    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'

    output = outdir +'/prediction_emb_' + _method + '_' + _sim_func + '.txt'
    ground_truth = load_labels(indir + '/mention_type_test.txt')

    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        predict(indir, outdir, _method, _sim_func, _threshold, output, none_label_index, val=False) # need modify predict function
    elif _task == 'classify':
        predict(indir, outdir, _method, _sim_func, _threshold, output, None, val=False)
    else:
        print 'wrong TASK argument!'
        exit(1)
        
    predictions = load_labels(output)
    print 'Evaluation:'
    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        prec, rec, f1 = evaluate_rm_neg(predictions, ground_truth, none_label_index)
        print 'precision:', prec
        print 'recall:', rec
        print 'f1:', f1
    elif _task == 'classify':
        prec, rec, f1 = evaluate_rm(predictions, ground_truth)
        # print 'accuracy:', prec
    else:
        print 'wrong TASK argument.'
        exit(1)