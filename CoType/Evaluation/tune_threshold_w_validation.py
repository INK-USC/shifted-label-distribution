__author__ = 'xiang'

import sys, os
from collections import  defaultdict
from emb_prediction import *
from evaluation import *
import pickle

def min_max_nomalization(prediction):
    min_val = sys.maxint
    max_val = -sys.maxint
    prediction_normalized = defaultdict(tuple)
    for i in prediction:
        if prediction[i][1] < min_val:
            min_val = prediction[i][1]
        if prediction[i][1] > max_val:
            max_val = prediction[i][1]
    for i in prediction:
        score_normalized = (prediction[i][1] - min_val) / (max_val - min_val + 1e-8)
        prediction_normalized[i] = (prediction[i][0], score_normalized)
    return prediction_normalized

def evaluate_threshold(_threshold, ground_truth):
    # print 'threshold = ', _threshold
    prediction_cutoff = defaultdict(set)
    for i in prediction:
        if prediction[i][1] > _threshold:
            prediction_cutoff[i] = set([prediction[i][0]])
    result = evaluate_rm(prediction_cutoff, ground_truth)
    # print result
    return result

def evaluate_threshold_neg(_threshold, ground_truth, none_label_index):
    # print 'threshold = ', _threshold
    prediction_cutoff = defaultdict(set)
    for i in prediction:
        if prediction[i][1] > _threshold:
            prediction_cutoff[i] = set([prediction[i][0]])
    result = evaluate_rm_neg(prediction_cutoff, ground_truth, none_label_index)
    # print result
    return result

def tune_threshold(_threshold_list, ground_truth, none_label_index):
    result = defaultdict(tuple)
    for _threshold in _threshold_list:
        if none_label_index == None:
            result[_threshold] = evaluate_threshold(_threshold, ground_truth)
        else:
            result[_threshold] = evaluate_threshold_neg(_threshold, ground_truth, none_label_index)
    return result

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

    if len(sys.argv) != 8:
        print 'Usage: tune_threshold.py -TASK (classifer/extract) -DATA(KBP/NYT/BioInfer) -MODE (emb) -METHOD(retype) -SIM(cosine/dot) -lr -iter'
        exit(-1)

    # do prediction here
    _task = sys.argv[1]
    _data = sys.argv[2]
    _mode = sys.argv[3]
    _method = sys.argv[4]
    _sim_func = sys.argv[5]
    _lr = float(sys.argv[6])
    _iter = float(sys.argv[7])

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'
    ground_truth = load_labels(indir + '/mention_type_dev.txt')
    prediction = load_label_score(outdir + '/prediction_' + _mode + '_' + _method + '_' + _sim_func + '_dev.txt')
    file_name = outdir + '/tune_thresholds_' + _mode + '_' + _method + '_' + _sim_func +'_dev.txt'
    # print _data, _mode, _method, _sim_func


    step_size = 1
    prediction = min_max_nomalization(prediction)
    threshold_list = [float(i)/100.0 for i in range(0, 101, step_size)]
    # print threshold_list[0], 'to', threshold_list[-1], ', step-size:', step_size / 100.0

    if _task == 'extract':
        none_label_index = find_none_index(indir + '/type.txt')
        # print '[None] label index: ', none_label_index
        result = tune_threshold(threshold_list, ground_truth, none_label_index)
    else:
        result = tune_threshold(threshold_list, ground_truth, None)


    ### Output
    prec_list = []
    recall_list = []
    f1_list = []
    threshold_list_str = []
    max_f1 = -sys.maxint
    max_prec = -sys.maxint
    max_recall = -sys.maxint
    max_threshold = -sys.maxint
    for _threshold in threshold_list:
        threshold_list_str.append(str(_threshold))
        precision, recall, f1 = result[_threshold]
        prec_list.append(str(precision))
        recall_list.append(str(recall))
        f1_list.append(str(f1))
        if max_f1 < f1:
            max_f1 = f1
            max_prec = precision
            max_recall = recall
            max_threshold = _threshold

    with open(file_name, 'w') as f0:
        for i in range(len(threshold_list_str)):
            if _method == 'line':
                f0.write(recall_list[i] + '\t' + str(float(prec_list[i])) + '\n')
            elif _method == 'retype':
                f0.write(str(float(recall_list[i])) + '\t' + str(float(prec_list[i])) + '\n')
            else:
                f0.write(recall_list[i] + '\t' + prec_list[i] + '\n')

    print 'Best Validation threshold:', max_threshold, '.\tPrecision:', max_prec, '.\tRecall:', max_recall, '.\tF1:', max_f1

    # evaluate on the test set
    ground_truth = load_labels(indir + '/mention_type_test.txt')
    prediction = load_label_score(outdir + '/prediction_' + _mode + '_' + _method + '_' + _sim_func + '.txt')
    if _task == 'extract':
        precision, recall, f1 = evaluate_threshold_neg(max_threshold, ground_truth, none_label_index)
    elif _task == 'classification':
        precision, recall, f1 = evaluate_threshold(max_threshold, ground_truth, None)
    print 'Test \tPrecision:', precision, '.\tRecall:', recall, '.\tF1:', f1
    
    save_log(_data, _lr, _iter, precision, recall, f1)